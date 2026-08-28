"""
Class definitions for different model + a warper class.
"""

from collections.abc import Sequence

import neurite as ne
import torch
import torch.nn.functional as F
from torch import nn

from deepfluxoptique.model_transformer import TransFlowSegUNet, TransMorph2D
from deepfluxoptique.my_utils import spatial_transform


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ndim=2,
        kernel_size=3,
        padding=1,
        activation=True,
        batchnorm=True,
    ):
        super().__init__()
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
        if activation:
            modules.append(nn.ReLU())  # the one used by default in vxm
        if batchnorm:
            modules.append(nn.BatchNorm2d(out_channels))  # None by default in vxm
        self.conv = nn.Sequential(*modules)  # TODO append à sequential directement

    def forward(self, x):
        return self.conv(x)


def downsampling_conv_blocks(
    ndim: int,
    in_channels: int,
    nb_features: Sequence[int],
    kernel_size: int = 3,
) -> tuple[nn.ModuleList, list[int]]:
    """
    Create an `nn.ModuleList` of downsampling conv blocks based the number of features per layer.

    Parameters
    ----------
    ndim : int
        Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
    nb_features : Sequence[int]
        Number of features at each level of the downsampling convs. Use `0` to indicate a
        pass-through level (pool only, no convolution). When `0` is used, the actual output
        channels equal the input channels (channels are preserved).
    kernel_size : int, default=3
        Size of the convolving kernel at each level.

    Returns
    -------
    tuple[nn.ModuleList, list[int]]
        A tuple of (blocks, actual_channels) where:
        - blocks: ModuleList of downsampling blocks (DownsampleConvBlock or DownsamplePassthrough)
        - actual_channels: List of actual output channel counts for each block, accounting for
          pass-through levels where channels are preserved

    Examples
    --------
    >>> # Standard usage
    >>> blocks, actual_ch = downsampling_conv_blocks(
    ...     ndim=2,
    ...     nb_features=[3, 16, 32],
    ...     kernel_size=3,
    ...     normalizations=["batch", "instance", "batch"],
    ...     activations="relu",
    ... )
    >>> actual_ch
    [16, 32]
    """

    # Init container for downsampling convs and track actual channels
    blocks = nn.ModuleList()

    in_ch = in_channels

    # Make downsampling conv block and append to list of them
    for out_ch in nb_features:
        block = ConvBlock(in_ch, out_ch, kernel_size)
        blocks.append(block)

        in_ch = out_ch

    return blocks


def upsampling_conv_blocks(
    ndim: int,
    nb_features: Sequence[int],
    kernel_size: int = 3,
    skip_channels: Sequence[int] | None = None,
    in_channels: int | None = None,
) -> tuple[nn.ModuleList, list[int]]:
    """
    Create an `nn.ModuleList` of upsampling conv blocks based the number of features per layer/
    level.

    Parameters
    ----------
    ndim : int
        Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
    nb_features : Sequence[int]
        Number of features at each upsampling conv block. Use `0` to indicate a pass-through
        level (upsample + skip concat only, no convolution). When `0` is used, the actual
        output channels equal the upsampled channels plus skip channels (if accepts_skip=True).
    kernel_size : int, default=3
        Size of the convolving kernel at each level.
    skip_channels : Sequence[int] or None, default=None
        List of channel counts for skip connections, one per upsampling block. If provided and
        accepts_skip=True, enables asymmetric downsampling/upsampling architectures. If None and
        accepts_skip=True, assumes symmetric architecture (skip channels equal upsampleda_bl
        channels).
    in_channels : int or None, default=None
        Actual input channels to the first upsampling block (from bottleneck). If None, uses
        nb_features[0]. This is needed when nb_features[0] is 0 (pass-through) but the actual
        input comes from a bottleneck layer with non-zero channels.

    Returns
    -------
    nn.ModuleList
        blocks: ModuleList of upsampling blocks (UpsampleConvBlock or UpsamplePassthrough)
    """

    # Init upsampling conv blocks container and track actual channels
    blocks = nn.ModuleList()

    in_ch = in_channels

    # Make upsampling conv block and append to list of them
    for out_ch, skip_ch in zip(nb_features, skip_channels):
        block = ConvBlock(in_ch + skip_ch, out_ch)
        blocks.append(block)

        in_ch = out_ch

    return blocks


class TwoFlowsUNet(nn.Module):
    """
    Flexible UNet with many configuration options.

    Attributes
    ----------
    downsampling_conv_blocks : nn.ModuleList
        Downsampling convolutional blocks.
    lowest_resolution_conv_block : nn.Module
        Central convolutional block at the lowest spatial resolution.
    upsampling_conv_blocks : nn.ModuleList
        Upsampling convolutional blocks.
    out_layer : nn.Module
        Final output layer.
    down_actual_channels : list[int]
        Actual output channel counts for each downsampling block (accounts for pass-through levels).

    Notes
    -----
    `BasicUNet` is derived from the architecture of the UNet described in
    [Olaf Ronneberger](https://arxiv.org/pdf/1505.04597)

    Use `0` in `nb_features` to indicate pass-through levels that skip convolutions:
    - Downsampling `0`: Pool only, no convolution. Channels preserved from input.
    - Upsampling `0`: Upsample + concat skip only, no conv. Output = upsampled + skip channels.

    Examples
    --------
    >>> model = BasicUNet(
    ...     ndim=2,
    ...     in_channels=1,
    ...     out_channels=1,
    ...     nb_features=[16, 32, 64],
    ...     normalizations="instance",
    ...     activations=nn.ReLU,
    ... )
    >>> input_tensor = torch.randn(1, 1, 128, 128)
    >>> output = model(input_tensor)
    >>> output.shape
    torch.Size([1, 1, 128, 128])

    >>> # Skip full-resolution convolutions using 0
    >>> model = BasicUNet(
    ...     ndim=2,
    ...     in_channels=1,
    ...     out_channels=1,
    ...     nb_features=[0, 32, 64],  # Pass-through at full resolution
    ... )
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        nb_features: Sequence[int] | Sequence[Sequence[int]] = (
            16,
            16,
            16,
            16,
            16,
        ),
    ):
        """
        Initialize `BasicUNet`

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions of the input (1, 2, or 3).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        nb_features : Union[Sequence[int], Sequence[Sequence[int]]], default=(16, 16, 16, 16, 16)
            Number of features at each level of the unet. Can be:
            - Single sequence: [16, 32, 64] (symmetric - downsampling uses [16, 32, 64],
              upsampling uses [64, 32, 16])
            - Sequence of sequences: [[downsampling_features], [upsampling_features]] for complete
              asymmetry
            Use `0` to indicate pass-through levels (no convolution):
            - Downsampling: [0, 32, 64] skips conv at first level (pool only, channels preserved)
            - Upsampling: [64, 32, 0] skips conv at last level (upsample + skip concat only)
        padding_mode : {'zeros', 'replicate', 'reflect'}, default='zeros'
            Padding mode for convolutional layers.
        upsample_mode : {'linear', 'transposed', 'nearest'}, default='linear'
            Upsampling mode for upsampling path.
        normalizations : Sequence[Union[Callable, str]], Callable, str, or None, default=None
            Normalization layers to use in each block. Can be a string or a sequence
            of strings specifying normalizations for each layer, or `None` for no normalization.
        activations : Sequence[Union[Callable, str]], Callable, str, or None, default=nn.ReLU
            Activation functions to use in each block. Can be a callable,
            a string, or a sequence of strings/callables.
        order : str, default='ca'
            Order of operations in each convolutional block (e.g., 'ncaca').
        final_activation : Union[str, nn.Module, None], default=None
            Activation function applied after the final output layer.
        skip_connections : bool, default=True
            Enable skip connections to concatenate features from downsampling path with upsampling
            path at matching resolutions.

        Examples
        --------
        >>> # Symmetric UNet (default behavior)
        >>> model = BasicUNet(ndim=2, in_channels=1, out_channels=1, nb_features=[16, 32, 64], activations=nn.ReLU)
        >>> # Asymmetric UNet with different downsampling/upsampling features
        >>> model = BasicUNet(
        ...     ndim=2,
        ...     in_channels=1,
        ...     out_channels=1,
        ...     nb_features=[[16, 32, 64], [128, 64, 32]],  # Lowest resolution: 64 -> 128
        ...     skip_connections=True,  # Fully supports asymmetric architectures!
        ...     activations=nn.ReLU,
        ... )
        >>> # Skip full-resolution convolutions
        >>> model = BasicUNet(
        ...     ndim=2,
        ...     in_channels=1,
        ...     out_channels=1,
        ...     nb_features=[0, 32, 64],  # First level: pool only, no conv
        ... )
        """
        super().__init__()

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Symmetric Unet
        nb_downsampling_features = tuple(nb_features)
        nb_upsampling_features = tuple(reversed(nb_features))
        nb_skip_channels = tuple(reversed(nb_features))

        # -------- Encoder ---------
        self.downsampling_conv_blocks = downsampling_conv_blocks(
            ndim=ndim,
            in_channels=self.in_channels,
            nb_features=nb_downsampling_features,
        )

        # -------- Bottleneck---------
        bottleneck_channels = nb_downsampling_features[-1]
        self.lowest_resolution_conv_block = ConvBlock(
            ndim=ndim,
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
        )

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        final_channels = nb_upsampling_features[-1]

        # -------- Decoder FWD ---------
        self.upsampling_conv_blocks_fwd = upsampling_conv_blocks(
            ndim=ndim,
            nb_features=nb_upsampling_features,
            in_channels=bottleneck_channels,
            skip_channels=nb_skip_channels,
        )

        self.out_layer_fwd = ConvBlock(
            ndim=ndim,
            in_channels=final_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation=False,
            batchnorm=False,
        )

        # -------- Decoder BWD ---------
        self.upsampling_conv_blocks_bwd = upsampling_conv_blocks(
            ndim=ndim,
            nb_features=nb_upsampling_features,
            in_channels=bottleneck_channels,
            skip_channels=nb_skip_channels,
        )

        self.out_layer_bwd = ConvBlock(
            ndim=ndim,
            in_channels=final_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation=False,
            batchnorm=False,
        )

    def forward(self, feature_tensor: torch.Tensor):
        """
        Forward pass through the `BasicUNet` model.

        Parameters
        ----------
        feature_tensor : torch.Tensor
            Tensor to be passed through the model. Assumed to have batch and channel dimensions.

        Returns
        -------
        torch.Tensor
            Result of forward pass of the model.
        """
        skip_connections = []

        # -------------------------- Downsampling path ----------------------------------------
        for block in self.downsampling_conv_blocks:
            feature_tensor = block(feature_tensor)
            skip_connections.append(feature_tensor)
            feature_tensor = F.max_pool2d(feature_tensor, 2)

        # -------------------------- Bottleneck ----------------------------------------
        feature_tensor = self.lowest_resolution_conv_block(feature_tensor)

        # -------------------------- Upsampling path FWD ----------------------------------------
        feature_tensor_fwd = feature_tensor
        for block, skip in zip(self.upsampling_conv_blocks_fwd, reversed(skip_connections)):
            feature_tensor_fwd = self.upsample(feature_tensor_fwd)
            feature_tensor_fwd = torch.cat([feature_tensor_fwd, skip], dim=1)
            feature_tensor_fwd = block(feature_tensor_fwd)
        feature_tensor_fwd = self.out_layer_fwd(feature_tensor_fwd)

        # -------------------------- Upsampling path BWD ----------------------------------------
        feature_tensor_bwd = feature_tensor
        for block, skip in zip(self.upsampling_conv_blocks_bwd, reversed(skip_connections)):
            feature_tensor_bwd = self.upsample(feature_tensor_bwd)
            feature_tensor_bwd = torch.cat([feature_tensor_bwd, skip], dim=1)
            feature_tensor_bwd = block(feature_tensor_bwd)
        feature_tensor_bwd = self.out_layer_bwd(feature_tensor_bwd)

        return feature_tensor_fwd, feature_tensor_bwd


class FlowSegUNet(nn.Module):
    """
    Flexible UNet with many configuration options.

    Attributes
    ----------
    downsampling_conv_blocks : nn.ModuleList
        Downsampling convolutional blocks.
    lowest_resolution_conv_block : nn.Module
        Central convolutional block at the lowest spatial resolution.
    upsampling_conv_blocks : nn.ModuleList
        Upsampling convolutional blocks.
    out_layer : nn.Module
        Final output layer.
    down_actual_channels : list[int]
        Actual output channel counts for each downsampling block (accounts for pass-through levels).

    Notes
    -----
    `BasicUNet` is derived from the architecture of the UNet described in
    [Olaf Ronneberger](https://arxiv.org/pdf/1505.04597)

    Use `0` in `nb_features` to indicate pass-through levels that skip convolutions:
    - Downsampling `0`: Pool only, no convolution. Channels preserved from input.
    - Upsampling `0`: Upsample + concat skip only, no conv. Output = upsampled + skip channels.

    Examples
    --------
    >>> model = BasicUNet(
    ...     ndim=2,
    ...     in_channels=1,
    ...     out_channels=1,
    ...     nb_features=[16, 32, 64],
    ...     normalizations="instance",
    ...     activations=nn.ReLU,
    ... )
    >>> input_tensor = torch.randn(1, 1, 128, 128)
    >>> output = model(input_tensor)
    >>> output.shape
    torch.Size([1, 1, 128, 128])

    >>> # Skip full-resolution convolutions using 0
    >>> model = BasicUNet(
    ...     ndim=2,
    ...     in_channels=1,
    ...     out_channels=1,
    ...     nb_features=[0, 32, 64],  # Pass-through at full resolution
    ... )
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        nb_class_segmentation: int = 1,
        nb_features: Sequence[int] | Sequence[Sequence[int]] = (
            16,
            32,
            64,
            128,
            256,
        ),
    ):
        """
        Initialize `BasicUNet`

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions of the input (1, 2, or 3).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        nb_features : Union[Sequence[int], Sequence[Sequence[int]]], default=(16, 16, 16, 16, 16)
            Number of features at each level of the unet. Can be:
            - Single sequence: [16, 32, 64] (symmetric - downsampling uses [16, 32, 64],
              upsampling uses [64, 32, 16])
            - Sequence of sequences: [[downsampling_features], [upsampling_features]] for complete
              asymmetry
            Use `0` to indicate pass-through levels (no convolution):
            - Downsampling: [0, 32, 64] skips conv at first level (pool only, channels preserved)
            - Upsampling: [64, 32, 0] skips conv at last level (upsample + skip concat only)
        padding_mode : {'zeros', 'replicate', 'reflect'}, default='zeros'
            Padding mode for convolutional layers.
        upsample_mode : {'linear', 'transposed', 'nearest'}, default='linear'
            Upsampling mode for upsampling path.
        normalizations : Sequence[Union[Callable, str]], Callable, str, or None, default=None
            Normalization layers to use in each block. Can be a string or a sequence
            of strings specifying normalizations for each layer, or `None` for no normalization.
        activations : Sequence[Union[Callable, str]], Callable, str, or None, default=nn.ReLU
            Activation functions to use in each block. Can be a callable,
            a string, or a sequence of strings/callables.
        order : str, default='ca'
            Order of operations in each convolutional block (e.g., 'ncaca').
        final_activation : Union[str, nn.Module, None], default=None
            Activation function applied after the final output layer.
        skip_connections : bool, default=True
            Enable skip connections to concatenate features from downsampling path with upsampling
            path at matching resolutions.

        Examples
        --------
        >>> # Symmetric UNet (default behavior)
        >>> model = BasicUNet(ndim=2, in_channels=1, out_channels=1, nb_features=[16, 32, 64], activations=nn.ReLU)
        >>> # Asymmetric UNet with different downsampling/upsampling features
        >>> model = BasicUNet(
        ...     ndim=2,
        ...     in_channels=1,
        ...     out_channels=1,
        ...     nb_features=[[16, 32, 64], [128, 64, 32]],  # Lowest resolution: 64 -> 128
        ...     skip_connections=True,  # Fully supports asymmetric architectures!
        ...     activations=nn.ReLU,
        ... )
        >>> # Skip full-resolution convolutions
        >>> model = BasicUNet(
        ...     ndim=2,
        ...     in_channels=1,
        ...     out_channels=1,
        ...     nb_features=[0, 32, 64],  # First level: pool only, no conv
        ... )
        """
        super().__init__()

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_class_segmentation = nb_class_segmentation

        # Symmetric Unet
        nb_downsampling_features = tuple(nb_features)
        nb_upsampling_features = tuple(reversed(nb_features))
        nb_skip_channels = tuple(reversed(nb_features))

        # -------- Encoder ---------
        self.downsampling_conv_blocks = downsampling_conv_blocks(
            ndim=ndim,
            in_channels=self.in_channels,
            nb_features=nb_downsampling_features,
        )

        # -------- Bottleneck---------
        bottleneck_channels = nb_downsampling_features[-1]
        self.lowest_resolution_conv_block = ConvBlock(
            ndim=ndim,
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
        )

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        final_channels = nb_upsampling_features[-1]

        # -------- Decoder Flow FWD ---------
        self.upsampling_conv_blocks_fwd = upsampling_conv_blocks(
            ndim=ndim,
            nb_features=nb_upsampling_features,
            in_channels=bottleneck_channels,
            skip_channels=nb_skip_channels,
        )

        self.out_layer_fwd = ConvBlock(
            ndim=ndim,
            in_channels=final_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation=False,
            batchnorm=False,
        )

        # -------- Decoder Segmentation ---------
        self.upsampling_conv_blocks_seg = upsampling_conv_blocks(
            ndim=ndim,
            nb_features=nb_upsampling_features,
            in_channels=bottleneck_channels,
            skip_channels=nb_skip_channels,
        )

        self.out_layer_seg = ConvBlock(
            ndim=ndim,
            in_channels=final_channels,
            out_channels=self.nb_class_segmentation,
            kernel_size=1,
            padding=0,
            activation=False,
            batchnorm=False,
        )

    def forward(self, feature_tensor: torch.Tensor):
        """
        Forward pass through the `BasicUNet` model.

        Parameters
        ----------
        feature_tensor : torch.Tensor
            Tensor to be passed through the model. Assumed to have batch and channel dimensions.

        Returns
        -------
        torch.Tensor
            Result of forward pass of the model.
        """
        skip_connections = []

        # -------------------------- Downsampling path ----------------------------------------
        for block in self.downsampling_conv_blocks:
            feature_tensor = block(feature_tensor)
            skip_connections.append(feature_tensor)
            feature_tensor = F.max_pool2d(feature_tensor, 2)

        # -------------------------- Bottleneck ----------------------------------------
        feature_tensor = self.lowest_resolution_conv_block(feature_tensor)

        # -------------------------- Upsampling path FWD ----------------------------------------
        feature_tensor_fwd = feature_tensor
        for block, skip in zip(self.upsampling_conv_blocks_fwd, reversed(skip_connections)):
            feature_tensor_fwd = self.upsample(feature_tensor_fwd)
            feature_tensor_fwd = torch.cat([feature_tensor_fwd, skip], dim=1)
            feature_tensor_fwd = block(feature_tensor_fwd)
        feature_tensor_fwd = self.out_layer_fwd(feature_tensor_fwd)

        # -------------------------- Upsampling path segmentation ----------------------------------------
        feature_tensor_seg = feature_tensor
        for block, skip in zip(self.upsampling_conv_blocks_seg, reversed(skip_connections)):
            feature_tensor_seg = self.upsample(feature_tensor_seg)
            feature_tensor_seg = torch.cat([feature_tensor_seg, skip], dim=1)
            feature_tensor_seg = block(feature_tensor_seg)
        feature_tensor_seg = self.out_layer_seg(feature_tensor_seg)

        return feature_tensor_fwd, feature_tensor_seg


class WarperVxm(nn.Module):
    """A warper architecture to perform 2D image registration using a flow field.

    Inspired by 'VxmPairWise' from :
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8).
    pp 1788-1800. 2019.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (2).
    source_channels : int
        Number of channels in the source image.
    target_channels : int
        Number of channels in the target image.
    nb_features : List[int], optional
        List of integers specifying the number of features in each
        level of the UNet architecture. Default is `[16, 16, 16, 16, 16]`.
    model_type : str
        Model to use, choices are "voxelmorph", "flowseg", "transflowseg", "transmorph2d". Default is "voxelmorph".
    flow_initializer : float, optional
        Standard deviation for initializing the flow layer weights with a
        normal distribution (mean=0). Default is `1e-5`.
    device : str
        Device : 'cpu' or 'cuda'.

    Attributes:
    ----------
    flow_layer : nn.Module
        A custom convolutional block used to generate the flow field
        from the combined source and target features.

    Methods:
    -------
    forward(source, target, return_warped_source, return_warped_target, return_field_type)
        Combines source and target images, processes them through the UNet and the flow layer,
        and returns the displacement field. Optionally returns warped source and/or
        target images.
    """

    def __init__(
        self,
        ndim: int,
        source_channels: int,
        target_channels: int,
        nb_features: Sequence[int] = (16, 16, 16, 16, 16),
        model_type: str = "voxelmorph",
        # activations: list[Callable | str] | Callable | str | None = nn.ReLU,
        # final_activation: str | nn.Module | None = None,
        flow_initializer: float = 1e-5,
        # integration_steps: int = 5,
        # resize_integrated_fields: bool = False,
        device: str = "cpu",
        # unet_kwargs: dict | None = None,
    ):
        """Initialize the `VxmPairwise`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the input (1, 2, or 3).
        source_channels : int
            Number of channels in the `source_tensor` input to the forward method of this class.
        target_channels : int
            Number of channels in the `target_tensor` input to the forward method of this class.
        nb_features : List[int]
            Number of features at each level of the unet. Must be a list of
            positive integers.
        activations : Union[List[str], str, Callable], optional
            Activation functions to use in each block. Can be a callable,
            a string, or a list of strings/callables.
        integration_steps : int, optional
            Number of scaling and squaring steps for integrating the flow field.
            Default is 5.
        device : str, optional
            Device identifier (e.g., 'cpu' or 'cuda') to place/run the model on.
        unet_kwargs : dict or None, optional
            Additional keyword arguments passed to `neurite.nn.models.BasicUNet`.
        """
        super().__init__()

        self.device = device

        self.model_type = model_type

        if self.model_type == "twoflows":
            self.model = TwoFlowsUNet(
                ndim=ndim,
                in_channels=(source_channels + target_channels),
                out_channels=ndim,
                nb_features=nb_features,
            ).to(self.device)
        elif self.model_type == "flowseg":
            self.model = FlowSegUNet(
                ndim=ndim,
                in_channels=(source_channels + target_channels),
                out_channels=ndim,
                nb_features=nb_features,
            ).to(self.device)
        elif self.model_type == "voxelmorph":
            self.model = ne.nn.models.BasicUNet(
                ndim=ndim,
                in_channels=(source_channels + target_channels),
                out_channels=ndim,
                nb_features=nb_features,
                activations=nn.ReLU,
                final_activation=None,
            ).to(self.device)
        elif self.model_type == "transflowseg":
            self.model = TransFlowSegUNet(
                ndim=ndim,
                in_channels=(source_channels + target_channels),
                out_channels=ndim,
            ).to(self.device)
        elif self.model_type == "transmorph2d":
            self.model = TransMorph2D(
                ndim=ndim,
                in_channels=(source_channels + target_channels),
                out_channels=ndim,
            ).to(self.device)

        self.flow_initializer = flow_initializer
        self._init_flow_layer(ndim, self.flow_initializer)

        # Initialize the spatial transformer
        self.spatial_transformer = spatial_transform.SpatialTransformer().to(self.device)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        return_warped_source: bool = False,
        return_warped_target: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass of `WarperVxm`.

        This forward pass concatenates the `source` and `target` images, processes them with the
        chosen backbone, and uses a flow layer to predict a displacement field.

        Parameters
        ----------
        source : torch.Tensor
            Source image tensor with shape (B, C_source, *spatial_dims).
        target : torch.Tensor
            Target image tensor with shape (B, C_target, *spatial_dims).
            Must have the same spatial dimensions as `source`.
        return_warped_source : bool, optional
            If `True`, include the warped source image in the return tuple. Default is `False`.
        return_warped_target : bool, optional
            If `True`, include the warped target image in the return tuple. Default is `False`.
            Requires `integration_steps > 0` to ensure proper inverse transformation via
            stationary velocity field integration.
        return_field_type : str, optional
            Type of field to return. Options are:
            - 'displacement': Return the integrated displacement field.
            - 'velocity' or 'svf': Return the velocity (stationary velocity field).
            Default is 'displacement'.

        Returns:
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]
            Return values depend on the flags that are set:
            - No flags (default): field (velocity or displacement based on return_field_type)
            - `return_warped_source=True` only: (field, warped_source)
            - `return_warped_target=True` only: (field, warped_target)
            - Both flags: (field, warped_source, warped_target) #TODO

            Where:
            - field shape (B, ndim, *spatial_dims) - velocity or displacement based on
              return_field_type
            - warped_source shape (B, C_source, *spatial_dims)
            - warped_target shape (B, C_target, *spatial_dims)
        """
        res = {}

        # Pass combined features through the model's backbone & flow layer
        combined_features = torch.cat([source, target], dim=1)
        if self.model_type == "twoflows":
            displacement, displacement_bwd = self.model(combined_features)
            res["displacement_bwd"] = displacement_bwd
        elif self.model_type == "voxelmorph":
            unet_out = self.model(combined_features)
            displacement = self.flow_layer(unet_out)
        elif self.model_type == "flowseg" or self.model_type == "transflowseg":
            displacement, segmentation = self.model(combined_features)
            res["source_segmentation"] = segmentation
        elif self.model_type == "transmorph2d":
            displacement = self.model(combined_features)

        if return_warped_source:
            warped_source = self.spatial_transformer(source, displacement)
            res["warped_source"] = warped_source
        # if return_warped_target:
        #     warped_target = self.spatial_transformer(target, neg_displacement)
        #     outputs.append(warped_target)
        res["displacement"] = displacement
        return res

    def _init_flow_layer(self, ndim: int, flow_initializer: float = 1e-5) -> None:
        """Initialize the flow layer with custom weight initialization.

        This layer is a convolutional block that produces a displacement (flow)
        field. The weights of its initial convolution are initialized from a
        normal distribution with mean=0 and std=flow_initializer, and biases
        are set to zero.

        Parameters
        ----------
        ndim : int
            **Spatial** dimensionality of the input (1, 2, or 3).
        features : int
            Number of input and output features for the flow layer.
        flow_initializer : float, optional
            Standard deviation for initializing the flow layer weights with a
            normal distribution (mean=0). Default is `1e-5`.
        """
        # Initialize weights from Normal(mean=0, std=flow_initializer)
        if self.model_type == "twoflows":
            with torch.no_grad():
                torch.nn.init.normal_(
                    self.model.out_layer_fwd.conv[0].weight,
                    mean=0.0,
                    std=flow_initializer,
                )
                # Set the bias term(s) to zero for the first (and only) conv
                if self.model.out_layer_fwd.conv[0].bias is not None:
                    self.model.out_layer_fwd.conv[0].bias.zero_()

            with torch.no_grad():
                torch.nn.init.normal_(
                    self.model.out_layer_bwd.conv[0].weight,
                    mean=0.0,
                    std=flow_initializer,
                )
                # Set the bias term(s) to zero for the first (and only) conv
                if self.model.out_layer_bwd.conv[0].bias is not None:
                    self.model.out_layer_bwd.conv[0].bias.zero_()
        elif self.model_type == "voxelmorph":
            flow_layer = ne.nn.modules.ConvBlock(ndim, 2, 2).to(
                self.device
            )  # reproduit VxmPairWise mais est ce que utile?
            # Initialize weights from Normal(mean=0, std=flow_initializer)
            with torch.no_grad():
                torch.nn.init.normal_(flow_layer.conv0.weight, mean=0.0, std=flow_initializer)
                # Set the bias term(s) to zero for the first (and only) conv
                if flow_layer.conv0.bias is not None:
                    flow_layer.conv0.bias.zero_()
            # Register the flow layer as a submodule
            self.add_module("flow_layer", flow_layer)
        elif self.model_type == "flowseg":
            with torch.no_grad():
                torch.nn.init.normal_(
                    self.model.out_layer_fwd.conv[0].weight,
                    mean=0.0,
                    std=flow_initializer,
                )
                # Set the bias term(s) to zero for the first (and only) conv
                if self.model.out_layer_fwd.conv[0].bias is not None:
                    self.model.out_layer_fwd.conv[0].bias.zero_()

        elif self.model_type == "transflowseg" or self.model_type == "transmorph2d":
            with torch.no_grad():
                torch.nn.init.normal_(
                    self.model.reg_head.weight,
                    mean=0.0,
                    std=flow_initializer,
                )
                # Set the bias term(s) to zero for the first (and only) conv
                if self.model.reg_head.bias is not None:
                    self.model.reg_head.bias.zero_()
