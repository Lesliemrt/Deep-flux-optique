from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn


class UNet(nn.Module):
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
    ...     ndim=2, in_channels=1, out_channels=1,
    ...     nb_features=[16, 32, 64],
    ...     normalizations='instance', activations=nn.ReLU
    ... )
    >>> input_tensor = torch.randn(1, 1, 128, 128)
    >>> output = model(input_tensor)
    >>> output.shape
    torch.Size([1, 1, 128, 128])

    >>> # Skip full-resolution convolutions using 0
    >>> model = BasicUNet(
    ...     ndim=2, in_channels=1, out_channels=1,
    ...     nb_features=[0, 32, 64],  # Pass-through at full resolution
    ... )
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        nb_features: Union[Sequence[int], Sequence[Sequence[int]]] = (
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
        >>> model = BasicUNet(
        ...     ndim=2, in_channels=1, out_channels=1,
        ...     nb_features=[16, 32, 64],
        ...     activations=nn.ReLU
        ... )
        >>> # Asymmetric UNet with different downsampling/upsampling features
        >>> model = BasicUNet(
        ...     ndim=2, in_channels=1, out_channels=1,
        ...     nb_features=[[16, 32, 64], [128, 64, 32]],  # Lowest resolution: 64 -> 128
        ...     skip_connections=True,  # Fully supports asymmetric architectures!
        ...     activations=nn.ReLU
        ... )
        >>> # Skip full-resolution convolutions
        >>> model = BasicUNet(
        ...     ndim=2, in_channels=1, out_channels=1,
        ...     nb_features=[0, 32, 64],  # First level: pool only, no conv
        ... )
        """
        super().__init__()

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Symmetric
        downsampling_features = tuple(nb_features)
        upsampling_features = tuple(reversed(nb_features))

        assert all(isinstance(f, int) for f in downsampling_features), (
            "All elements in nb_features must be integers"
        )

        # Store feature specs as immutables
        self.downsampling_features = tuple(downsampling_features)
        self.upsampling_features = tuple(upsampling_features)

        # Downsampling conv blocks
        self.downsampling_conv_blocks, down_actual_channels = downsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.nb_features_fwd,
        )

        # Store actual downsampling channels for reference
        self.down_actual_channels = down_actual_channels

        # Convolutional block between downsampling and upsampling arms (lowest resolution)
        down_output_channels = down_actual_channels[-1]

        # Determine first upsampling feature (0 is falsy)
        bottleneck_out_channels = self.upsampling_features[0] or down_output_channels

        self.lowest_resolution_conv_block = ConvBlock(
            ndim=ndim,
            in_channels=down_output_channels,
            out_channels=bottleneck_out_channels,
        )

        # Upsampling convolutional blocks (returns blocks and actual channel counts)
        self.upsampling_conv_blocks, up_actual_channels = upsampling_conv_blocks(
            ndim=ndim,
            nb_features=self.upsampling_features,
            in_channels=bottleneck_out_channels,
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        # Store actual upsampling channels for reference
        self.up_actual_channels = up_actual_channels

        # Final convolutional block - use actual upsampling output channels
        if up_actual_channels:
            up_output_channels = up_actual_channels[-1]
        else:
            up_output_channels = bottleneck_out_channels

        self.out_layer = ConvBlock(
            ndim=ndim,
            in_channels=up_output_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
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
            skip = block(feature_tensor)
            feature_tensor = nn.MaxPool2d(skip)
            skip_connections.append(skip)

        feature_tensor = self.lowest_resolution_conv_block(feature_tensor)

        # -------------------------- Upsampling path FWD ----------------------------------------
        for block, skip in zip(self.upsampling_conv_blocks, reversed(skip_connections)):
            feature_tensor_fwd = self.upsample(feature_tensor)
            feature_tensor_fwd = torch.cat([feature_tensor_fwd, skip], dim=1)
            feature_tensor_fwd = block(feature_tensor_fwd)
            feature_tensor_fwd = self.out_layer(feature_tensor_fwd)

        # -------------------------- Upsampling path BWD ----------------------------------------
        for block, skip in zip(self.upsampling_conv_blocks, reversed(skip_connections)):
            feature_tensor_bwd = self.upsample(feature_tensor)
            feature_tensor_bwd = torch.cat([feature_tensor_bwd, skip], dim=1)
            feature_tensor_bwd = block(feature_tensor_bwd, skip)
            feature_tensor_bwd = self.out_layer(feature_tensor_bwd)

        return feature_tensor_fwd, feature_tensor_bwd


def downsampling_conv_blocks(
    ndim: int,
    nb_features: Sequence[int],
    kernel_size: int = 2,
) -> Tuple[nn.ModuleList, List[int]]:
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
    ...     activations="relu"
    ... )
    >>> actual_ch
    [16, 32]

    >>> # With pass-through (0) at first level
    >>> blocks, actual_ch = downsampling_conv_blocks(
    ...     ndim=2,
    ...     nb_features=[3, 0, 32],  # Skip conv at first level
    ...     return_skip=True
    ... )
    >>> actual_ch
    [3, 32]  # First level preserves input channels (3)
    """

    # Init container for downsampling convs and track actual channels
    blocks = nn.ModuleList()

    # Important for pass through connections
    actual_channels = [nb_features[0]]  # Start with input channels

    # Make downsampling conv block and append to list of them
    for i in range(len(nb_features) - 1):
        in_ch = actual_channels[-1]  # Use actual channels from previous block
        out_ch = nb_features[i + 1]

        block = ConvBlock(in_ch, out_ch, kernel_size)
        blocks.append(block)

        # Track actual output channels (passthrough preserves input channels)
        actual_channels.append(out_ch)

    # Return blocks and actual output channels (excluding input channels)
    return blocks, actual_channels[1:]


def upsampling_conv_blocks(
    ndim: int,
    nb_features: Sequence[int],
    kernel_size: int = 3,
    skip_channels: Union[Sequence[int], None] = None,
    in_channels: Optional[int] = None,
) -> Tuple[nn.ModuleList, List[int]]:
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
        accepts_skip=True, assumes symmetric architecture (skip channels equal upsampled
        channels).
    in_channels : int or None, default=None
        Actual input channels to the first upsampling block (from bottleneck). If None, uses
        nb_features[0]. This is needed when nb_features[0] is 0 (pass-through) but the actual
        input comes from a bottleneck layer with non-zero channels.

    Returns
    -------
    tuple[nn.ModuleList, list[int]]
        A tuple of (blocks, actual_channels) where:
        - blocks: ModuleList of upsampling blocks (UpsampleConvBlock or UpsamplePassthrough)
        - actual_channels: List of actual output channel counts for each block, accounting for
          pass-through levels where output = upsampled + skip channels

    Notes
    -----
    - If `normalizations` or `activations` are a list, they must be the same length as the
    nb_features.

    Examples
    --------
    >>> # Standard usage
    >>> blocks, actual_ch = upsampling_conv_blocks(
    ...     ndim=2,
    ...     nb_features=[32, 16, 4],
    ...     upsample_kernel_size=4,
    ...     normalizations=["batch", "instance", "batch"],
    ...     activations="relu"
    ... )
    >>> actual_ch
    [16, 4, 4]

    >>> # With pass-through (0) at last level
    >>> blocks, actual_ch = upsampling_conv_blocks(
    ...     ndim=2,
    ...     nb_features=[32, 16, 0],  # Pass-through at last level
    ...     skip_channels=[16, 8, 4],
    ...     accepts_skip=True
    ... )
    >>> actual_ch
    [16, 0 + skip, ...]  # Last level: upsampled + skip channels
    """

    # Init upsampling conv blocks container and track actual channels
    blocks = nn.ModuleList()
    # Use in_channels if provided, otherwise use nb_features[0]
    # This handles the case where nb_features[0] is 0 (pass-through) but actual input
    # comes from a bottleneck with non-zero channels
    initial_channels = in_channels if in_channels is not None else nb_features[0]
    actual_channels = [initial_channels]

    # make the number of features for the upsampling conv blocks
    nb_features_extended = [*nb_features, nb_features[-1]]

    # Validate skip_channels length if provided
    num_blocks = len(nb_features_extended) - 1
    assert skip_channels is None or len(skip_channels) == num_blocks, (
        f"skip_channels must have length {num_blocks} "
        f"(one per upsampling block), got {len(skip_channels)}"
    )

    # Make upsampling conv block and append to list of them
    for i in range(num_blocks):
        in_ch = actual_channels[-1]  # Use actual channels from previous block
        out_ch = nb_features_extended[i + 1]
        skip_ch = skip_channels[i]

        block = ConvBlock(in_ch + skip_ch, out_ch)
        blocks.append(block)

        actual_out = in_ch + (skip_ch or in_ch)
        actual_channels.append(actual_out)

    # Return blocks and actual output channels (excluding input channels)
    return blocks, actual_channels[1:]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.Relu(),  # the one used by default in vxm
            nn.BatchNorm2d(out_channels),  # None by default in vxm
        )

    def forward(self, x):
        return self.conv(x)
