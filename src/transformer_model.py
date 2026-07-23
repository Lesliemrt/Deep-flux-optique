from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import timm


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
        modules = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            )
        ]
        if activation:
            modules.append(nn.ReLU())  # the one used by default in vxm
        if batchnorm:
            modules.append(nn.BatchNorm2d(out_channels))  # None by default in vxm
        self.conv = nn.Sequential(*modules)  # TODO append à sequential directement

    def forward(self, x):
        return self.conv(x)


def upsampling_2conv_blocks(
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
        block1 = ConvBlock(in_ch + skip_ch, out_ch, batchnorm=False)
        block2 = ConvBlock(out_ch, out_ch, batchnorm=False)
        blocks.append(block1)
        blocks.append(block2)

        in_ch = out_ch

    return blocks


# class SwinTransformer(nn.Module):
#     r"""Swin Transformer
#         A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
#           https://arxiv.org/pdf/2103.14030
#     Args:
#         img_size (int | tuple(int)): Input image size. Default 224
#         patch_size (int | tuple(int)): Patch size. Default: 4
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         embed_dim (int): Patch embedding dimension. Default: 96
#         depths (tuple(int)): Depth of each Swin Transformer layer.
#         num_heads (tuple(int)): Number of attention heads in different layers.
#         window_size (tuple): Window size. Default: 7
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
#         drop_rate (float): Dropout rate. Default: 0
#         attn_drop_rate (float): Attention dropout rate. Default: 0
#         drop_path_rate (float): Stochastic depth rate. Default: 0.1
#         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#         patch_norm (bool): If True, add normalization after patch embedding. Default: True
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
#     """

#     def __init__(
#         self,
#         pretrain_img_size=224,
#         patch_size=4,
#         in_chans=3,
#         embed_dim=96,
#         depths=[2, 2, 6, 2],
#         num_heads=[3, 6, 12, 24],
#         window_size=(7, 7, 7),
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         qk_scale=None,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         drop_path_rate=0.2,
#         norm_layer=nn.LayerNorm,
#         patch_norm=True,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=-1,
#         use_checkpoint=False,
#         pat_merg_rf=2,
#     ):
#         super().__init__()
#         self.pretrain_img_size = pretrain_img_size
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.patch_norm = patch_norm
#         self.out_indices = out_indices
#         self.frozen_stages = frozen_stages
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             patch_size=patch_size,
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None,
#         )

#         # absolute position embedding
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         # stochastic depth
#         dpr = [
#             x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
#         ]  # stochastic depth decay rule

#         # build layers
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = BasicLayer(
#                 dim=int(embed_dim * 2**i_layer),
#                 depth=depths[i_layer],
#                 num_heads=num_heads[i_layer],
#                 window_size=window_size,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 rpe=True,
#                 qk_scale=qk_scale,
#                 drop=drop_rate,
#                 attn_drop=attn_drop_rate,
#                 drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
#                 norm_layer=norm_layer,
#                 downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
#                 use_checkpoint=use_checkpoint,
#                 pat_merg_rf=pat_merg_rf,
#             )
#             self.layers.append(layer)

#         num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
#         self.num_features = num_features

#         # add a norm layer for each output
#         for i_layer in out_indices:
#             layer = norm_layer(num_features[i_layer])
#             layer_name = f"norm{i_layer}"
#             self.add_module(layer_name, layer)

#         self._freeze_stages()

#     def _freeze_stages(self):
#         if self.frozen_stages >= 0:
#             self.patch_embed.eval()
#             for param in self.patch_embed.parameters():
#                 param.requires_grad = False

#         if self.frozen_stages >= 1 and self.ape:
#             self.absolute_pos_embed.requires_grad = False

#         if self.frozen_stages >= 2:
#             self.pos_drop.eval()
#             for i in range(0, self.frozen_stages - 1):
#                 m = self.layers[i]
#                 m.eval()
#                 for param in m.parameters():
#                     param.requires_grad = False

#     # def init_weights(self, pretrained=None):
#     #     """Initialize the weights in backbone.
#     #     Args:
#     #         pretrained (str, optional): Path to pre-trained weights.
#     #             Defaults to None.
#     #     """

#     #     def _init_weights(m):
#     #         if isinstance(m, nn.Linear):
#     #             trunc_normal_(m.weight, std=0.02)
#     #             if isinstance(m, nn.Linear) and m.bias is not None:
#     #                 nn.init.constant_(m.bias, 0)
#     #         elif isinstance(m, nn.LayerNorm):
#     #             nn.init.constant_(m.bias, 0)
#     #             nn.init.constant_(m.weight, 1.0)

#     #     if isinstance(pretrained, str):
#     #         self.apply(_init_weights)
#     #     elif pretrained is None:
#     #         self.apply(_init_weights)
#     #     else:
#     #         raise TypeError("pretrained must be a str or None")

#     def forward(self, x):
#         """Forward function."""
#         x = self.patch_embed(x)

#         Wh, Ww = x.size(2), x.size(3)

#         if self.ape:
#             # interpolate the position embedding to the corresponding size
#             absolute_pos_embed = nnf.interpolate(
#                 self.absolute_pos_embed, size=(Wh, Ww), mode="bilinear"
#             )
#             x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
#         elif self.spe:
#             x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
#         else:
#             x = x.flatten(2).transpose(1, 2)
#         x = self.pos_drop(x)

#         outs = []
#         for i in range(self.num_layers):
#             layer = self.layers[i]
#             x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
#             if i in self.out_indices:
#                 norm_layer = getattr(self, f"norm{i}")
#                 x_out = norm_layer(x_out)

#                 out = (
#                     x_out.view(-1, H, W, self.num_features[i])
#                     .permute(0, 4, 1, 2, 3)
#                     .contiguous()
#                 )
#                 outs.append(out)
#         return outs

#     def train(self, mode=True):
#         """Convert the model into training mode while keep layers freezed."""
#         super(SwinTransformer, self).train(mode)
#         self._freeze_stages()


class TransFlowSegUNet(nn.Module):
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
        nb_class_segmentation: int = 1,
        embed_dim: int = 96,
        reg_head_channels: int = 16,
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
        embed_dim : int
        reg_head_channels : int

        Examples
        --------
        >>> # Symmetric UNet (default behavior)
        >>> model = BasicUNet(
        ...     ndim=2, in_channels=1, out_channels=1,
        ...     nb_features=[16, 32, 64],
        ...     activations=nn.ReLU
        ... )
        """
        super().__init__()

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_class_segmentation = nb_class_segmentation
        self.reg_head_channels = reg_head_channels

        # Symmetric Unet
        nb_features = [
            embed_dim // 2,
            embed_dim,
            embed_dim * 2,
            embed_dim * 4,
        ]  # TODO pb il faut 5 car 5 skip connections mettre 16?
        nb_upsampling_features = tuple(reversed(nb_features))
        nb_skip_channels = tuple(reversed(nb_features))

        # -------- Encoder ---------
        self.avg_pool = nn.AvgPool2d(1, stride=2, padding=1)
        self.convrelu1 = ConvBlock(
            ndim=ndim, in_channels=2, out_channels=embed_dim // 2
        )
        self.convrelu2 = ConvBlock(
            ndim=ndim, in_channels=2, out_channels=self.reg_head_channels
        )

        self.transformer = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=True,
            in_chans=2,
            features_only=True,
            out_indices=[0, 1, 2, 3],
        )

        # -------- Bottleneck---------
        bottleneck_channels = embed_dim * 8
        self.lowest_resolution_conv_block = ConvBlock(
            ndim=ndim,
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
        )

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        final_channels = nb_upsampling_features[-1]

        # -------- Decoder Flow FWD ---------
        self.upsampling_conv_blocks_fwd = upsampling_2conv_blocks(
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

        self.reg_head = nn.Conv2d(
            self.reg_head_channels, out_channels, kernel_size=1, padding=0
        )

        # -------- Decoder Segmentation ---------
        self.upsampling_conv_blocks_seg = upsampling_2conv_blocks(
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

        # -------------------------- Downsampling path ----------------------------------------
        # for block in self.downsampling_conv_blocks:
        #     feature_tensor = block(feature_tensor)
        #     skip_connections.append(feature_tensor)
        #     feature_tensor = F.max_pool2d(feature_tensor, 2)

        f4 = self.convrelu1(self.avg_pool(feature_tensor))
        f5 = self.convrelu2(feature_tensor.clone())

        out = self.transformer(feature_tensor)

        f1 = out[-2]
        f2 = out[-3]
        f3 = out[-4]

        skip_connections = [f1, f2, f3, f4, f5]

        # -------------------------- Bottleneck ----------------------------------------
        feature_tensor = self.lowest_resolution_conv_block(out[-1])

        # -------------------------- Upsampling path FWD ----------------------------------------
        feature_tensor_fwd = feature_tensor
        for block, skip in zip(self.upsampling_conv_blocks_fwd, skip_connections):
            feature_tensor_fwd = self.upsample(feature_tensor_fwd)
            feature_tensor_fwd = torch.cat([feature_tensor_fwd, skip], dim=1)
            feature_tensor_fwd = block(feature_tensor_fwd)
        # feature_tensor_fwd = self.out_layer_fwd(feature_tensor_fwd)

        feature_tensor_fwd = self.reg_head(feature_tensor_fwd)

        # -------------------------- Upsampling path segmentation ----------------------------------------
        feature_tensor_seg = feature_tensor
        for block, skip in zip(self.upsampling_conv_blocks_seg, skip_connections):
            feature_tensor_seg = self.upsample(feature_tensor_seg)
            feature_tensor_seg = torch.cat([feature_tensor_seg, skip], dim=1)
            feature_tensor_seg = block(feature_tensor_seg)
        # feature_tensor_seg = self.out_layer_seg(feature_tensor_seg)
        feature_tensor_seg = self.reg_head(feature_tensor_seg)

        return feature_tensor_fwd, feature_tensor_seg
