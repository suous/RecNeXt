# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Demystify Mamba in Vision: A Linear Attention Perspective
# Modified by Dongchen Han
# -----------------------------------------------------------------------
# RecConv: Efficient Recursive Convolutions for Multi-Frequency Representations 
# Modified for abaltion study
# -----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class LinearAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=qkv_bias, groups=2)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        s = n ** -0.5

        qk = self.elu(self.qk(x)) + 1.0 # [b, 2*c, h, w]
        q, k = qk.chunk(2, dim=1)       # [b,   c, h, w]
        q_rope = self.rope(q.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).reshape(b, self.num_heads, self.head_dim, n)
        k_rope = self.rope(k.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).reshape(b, self.num_heads, self.head_dim, n)

        q = q.reshape(b, self.num_heads, self.head_dim, n)
        k = k.reshape(b, self.num_heads, self.head_dim, n)
        v = x

        z = 1 / (q.transpose(-1, -2) @ k.mean(dim=-1, keepdim=True) + 1e-6)                    # [b, num_heads, n, 1]
        kv = (k_rope*s) @ (v.reshape(b, self.num_heads, self.head_dim, n).transpose(-1, -2)*s) # [b, num_heads, head_dim, head_dim]
        x = q_rope.transpose(-1, -2) @ kv * z                                                  # [b, num_heads, n, head_dim]

        return x.transpose(-1, -2).reshape(b, c, h, w) + self.lepe(v)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


class RecAttn2d(nn.Module):
    def __init__(self, in_channels, input_resolution, num_heads, qkv_bias=True, mode='nearest'):
        super().__init__()
        down_channels = in_channels 
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, down_channels, kernel_size=5, padding=2, stride=2, groups=down_channels),
            LinearAttention(dim=down_channels, input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2), num_heads=num_heads, qkv_bias=qkv_bias),
            nn.Upsample(scale_factor=2, mode=mode)
        )
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)

    def forward(self, x):
        return self.conv(x + self.down(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class MLLABlock(nn.Module):
    def __init__(self, dim, input_resolution, level, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=False, kernel_size=5, expansion_ratio=2):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.level = level
        self.downsample = downsample
        stride = 2 if downsample else 1
        self.cpe1 = nn.Conv2d(dim, stride * dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, stride=stride)

        dim = stride * dim
        hidden = int(dim * expansion_ratio)
        g_dim = c_dim = hidden // 2

        self.norm1 = norm_layer(dim)
        self.i_proj = nn.Linear(dim, hidden)
        self.o_proj = nn.Conv2d(g_dim, dim, kernel_size=1)

        self.act = nn.SiLU()
        self.agg = RecAttn2d(c_dim, input_resolution=(input_resolution[0] // stride, input_resolution[1] // stride), num_heads=num_heads, qkv_bias=qkv_bias) 

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.cpe1(x) if self.downsample else x + self.cpe1(x)
        g, c = self.i_proj(self.norm1(x.permute(0,2,3,1))).permute(0,3,1,2).chunk(2, dim=1)
        x = x + self.drop_path(self.o_proj(self.act(g) * self.agg(c)))
        x = x + self.cpe2(x)
        return x + self.drop_path(self.mlp(self.norm2(x.permute(0,2,3,1))).permute(0,3,1,2))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, mlp_ratio={self.mlp_ratio}, level={self.level}, downsample={self.downsample}"


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, level, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=False, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.level = level
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        kwargs = {
            'dim': dim, 
            'input_resolution': input_resolution, 
            'level': level, 
            'num_heads': num_heads, 
            'mlp_ratio': mlp_ratio, 
            'qkv_bias': qkv_bias, 
            'drop': drop, 
            'norm_layer': norm_layer
        }
        # build blocks
        self.blocks = nn.ModuleList([MLLABlock(drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, downsample=False, **kwargs) for i in range(depth)])
        # patch merging layer
        kwargs['level'] = kwargs['level'] - 1
        self.downsample = MLLABlock(drop_path=drop_path[-1] if isinstance(drop_path, list) else drop_path, downsample=True, **kwargs) if downsample else nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return self.downsample(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, level={self.level}, depth={self.depth}"


class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        return x


class MLLA(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               level=1-i_layer,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=i_layer < self.num_layers - 1,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        return self.norm(x.mean([2, 3])) # (B, C, H, W) -> (B, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
