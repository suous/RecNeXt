import math
import torch
import torch.nn as nn
from timm.layers import trunc_normal_, DropPath
from timm.models import register_model, create_model, build_model_with_cfg, generate_default_cfgs


class RepVGGDW(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lk = ConvNorm(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.sk = ConvNorm(in_channels, in_channels, kernel_size=1, padding=0, groups=in_channels)
    
    def forward(self, x):
        return self.lk(x) + self.sk(x) + x
    
    @torch.no_grad()
    def fuse(self):
        lk = self.lk.fuse()
        sk = self.sk.fuse()
        
        lk_w, lk_b = lk.weight, lk.bias
        sk_w, sk_b = sk.weight, sk.bias
        
        sk_w = torch.nn.functional.pad(sk_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(lk_w.shape[0], lk_w.shape[1], 1, 1, device=lk_w.device), [1,1,1,1])

        final_conv_w = lk_w + sk_w + identity
        final_conv_b = lk_b + sk_b

        lk.weight.data.copy_(final_conv_w)
        lk.bias.data.copy_(final_conv_b)
        return lk


class LinearAttention1(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.head_dim = dim 
        self.qk = ConvNorm(dim, dim * 2, kernel_size=1, groups=2)
        self.pe = ConvNorm(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        s = n ** -0.5

        qk = nn.functional.elu(self.qk(x)) + 1.0 
        (q, k), v = qk.view(b, 2, self.head_dim, n).unbind(dim=1), x

        q_t = q.transpose(-1, -2)                                       # [b, n, head_dim]
        kv = (k*s) @ (v.view(b, self.head_dim, n).transpose(-1, -2)*s)  # [b, head_dim, head_dim]
        x = q_t @ kv / (q_t @ k.mean(dim=-1, keepdim=True) + 1e-6)      # [b, n, head_dim]

        return x.transpose(-1, -2).reshape(b, c, h, w) + self.pe(v)

    def extra_repr(self):
        return f"head_dim={self.head_dim}"


class LinearAttention2(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.head_dim = dim 
        self.qk = ConvNorm(dim, dim * 2, kernel_size=1, groups=2)
        self.pe = ConvNorm(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        s = n ** -0.5

        qk = nn.functional.elu(self.qk(x)) + 1.0 
        (q, k), v = qk.view(b, 2, self.head_dim, n).unbind(dim=1), x

        qk = q.transpose(-1, -2) @ k                                    # [b, n, n]
        qk = qk / (qk.mean(dim=-1, keepdim=True) + 1e-6)                # [b, n, n]
        x = (qk*s) @ (v.view(b, self.head_dim, n).transpose(-1, -2)*s)  # [b, n, head_dim]

        return x.transpose(-1, -2).reshape(b, c, h, w) + self.pe(v)

    def extra_repr(self):
        return f"head_dim={self.head_dim}"


class LinearAttention3(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.head_dim = dim // 2
        self.qk = ConvNorm(dim, dim, kernel_size=1, groups=1)
        self.pe = ConvNorm(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        s = n ** -0.5

        qk = nn.functional.elu(self.qk(x)) + 1.0 
        (q, k), v = qk.view(b, 2, self.head_dim, n).unbind(dim=1), x

        qk = q.transpose(-1, -2) @ k                         # [b, n, n]
        qk = qk / (qk.mean(dim=-1, keepdim=True) + 1e-6)     # [b, n, n]
        x = (qk*s) @ (v.view(b, -1, n).transpose(-1, -2)*s)  # [b, n, head_dim]

        return x.transpose(-1, -2).reshape(b, c, h, w) + self.pe(v)

    def extra_repr(self):
        return f"head_dim={self.head_dim}"


class RecAttn2d(nn.Module):
    def __init__(self, dim, kernel_size=5, stage=1, mode="nearest"):
        super().__init__()
        self.mode = mode
        LinearAttention = [LinearAttention1, LinearAttention2, LinearAttention2, LinearAttention2][stage]
        self.down = nn.Sequential(
            ConvNorm(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, stride=2, groups=dim),
            LinearAttention(dim=dim),
        )
        self.conv = ConvNorm(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)

    def forward(self, x):
        return self.conv(x + nn.functional.interpolate(self.down(x), size=x.shape[2:], mode=self.mode))


class ConvNorm(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        bn_weight_init=1,
    ):
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias))
        self.add_module("norm", nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.norm.weight, bn_weight_init)
        nn.init.constant_(self.norm.bias, 0)

    @torch.no_grad()
    def fuse(self):
        w = self.norm.weight / (self.norm.running_var + self.norm.eps) ** 0.5
        b = self.norm.bias - w * self.norm.running_mean

        if self.conv.bias is not None:
            b += w * self.conv.bias

        w = w[:, None, None, None] * self.conv.weight

        m = nn.Conv2d(
            w.size(1) * self.conv.groups,
            w.size(0),
            w.shape[2:],
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            device=self.conv.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class NormLinear(nn.Sequential):
    def __init__(self, in_channels, out_channels, bias=True, std=0.02):
        super().__init__()
        self.add_module("norm", nn.BatchNorm1d(in_channels))
        self.add_module("linear", nn.Linear(in_channels, out_channels, bias=bias))
        trunc_normal_(self.linear.weight, std=std)
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    @torch.no_grad()
    def fuse(self):
        norm, linear = self._modules.values()
        w = norm.weight / (norm.running_var + norm.eps) ** 0.5
        b = norm.bias - self.norm.running_mean * self.norm.weight / (norm.running_var + norm.eps) ** 0.5
        w = linear.weight * w[None, :]
        if linear.bias is None:
            b = b @ self.linear.weight.T
        else:
            b = (linear.weight @ b[:, None]).view(-1) + self.linear.bias
        m = nn.Linear(w.size(1), w.size(0), device=linear.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def mlp(in_channels, hidden_channels, act_layer=nn.GELU):
    hidden_channels = int(hidden_channels)
    return nn.Sequential(
        ConvNorm(in_channels, hidden_channels, kernel_size=1),
        act_layer(),
        ConvNorm(hidden_channels, in_channels, kernel_size=1),
    )


class RecNextStem(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer=nn.GELU, kernel_size=3, stride=2, additional_activation=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        kwargs = {"kernel_size": kernel_size, "stride": stride, "padding": padding}
        self.stem = nn.Sequential(
            ConvNorm(in_channels, out_channels // 4, **kwargs),
            act_layer(),
            ConvNorm(out_channels // 4, out_channels // 2, **kwargs),
            act_layer(),
            ConvNorm(out_channels // 2, out_channels, **kwargs),
            act_layer() if additional_activation else nn.Identity(),
        )

    def forward(self, x):
        return self.stem(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_ratio=2, act_layer=nn.GELU, kernel_size=5, stage=0, drop_path=0):
        super().__init__()
        self.token_mixer = ConvNorm(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size-1) // 2 , stride=2, groups=math.gcd(in_channels, out_channels))
        self.channel_mixer = mlp(out_channels, out_channels * mlp_ratio, act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.token_mixer(x)
        return x + self.drop_path(self.channel_mixer(x))


class RecNextClassifier(nn.Module):
    def __init__(self, dim, num_classes, distillation=False, drop=0.0):
        super().__init__()
        self.head_drop = nn.Dropout(drop)
        self.head = NormLinear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.distillation = distillation
        self.num_classes = num_classes
        self.head_dist = NormLinear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.head_drop(x)
        x1, x2 = self.head(x), self.head_dist(x)
        if self.training and self.distillation and not torch.jit.is_scripting():
            return x1, x2
        else:
            return (x1 + x2) / 2

    @torch.no_grad()
    def fuse(self):
        if not self.num_classes > 0:
            return nn.Identity()
        head = self.head.fuse()
        head_dist = self.head_dist.fuse()
        head.weight += head_dist.weight
        head.bias += head_dist.bias
        head.weight /= 2
        head.bias /= 2
        return head


class PartialChannelOperation(nn.Module):
    def __init__(self, attn, part):
        super().__init__()
        self.attn = attn
        self.part = part
        
    def forward(self, x, x1s=None):
        x1, x2 = x[:, :self.part], x[:, self.part:]
        x1 = self.attn(x1)
        
        if x1s is not None:
            x1s.append(x1)
        
        return torch.cat([x1, x2], dim=1)


class ShareChannelOperation(nn.Module):
    def forward(self, x, x1s=None):
        return x + torch.cat(x1s, dim=1)


class MetaNeXtBlock(nn.Module):
    def __init__(self, in_channels, mlp_ratio, act_layer=nn.GELU, stage=0, block=0, drop_path=0, split_rate=4, is_share_stage=False):
        super().__init__()
        self.is_share_block = is_share_stage and (block + 1) % (split_rate + 1) == 0
        self.rep_mixer = RepVGGDW(in_channels)
        
        if self.is_share_block:
            # share previous partial channel operation
            self.token_mixer = ShareChannelOperation()
        else:
            split_part = in_channels // split_rate
            RecAttn = LinearAttention3 if stage >= 2 else RecAttn2d
            self.token_mixer = PartialChannelOperation(attn=RecAttn(split_part, stage=stage), part=split_part) 
        self.channel_mixer = mlp(in_channels, in_channels * mlp_ratio, act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, x1s=None):
        x = self.rep_mixer(x)
        return x + self.drop_path(self.channel_mixer(self.token_mixer(x, x1s)))


class RecNextStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth, mlp_ratio, act_layer=nn.GELU, downsample=True, stage=0, split_rate=4, drop_path_rates=None, share_stage=3):
        super().__init__()
        drop_path_rates = drop_path_rates or [0.] * depth
        self.not_share_stage = stage < share_stage
        self.downsample = Downsample(in_channels, out_channels, mlp_ratio, act_layer=act_layer, stage=stage, drop_path=drop_path_rates[0]) if downsample else nn.Identity()
        blocks = [MetaNeXtBlock(out_channels, mlp_ratio, act_layer, stage, i, drop_path_rates[i], split_rate, not self.not_share_stage) for i in range(depth)]
        self.blocks = nn.Sequential(*blocks) if self.not_share_stage else nn.ModuleList(blocks)
    
    def forward(self, x):
        x = self.downsample(x)
        if self.not_share_stage:
            return self.blocks(x)
        
        x1s = []
        for block in self.blocks:
            x = block(x, x1s=x1s)
            if block.is_share_block:
                x1s.clear()
        return x


class RecNext(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dim=(48,),
        depth=(2,),
        mlp_ratios=(2,),
        global_pool="avg",
        num_classes=1000,
        act_layer=nn.GELU,
        distillation=False,
        split_rates=(4,),
        drop_rate=0.0,
        drop_path_rate=0.0,
        share_stage=3,
        **kwargs
    ):
        super().__init__()
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        in_channels = embed_dim[0]
        additional_activation = depth[0] == 0
        self.stem = RecNextStem(in_chans, in_channels, act_layer=act_layer, additional_activation=additional_activation)
        stride = 4
        self.feature_info = []
        stages = []
        drop_path_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depth)).split(depth)]
        for i in range(len(embed_dim)):
            downsample = i != 0
            stages.append(
                RecNextStage(
                    in_channels,
                    embed_dim[i],
                    depth[i],
                    mlp_ratio=mlp_ratios[i],
                    act_layer=act_layer,
                    downsample=downsample,
                    stage=i,
                    split_rate=split_rates[i],
                    drop_path_rates=drop_path_rates[i],
                    share_stage=share_stage,
                )
            )
            stage_stride = 2 if downsample else 1
            stride *= stage_stride
            self.feature_info += [dict(num_chs=embed_dim[i], reduction=stride, module=f"stages.{i}")]
            in_channels = embed_dim[i]
        self.stages = nn.Sequential(*stages)

        self.num_features = embed_dim[-1]
        self.head_drop = nn.Dropout(drop_rate)
        self.head = RecNextClassifier(embed_dim[-1], num_classes, distillation)

    def forward_features(self, x):
        return self.stages(self.stem(x))

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = x.mean((2, 3), keepdim=False)
        return self.head(self.head_drop(x))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    @torch.no_grad()
    def fuse(self):
        def fuse_children(net):
            for child_name, child in net.named_children():
                if hasattr(child, "fuse"):
                    fused = child.fuse()
                    setattr(net, child_name, fused)
                    fuse_children(fused)
                else:
                    fuse_children(child)

        fuse_children(self)


def _create_recnext(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop("out_indices", (0, 1, 2, 3))
    model = build_model_with_cfg(
        RecNext,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs,
    )
    return model



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        **kwargs,
    }

default_cfgs = generate_default_cfgs(
    {
        'recnext_t_share_channel.base_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_t_share_channel.base_300e_in1k',
            tag=['base', 'without-distillation']
        ),
        'recnext_s_share_channel.base_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_s_share_channel.base_300e_in1k',
            tag=['base', 'without-distillation']
        ),
        'recnext_b_share_channel.base_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_b_share_channel.base_300e_in1k',
            tag=['base', 'without-distillation']
        ),
        'recnext_t_share_channel.dist_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_t_share_channel.dist_300e_in1k',
            tag=['dist', 'knowledge-distillation']
        ),
        'recnext_s_share_channel.dist_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_s_share_channel.dist_300e_in1k',
            tag=['dist', 'knowledge-distillation']
        ),
        'recnext_b_share_channel.dist_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_b_share_channel.dist_300e_in1k',
            tag=['dist', 'knowledge-distillation']
        ),
    }
)

@register_model
def recnext_t_share_channel(pretrained=False, **kwargs):
    distillation = kwargs.pop('distillation', False)
    variant = 'dist' if distillation else 'base'
    drop_path_rate = 0.0 
    model_args = dict(embed_dim=(64, 128, 256, 512), depth=(0, 2, 8, 10), mlp_ratios=(2, 2, 2, 1.5), drop_path_rate=drop_path_rate, split_rates=(4, 4, 4, 4), share_stage=3)
    return _create_recnext(f'recnext_t_share_channel.{variant}_300e_in1k', pretrained=pretrained, distillation=distillation, **dict(model_args, **kwargs))


@register_model
def recnext_s_share_channel(pretrained=False, **kwargs):
    distillation = kwargs.pop('distillation', False)
    variant = 'dist' if distillation else 'base'
    drop_path_rate = 0.0 if distillation else 0.1
    model_args = dict(embed_dim=(128, 256, 384, 512), depth=(0, 2, 8, 10), mlp_ratios=(2, 2, 2, 1.5), drop_path_rate=drop_path_rate, split_rates=(4, 4, 4, 4), share_stage=3)
    return _create_recnext(f'recnext_s_share_channel.{variant}_300e_in1k', pretrained=pretrained, distillation=distillation, **dict(model_args, **kwargs))


@register_model
def recnext_b_share_channel(pretrained=False, **kwargs):
    distillation = kwargs.pop('distillation', False)
    variant = 'dist' if distillation else 'base'
    drop_path_rate = 0.0 if distillation else 0.2
    model_args = dict(embed_dim=(128, 256, 384, 512), depth=(2, 8, 8, 12), mlp_ratios=(2, 2, 2, 1.5), drop_path_rate=drop_path_rate, split_rates=(4, 4, 4, 4), share_stage=3)
    return _create_recnext(f'recnext_b_share_channel.{variant}_300e_in1k_share_channel', pretrained=pretrained, distillation=distillation, **dict(model_args, **kwargs))


if __name__ == "__main__":
    model = create_model("recnext_s")

    model.eval()
    print(str(model))
    try:
        import pytorch_model_summary
        x = torch.randn(1, 3, 224, 224)
        print(pytorch_model_summary.summary(model, x))
    except ModuleNotFoundError:
        pass

"""
# jit script
from typing import List, Optional


class PartialChannelOperation(nn.Module):
    def __init__(self, attn, part):
        super().__init__()
        self.attn = attn
        self.part = part

    def forward(self, x: torch.Tensor, x1s: Optional[List[torch.Tensor]]) -> torch.Tensor:
        x1, x2 = x[:, :self.part], x[:, self.part:]
        x1 = self.attn(x1)

        if x1s is not None:
            x1s.append(x1)

        return torch.cat([x1, x2], dim=1)


class ShareChannelOperation(nn.Module):
    def forward(self, x: torch.Tensor, x1s: Optional[List[torch.Tensor]]) -> torch.Tensor:
        if x1s is not None:
            return x + torch.cat(x1s, dim=1)
        return x


class MetaNeXtBlock(nn.Module):
    def __init__(self, in_channels, mlp_ratio, act_layer=nn.GELU, stage=0, block=0, drop_path=0, split_rate=4, is_share_stage=False):
        super().__init__()
        self.is_share_block = is_share_stage and (block + 1) % (split_rate + 1) == 0
        self.rep_mixer = RepVGGDW(in_channels)

        if self.is_share_block:
            # share previous partial channel operation
            self.token_mixer = ShareChannelOperation()
        else:
            split_part = in_channels // split_rate
            RecAttn = LinearAttention3 if stage >= 2 else RecAttn2d
            self.token_mixer = PartialChannelOperation(attn=RecAttn(split_part, stage=stage), part=split_part)
        self.channel_mixer = mlp(in_channels, in_channels * mlp_ratio, act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, x1s: Optional[List[torch.Tensor]]) -> torch.Tensor:
        x = self.rep_mixer(x)
        return x + self.drop_path(self.channel_mixer(self.token_mixer(x, x1s)))


class RecNextStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth, mlp_ratio, act_layer=nn.GELU, downsample=True, stage=0, split_rate=4, drop_path_rates=None, share_stage=3):
        super().__init__()
        drop_path_rates = drop_path_rates or [0.] * depth
        self.is_share_stage = stage >= share_stage
        self.downsample = Downsample(in_channels, out_channels, mlp_ratio, act_layer=act_layer, stage=stage, drop_path=drop_path_rates[0]) if downsample else nn.Identity()
        blocks = [MetaNeXtBlock(out_channels, mlp_ratio, act_layer, stage, i, drop_path_rates[i], split_rate, self.is_share_stage) for i in range(depth)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.downsample(x)
        if not self.is_share_stage:
            for block in self.blocks:
                x = block(x, None)
            return x

        x1s: List[torch.Tensor] = []
        for block in self.blocks:
            x = block(x, x1s)
            if block.is_share_block:
                x1s.clear()
        return x
"""