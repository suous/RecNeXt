import torch
import torch.nn as nn

from timm.layers import trunc_normal_, DropPath
from timm.models import register_model, create_model, build_model_with_cfg, generate_default_cfgs


class LinearAttention1(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk = ConvNorm(dim, dim * 2, kernel_size=1, groups=2)
        self.pe = ConvNorm(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        s = n ** -0.5

        qk = nn.functional.elu(self.qk(x)) + 1.0 
        (q, k), v = qk.view(b, 2, self.num_heads, self.head_dim, n).unbind(dim=1), x

        q_t = q.transpose(-1, -2)                                                       # [b, num_heads, n, head_dim]
        kv = (k*s) @ (v.view(b, self.num_heads, self.head_dim, n).transpose(-1, -2)*s)  # [b, num_heads, head_dim, head_dim]
        x = q_t @ kv / (q_t @ k.mean(dim=-1, keepdim=True) + 1e-6)                      # [b, num_heads, n, head_dim]

        return x.transpose(-1, -2).reshape(b, c, h, w) + self.pe(v)


class LinearAttention2(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk = ConvNorm(dim, dim * 2, kernel_size=1, groups=2)
        self.pe = ConvNorm(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        s = n ** -0.5

        qk = nn.functional.elu(self.qk(x)) + 1.0 
        (q, k), v = qk.view(b, 2, self.num_heads, self.head_dim, n).unbind(dim=1), x

        qk = q.transpose(-1, -2) @ k                                                    # [b, num_heads, n, n]
        qk = qk / (qk.mean(dim=-1, keepdim=True) + 1e-6)                                # [b, num_heads, n, n]
        x = (qk*s) @ (v.view(b, self.num_heads, self.head_dim, n).transpose(-1, -2)*s)  # [b, num_heads, n, head_dim]

        return x.transpose(-1, -2).reshape(b, c, h, w) + self.pe(v)


class RecAttn2d(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=5, stage=1, mode="nearest"):
        super().__init__()
        self.mode = mode
        # LinearAttention1 and LinearAttention2 are interchangeable
        LinearAttention = LinearAttention2 if stage >= 3 else LinearAttention1
        self.down = nn.Sequential(
            ConvNorm(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, stride=2, groups=dim),
            LinearAttention(dim=dim, num_heads=num_heads),
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
        bias=False,
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
    def __init__(self, in_channels, out_channels, act_layer=nn.GELU, kernel_size=3, stride=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        kwargs = {"kernel_size": kernel_size, "stride": stride, "padding": padding}
        self.stem = nn.Sequential(
            ConvNorm(in_channels, out_channels // 2, **kwargs),
            act_layer(),
            ConvNorm(out_channels // 2, out_channels, **kwargs),
        )

    def forward(self, x):
        return self.stem(x)


class MetaNeXtBlock(nn.Module):
    def __init__(self, in_channels, mlp_ratio, act_layer=nn.GELU, stage=0, drop_path=0):
        super().__init__()
        self.token_mixer = RecAttn2d(in_channels, num_heads=2**(stage+1), stage=stage) 
        self.channel_mixer = mlp(in_channels, in_channels * mlp_ratio, act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.channel_mixer(self.token_mixer(x)))


class Downsample(nn.Module):
    def __init__(self, in_channels, mlp_ratio, act_layer=nn.GELU):
        super().__init__()
        out_channels = in_channels * 2
        self.token_mixer = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, groups=in_channels, stride=2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.channel_mixer = mlp(out_channels, out_channels * mlp_ratio, act_layer=act_layer)

    def forward(self, x):
        x = self.norm(self.token_mixer(x))
        return x + self.channel_mixer(x)


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


class RecNextStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth, mlp_ratio, act_layer=nn.GELU, downsample=True, stage=0, drop_path=0):
        super().__init__()
        self.downsample = Downsample(in_channels, mlp_ratio, act_layer=act_layer) if downsample else nn.Identity()
        self.blocks = nn.Sequential(*[MetaNeXtBlock(out_channels, mlp_ratio, act_layer=act_layer, stage=stage, drop_path=drop_path) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.downsample(x))


class RecNext(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dim=(48,),
        depth=(2,),
        mlp_ratio=2,
        global_pool="avg",
        num_classes=1000,
        act_layer=nn.GELU,
        distillation=False,
        drop_rate=0.0,
        drop_path=0.0,
        **kwargs
    ):
        super().__init__()
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        in_channels = embed_dim[0]
        self.stem = RecNextStem(in_chans, in_channels, act_layer=act_layer)
        stride = 4
        self.feature_info = []
        stages = []
        for i in range(len(embed_dim)):
            downsample = True if i != 0 else False
            stages.append(
                RecNextStage(
                    in_channels,
                    embed_dim[i],
                    depth[i],
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    downsample=downsample,
                    stage=i,
                    drop_path=drop_path,
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
        'recnext_a0.base_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_a0.base_300e_in1k',
            tag=['base', 'without-distillation']
        ),
        'recnext_a1.base_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_a1.base_300e_in1k',
            tag=['base', 'without-distillation']
        ),
        'recnext_a2.base_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_a2.base_300e_in1k',
            tag=['base', 'without-distillation']
        ),
        'recnext_a3.base_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_a3.base_300e_in1k',
            tag=['base', 'without-distillation']
        ),
        'recnext_a4.base_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_a4.base_300e_in1k',
            tag=['base', 'without-distillation']
        ),
        'recnext_a5.base_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_a5.base_300e_in1k',
            tag=['base', 'without-distillation']
        ),
        'recnext_a0.dist_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_a0.dist_300e_in1k',
            tag=['dist', 'knowledge-distillation']
        ),
        'recnext_a1.dist_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_a1.dist_300e_in1k',
            tag=['dist', 'knowledge-distillation']
        ),
        'recnext_a2.dist_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_a2.dist_300e_in1k',
            tag=['dist', 'knowledge-distillation']
        ),
        'recnext_a3.dist_300e_in1k': _cfg(  
            hf_hub_id='suous/recnext_a3.dist_300e_in1k',
            tag=['dist', 'knowledge-distillation']
        ),
        'recnext_a4.dist_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_a4.dist_300e_in1k',
            tag=['dist', 'knowledge-distillation']
        ),
        'recnext_a5.dist_300e_in1k': _cfg(
            hf_hub_id='suous/recnext_a5.dist_300e_in1k',
            tag=['dist', 'knowledge-distillation']
        ),
    }
)


@register_model
def recnext_a0(pretrained=False, **kwargs):
    distillation = kwargs.pop('distillation', False)
    variant = 'dist' if distillation else 'base'
    model_args = dict(embed_dim=(40, 80, 160, 320), depth=(2, 2, 9, 1))
    return _create_recnext(f'recnext_a0.{variant}_300e_in1k', pretrained=pretrained, distillation=distillation, **dict(model_args, **kwargs))

@register_model
def recnext_a1(pretrained=False, **kwargs):
    distillation = kwargs.pop('distillation', False)
    variant = 'dist' if distillation else 'base'
    model_args = dict(embed_dim=(48, 96, 192, 384), depth=(3, 3, 15, 2))
    return _create_recnext(f'recnext_a1.{variant}_300e_in1k', pretrained=pretrained, distillation=distillation, **dict(model_args, **kwargs))

@register_model
def recnext_a2(pretrained=False, **kwargs):
    distillation = kwargs.pop('distillation', False)
    variant = 'dist' if distillation else 'base'
    model_args = dict(embed_dim=(56, 112, 224, 448), depth=(3, 3, 15, 2))
    return _create_recnext(f'recnext_a2.{variant}_300e_in1k', pretrained=pretrained, distillation=distillation, **dict(model_args, **kwargs))

@register_model
def recnext_a3(pretrained=False, **kwargs):
    distillation = kwargs.pop('distillation', False)
    variant = 'dist' if distillation else 'base'
    model_args = dict(embed_dim=(64, 128, 256, 512), depth=(3, 3, 13, 2), mlp_ratio=1.875)
    return _create_recnext(f'recnext_a3.{variant}_300e_in1k', pretrained=pretrained, distillation=distillation, **dict(model_args, **kwargs))

@register_model
def recnext_a4(pretrained=False, **kwargs):
    distillation = kwargs.pop('distillation', False)
    variant = 'dist' if distillation else 'base'
    drop_path = 0.0 if distillation else 0.2
    model_args = dict(embed_dim=(64, 128, 256, 512), depth=(5, 5, 25, 4), mlp_ratio=1.875, drop_path=drop_path)
    return _create_recnext(f'recnext_a4.{variant}_300e_in1k', pretrained=pretrained, distillation=distillation, **dict(model_args, **kwargs))

@register_model
def recnext_a5(pretrained=False, **kwargs):
    distillation = kwargs.pop('distillation', False)
    variant = 'dist' if distillation else 'base'
    drop_path = 0.0 if distillation else 0.3
    model_args = dict(embed_dim=(80, 160, 320, 640), depth=(7, 7, 35, 2), mlp_ratio=1.875, drop_path=drop_path)
    return _create_recnext(f'recnext_a5.{variant}_300e_in1k', pretrained=pretrained, distillation=distillation, **dict(model_args, **kwargs))


if __name__ == "__main__":
    model = create_model("recnext_a1")

    model.eval()
    print(str(model))
    try:
        import pytorch_model_summary
        x = torch.randn(1, 3, 384, 384)
        print(pytorch_model_summary.summary(model, x))
    except ModuleNotFoundError:
        pass

