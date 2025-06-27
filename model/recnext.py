import torch
import torch.nn as nn

from timm.layers import trunc_normal_, DropPath
from timm.models import register_model, create_model, build_model_with_cfg


class RecConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, bias=False, level=2, mode='bilinear'):
        super().__init__()
        self.level = level
        self.mode = mode
        kwargs = {
            'in_channels': in_channels, 
            'out_channels': in_channels, 
            'groups': in_channels,
            'kernel_size': kernel_size, 
            'padding': kernel_size // 2, 
            'bias': bias, 
        }
        self.down = nn.Conv2d(stride=2, **kwargs)
        self.convs = nn.ModuleList([nn.Conv2d(**kwargs) for _ in range(level+1)])

    def forward(self, x):
        i = x
        features = []
        for _ in range(self.level):
            x, s = self.down(x), x.shape[2:]
            features.append((x, s))

        x = 0
        for conv, (f, s) in zip(self.convs, reversed(features)):
            x = nn.functional.interpolate(conv(f + x), size=s, mode=self.mode)
        return self.convs[self.level](i + x)

    '''
    # jit script forward
    def forward(self, x):
        i = x
        features: List[Tuple[torch.Tensor, List[int]]] = []
        for l in range(self.level):
            x, s = self.down(x), x.shape[2:]
            features.append((x, s))

        x = 0
        features.reverse()
        for l, conv in enumerate(self.convs):
            if l == self.level and isinstance(x, torch.Tensor):
                x = conv(i + x)
            else:
                x = nn.functional.interpolate(conv(features[l][0] + x), size=features[l][1], mode=self.mode)
        return x
    '''


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
    def __init__(self, in_channels, mlp_ratio, act_layer=nn.GELU, stage=0):
        super().__init__()
        self.token_mixer = RecConv2d(in_channels, level=4-stage, kernel_size=5) 
        self.norm = nn.BatchNorm2d(in_channels)
        self.channel_mixer = mlp(in_channels, in_channels * mlp_ratio, act_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.channel_mixer(self.norm(self.token_mixer(x))))


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
    def __init__(self, in_channels, out_channels, depth, mlp_ratio, act_layer=nn.GELU, downsample=True, stage=0):
        super().__init__()
        self.downsample = Downsample(in_channels, mlp_ratio, act_layer=act_layer) if downsample else nn.Identity()
        self.blocks = nn.Sequential(*[MetaNeXtBlock(out_channels, mlp_ratio, act_layer=act_layer, stage=stage) for _ in range(depth)])

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


@register_model
def recnext_m0(pretrained=False, **kwargs):
    model_args = dict(embed_dim=(40, 80, 160, 320), depth=(2, 2, 9, 1))
    return _create_recnext("recnext_m0", pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def recnext_m1(pretrained=False, **kwargs):
    model_args = dict(embed_dim=(48, 96, 192, 384), depth=(3, 3, 15, 2))
    return _create_recnext("recnext_m1", pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def recnext_m2(pretrained=False, **kwargs):
    model_args = dict(embed_dim=(56, 112, 224, 448), depth=(3, 3, 15, 2))
    return _create_recnext("recnext_m2", pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def recnext_m3(pretrained=False, **kwargs):
    model_args = dict(embed_dim=(64, 128, 256, 512), depth=(3, 3, 13, 2))
    return _create_recnext("recnext_m3", pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def recnext_m4(pretrained=False, **kwargs):
    # Use drop path when trained without knowledge distillation fixed performance saturation problem.
    model_args = dict(embed_dim=(64, 128, 256, 512), depth=(5, 5, 25, 4), drop_path_rate=0.2)
    return _create_recnext("recnext_m4", pretrained=pretrained, **dict(model_args, **kwargs))

@register_model
def recnext_m5(pretrained=False, **kwargs):
    model_args = dict(embed_dim=(80, 160, 320, 640), depth=(7, 7, 35, 2), drop_path_rate=0.3)
    return _create_recnext("recnext_m5", pretrained=pretrained, **dict(model_args, **kwargs))


if __name__ == "__main__":
    model = create_model("recnext_m1")

    model.eval()
    print(str(model))
    try:
        import pytorch_model_summary
        x = torch.randn(1, 3, 384, 384)
        print(pytorch_model_summary.summary(model, x))
    except ModuleNotFoundError:
        pass

    b, c, h, w = 1, 64, 56, 56
    level = 3
    x = torch.randn(b, c, h, w)
    kernel_size = 5
    model = RecConv2d(c, kernel_size=kernel_size, level=level)
    y = model(x)
    print(y.shape)
    print(model)

    try:
        from fvcore.nn import FlopCountAnalysis
        inputs = torch.randn(b, c, h, w)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters / 1e6)
        flops = FlopCountAnalysis(model, inputs)
        print("flops: ", flops.total() / 1e9)
    except ModuleNotFoundError:
        pass


'''
# downsample and upsample through maxpool and maxunpool
# with higher gpu throughput and less parameters, but not coreml and onnx friendly
class RecConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, bias=False, level=2):
        super().__init__()
        self.level = level
        kwargs = {
            'in_channels': in_channels,
            'out_channels': in_channels,
            'groups': in_channels,
            'kernel_size': kernel_size,
            'padding': kernel_size // 2,
            'bias': bias,
        }
        self.convs = nn.ModuleList([nn.Conv2d(**kwargs) for _ in range(level+1)])

    def forward(self, x):
        i = x
        features = []
        for _ in range(self.level):
            (x, d), s = nn.functional.max_pool2d(x, kernel_size=2, stride=2, return_indices=True), x.shape[2:]
            features.append((x, d, s))

        x = 0
        for conv, (f, d, s) in zip(self.convs, reversed(features)):
            x = nn.functional.max_unpool2d(conv(f + x), indices=d, kernel_size=2, stride=2, output_size=s)
        return self.convs[self.level](i + x)
'''

'''
# bilinear upsample can be replaced by convtranspose2d
# element-wise addition and be replaced by hadamard product
import operator


class RecConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, bias=False, level=2, act_layer=None, agg=operator.add):
        super().__init__()
        self.level = level
        self.agg = agg
        kwargs = {
            'in_channels': in_channels,
            'out_channels': in_channels,
            'groups': in_channels,
            'kernel_size': kernel_size,
            'padding': kernel_size // 2,
            'bias': bias,
        }
        self.down = nn.Conv2d(stride=2, **kwargs)
        self.convs = nn.ModuleList([nn.Conv2d(**kwargs) for _ in range(level+1)])

        # this is the simplest modification, only support resoltions like 256, 384, etc
        kwargs['kernel_size'] = kernel_size + 1
        self.up = nn.ConvTranspose2d(stride=2, **kwargs) if act_layer is None else nn.Sequential(nn.ConvTranspose2d(stride=2, **kwargs), act_layer())

    def forward(self, x):
        i = x
        features = []
        for _ in range(self.level):
            x = self.down(x)
            features.append(x)

        x = None
        for conv, f in zip(self.convs, reversed(features)):
            x = self.up(conv(f if x is None else self.agg(f, x)))
        return self.convs[self.level](self.agg(i, x))
'''

'''
# recursive decomposition on both spatial and channel dimensions
class RecConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, bias=False, level=2, mode='bilinear'):
        super().__init__()
        self.level = level
        self.mode = mode
        kwargs = {'kernel_size': kernel_size, 'padding': kernel_size // 2, 'bias': bias}
        downs = []
        for l in range(level):
            channels = in_channels // (2 ** (l+1))
            downs.append(nn.Conv2d(in_channels=channels, out_channels=channels, groups=channels, stride=2, **kwargs))
        self.downs = nn.ModuleList(downs)

        convs = []
        for l in range(level+1):
            channels = in_channels // (2 ** l)
            convs.append(nn.Conv2d(in_channels=channels, out_channels=channels, groups=channels, **kwargs))
        self.convs = nn.ModuleList(reversed(convs))

    def forward(self, x):
        features = []
        for down in self.downs:
            r, x = torch.chunk(x, 2, dim=1)
            x, s = down(x), x.shape[2:]
            features.append((r, s))

        for conv, (r, s) in zip(self.convs, reversed(features)):
            x = torch.cat([r, nn.functional.interpolate(conv(x), size=s, mode=self.mode)], dim=1)
        return self.convs[self.level](x)
'''

'''
# RecConv Variant A
# recursive decomposition on both spatial and channel dimensions
# downsample and upsample through group convolutions
class RecConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, bias=False, level=2):
        super().__init__()
        self.level = level
        kwargs = {'kernel_size': kernel_size, 'padding': kernel_size // 2, 'bias': bias}
        downs = []
        for l in range(level):
            i_channels = in_channels // (2 ** l)
            o_channels = in_channels // (2 ** (l+1))
            downs.append(nn.Conv2d(in_channels=i_channels, out_channels=o_channels, groups=o_channels, stride=2, **kwargs))
        self.downs = nn.ModuleList(downs)

        convs = []
        for l in range(level+1):
            channels = in_channels // (2 ** l)
            convs.append(nn.Conv2d(in_channels=channels, out_channels=channels, groups=channels, **kwargs))
        self.convs = nn.ModuleList(reversed(convs))

        # this is the simplest modification, only support resoltions like 256, 384, etc
        kwargs['kernel_size'] = kernel_size + 1
        ups = []
        for l in range(level):
            i_channels = in_channels // (2 ** (l+1))
            o_channels = in_channels // (2 ** l)
            ups.append(nn.ConvTranspose2d(in_channels=i_channels, out_channels=o_channels, groups=i_channels, stride=2, **kwargs))
        self.ups = nn.ModuleList(reversed(ups))

    def forward(self, x):
        i = x
        features = []
        for down in self.downs:
            x = down(x)
            features.append(x)

        x = 0
        for conv, up, f in zip(self.convs, self.ups, reversed(features)):
            x = up(conv(f + x))
        return self.convs[self.level](i + x)
'''

'''
# RecConv Variant B
# recursive decomposition on both spatial and channel dimensions
# downsample using channel-wise split, followed by depthwise convolution with a stride of 2
# upsample through channel-wise concatenation
class RecConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=5, bias=False, level=2):
        super().__init__()
        self.level = level
        kwargs = {'kernel_size': kernel_size, 'padding': kernel_size // 2, 'bias': bias}
        downs = []
        for l in range(level):
            channels = in_channels // (2 ** (l+1))
            downs.append(nn.Conv2d(in_channels=channels, out_channels=channels, groups=channels, stride=2, **kwargs))
        self.downs = nn.ModuleList(downs)

        convs = []
        for l in range(level+1):
            channels = in_channels // (2 ** l)
            convs.append(nn.Conv2d(in_channels=channels, out_channels=channels, groups=channels, **kwargs))
        self.convs = nn.ModuleList(reversed(convs))

 .      # this is the simplest modification, only support resoltions like 256, 384, etc
        kwargs['kernel_size'] = kernel_size + 1
        ups = []
        for l in range(level):
            channels = in_channels // (2 ** (l+1))
            ups.append(nn.ConvTranspose2d(in_channels=channels, out_channels=channels, groups=channels, stride=2, **kwargs))
        self.ups = nn.ModuleList(reversed(ups))

    def forward(self, x):
        features = []
        for down in self.downs:
            r, x = torch.chunk(x, 2, dim=1)
            x = down(x)
            features.append(r)

        for conv, up, r in zip(self.convs, self.ups, reversed(features)):
            x = torch.cat([r, up(conv(x))], dim=1)
        return self.convs[self.level](x)
'''

