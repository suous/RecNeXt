import time

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


class LinearAttention1(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, groups=2)
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

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
        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, groups=2)
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

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


for dim, num_heads, resolution in [
    (16, 2, 32),
    (64, 4, 16),
    (1024, 8, 8),
    (1024, 16, 8),
    (2048, 4, 4),
]:
    head_dim = dim // num_heads
    seq_len = resolution**2
    print("="*100)
    print(f"dim: {dim}, num_heads: {num_heads}, seq_len: {seq_len}, head_dim: {head_dim}")
    print()
    inputs = torch.randn(1, dim, resolution, resolution)
    model1 = LinearAttention1(dim, num_heads)
    outputs1 = model1(inputs)
    flops1 = FlopCountAnalysis(model1, inputs)
    print("flops1: ", flops1.total() / 1e9)
    print()

    model2 = LinearAttention2(dim, num_heads)
    model2.load_state_dict(model1.state_dict())
    outputs2 = model2(inputs)
    flops2 = FlopCountAnalysis(model2, inputs)
    print("flops2: ", flops2.total() / 1e9)
    print()

    assert torch.allclose(outputs1, outputs2, atol=1e-4)
    if seq_len > head_dim:
        assert flops1.total() <= flops2.total()
    else:
        assert flops1.total() > flops2.total()
