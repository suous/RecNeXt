import torch
import torch.nn as nn
import triton
import triton.language as tl


class LinearAttention1(nn.Module):
    def __init__(self, dim, num_heads, **kwargs):
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
    def __init__(self, dim, num_heads, **kwargs):
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


class LinearAttention3(nn.Module):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()
        self.num_heads = num_heads // 2
        self.head_dim = dim // self.num_heads // 2
        self.qk = nn.Conv2d(dim, dim, kernel_size=1, groups=1)
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        s = n ** -0.5

        qk = nn.functional.elu(self.qk(x)) + 1.0 
        (q, k), v = qk.view(b, 2, self.num_heads, self.head_dim, n).unbind(dim=1), x

        qk = q.transpose(-1, -2) @ k                                         # [b, num_heads, n, n]
        qk = qk / (qk.mean(dim=-1, keepdim=True) + 1e-6)                     # [b, num_heads, n, n]
        x = (qk*s) @ (v.view(b, self.num_heads, -1, n).transpose(-1, -2)*s)  # [b, num_heads, n, head_dim]

        return x.transpose(-1, -2).reshape(b, c, h, w) + self.pe(v)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _attention_kernel(
    Q, K, V,
    output,
    B, H, D, N,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kd, stride_kn,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    SCALE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_D_VOUT: tl.constexpr,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    m_block_start_idx = tl.program_id(2)

    q_ptr = Q + b_idx * stride_qb + h_idx * stride_qh
    k_ptr = K + b_idx * stride_kb + h_idx * stride_kh
    v_ptr = V + b_idx * stride_vb + h_idx * stride_vh

    o_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D_VOUT), dtype=tl.float32)
    qk_sum_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    m_range = m_block_start_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_range < N
    q_offsets = m_range[:, None] * stride_qn + tl.arange(0, BLOCK_SIZE_D)[None, :] * stride_qd
    q = tl.load(q_ptr + q_offsets, mask=m_mask[:, None], other=0.0)

    for n_offset in range(0, N, BLOCK_SIZE_N):
        n_range = n_offset + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_range < N
        
        k_offsets = tl.arange(0, BLOCK_SIZE_D)[:, None] * stride_kd + n_range[None, :] * stride_kn
        k = tl.load(k_ptr + k_offsets, mask=n_mask[None, :], other=0.0)

        v_offsets = n_range[:, None] * stride_vn + tl.arange(0, BLOCK_SIZE_D_VOUT)[None, :] * stride_vd
        v = tl.load(v_ptr + v_offsets, mask=n_mask[:, None], other=0.0)
        
        qk = tl.dot(q, k)
        qk_sum_acc += tl.sum(qk, axis=1)
        o_acc += tl.dot(qk, v)

    qk_sum_acc = qk_sum_acc / N
    o = o_acc * (SCALE * SCALE / (qk_sum_acc[:, None] + 1e-6))
    
    output_offset = b_idx * stride_ob + h_idx * stride_oh
    o_offsets = m_range[:, None] * stride_on + tl.arange(0, BLOCK_SIZE_D_VOUT)[None, :] * stride_od
    tl.store(output + output_offset + o_offsets, o, mask=m_mask[:, None])


class TritonLinearAttention2(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, groups=2)
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w

        qk = nn.functional.elu(self.qk(x)) + 1.0
        q, k = qk.view(b, 2, self.num_heads, self.head_dim, n).unbind(dim=1)
        v = x.view(b, self.num_heads, self.head_dim, n)

        q_t = q.transpose(-1, -2).contiguous()
        v_t = v.transpose(-1, -2).contiguous()

        output = torch.empty_like(q_t)
        
        grid = lambda meta: (b, self.num_heads, triton.cdiv(n, meta['BLOCK_SIZE_M']))
        
        scale = n ** -0.5

        _attention_kernel[grid](
            q_t, k, v_t,
            output,
            b, self.num_heads, self.head_dim, n,
            q_t.stride(0), q_t.stride(1), q_t.stride(2), q_t.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v_t.stride(0), v_t.stride(1), v_t.stride(2), v_t.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            SCALE=scale,
            BLOCK_SIZE_D=self.head_dim,
            BLOCK_SIZE_D_VOUT=self.head_dim,
        )

        return output.transpose(-1, -2).reshape(b, c, h, w) + self.pe(x)


class TritonLinearAttention3(nn.Module):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()
        self.num_heads = num_heads // 2
        self.head_dim = dim // (num_heads)
        self.qk = nn.Conv2d(dim, dim, kernel_size=1, groups=1)
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w

        qk = nn.functional.elu(self.qk(x)) + 1.0
        q, k = qk.view(b, 2, self.num_heads, self.head_dim, n).unbind(dim=1)
        
        v = x.view(b, self.num_heads, self.head_dim * 2, n)

        q_t = q.transpose(-1, -2).contiguous()
        v_t = v.transpose(-1, -2).contiguous()
        
        output = torch.empty_like(v_t)
        
        grid = lambda meta: (b, self.num_heads, triton.cdiv(n, meta['BLOCK_SIZE_M']))
        
        scale = n ** -0.5

        _attention_kernel[grid](
            q_t, k, v_t,
            output,
            b, self.num_heads, self.head_dim, n,
            q_t.stride(0), q_t.stride(1), q_t.stride(2), q_t.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v_t.stride(0), v_t.stride(1), v_t.stride(2), v_t.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            SCALE=scale,
            BLOCK_SIZE_D=self.head_dim,
            BLOCK_SIZE_D_VOUT=self.head_dim * 2,
        )

        return output.transpose(-1, -2).reshape(b, c, h, w) + self.pe(x)


if __name__ == "__main__":
    # LinearAttention1 and LinearAttention2 are equivalent.
    for dim, num_heads, resolution in [
        (64, 2, 4),
        (128, 2, 8),
        (256, 4, 8),
    ]:
        head_dim = dim // num_heads
        seq_len = resolution**2
        print("="*100)
        print(f"dim: {dim}, num_heads: {num_heads}, seq_len: {seq_len}, head_dim: {head_dim}")
        print()
        inputs = torch.randn(1, dim, resolution, resolution).cuda()
        model1 = LinearAttention1(dim, num_heads).to("cuda")
        outputs1 = model1(inputs)
    
        model2 = LinearAttention2(dim, num_heads).to("cuda")
        model2.load_state_dict(model1.state_dict())
        outputs2 = model2(inputs)

        model3 = TritonLinearAttention2(dim, num_heads).to("cuda")
        model3.load_state_dict(model1.state_dict())
        outputs3 = model3(inputs)

        assert torch.allclose(outputs1, outputs2, atol=1e-4)
        assert torch.allclose(outputs2, outputs3, atol=1e-3)

        la3 = LinearAttention3(dim, num_heads).to("cuda")
        la3_opt = la3(inputs)

        tla3 = TritonLinearAttention3(dim, num_heads).to("cuda")
        tla3.load_state_dict(la3.state_dict())
        tla3_opt = tla3(inputs)

        assert torch.allclose(la3_opt, tla3_opt, atol=1e-3)

        print("\nBenchmarking...")
        la3_torch = triton.testing.do_bench(lambda: la3(inputs))
        la3_triton = triton.testing.do_bench(lambda: tla3(inputs))
        print(f"PyTorch: {la3_torch:.4f} ms")
        print(f"Triton:  {la3_triton:.4f} ms")
