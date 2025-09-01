import time
import argparse

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention


class Elu(nn.Module):
    def forward(self, x):
        return nn.functional.elu(x) + 1.0


class Softplus(nn.Module):
    def forward(self, x):
        # beta=3.5 is an empirical value for higher correlation with scaled dot product attention.
        return nn.functional.softplus(x, beta=3.5)


class LinearAttention3(nn.Module):
    """
    The computational complexity is quadratic relative to the sequence length.
    Complexity: O(c n^2)
    x_{\text{out}}^{(3)} = \frac{q^\top k}{\text{mean}(q^\top k, \text{dim}=-1) + \epsilon} \cdot v^\top= \frac{q^\top k v^\top}{\text{mean}(q^\top k, \text{dim}=-1) + \epsilon}

    Example:
        q = torch.rand(4, 2) # [c, n]
        k = torch.rand(4, 2) # [c, n]
        v = torch.rand(4, 2) # [c, n]

        qk = q.T @ k                               # [n, c] @ [c, n] -> [n, n]
        u = qk @ v.T                               # [n, n] @ [n, c] -> [n, c]
        x = u / qk.mean(dim=-1, keepdims=True)     # [n, c] / [n, 1] -> [n, c]
    """

    def __init__(self, dim, kernel=Elu(), **kwargs):
        super().__init__()
        self.head_dim = dim // 2
        self.qk = nn.Conv2d(dim, dim, kernel_size=1)
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.kernel = kernel

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        s = n**-0.5

        qk = self.kernel(self.qk(x))
        (q, k), v = qk.view(b, 2, self.head_dim, n).unbind(dim=1), x

        qk = q.transpose(-1, -2) @ k                             # [b, n, n]
        qk = qk / (qk.mean(dim=-1, keepdim=True) + 1e-6)         # [b, n, n]
        x = (qk * s) @ (v.view(b, -1, n).transpose(-1, -2) * s)  # [b, n, c]

        return x.transpose(-1, -2).reshape(b, c, h, w) + self.pe(v)


class LinearAttention4(nn.Module):
    """
    The computational complexity is quadratic relative to the channel dimension.
    Complexity: O(n c^2)
    x_{\text{out}}^{(4)} = \frac{q^\top}{q^\top \text{mean}(k, \text{dim}=-1) + \epsilon} \cdot k v^\top = \frac{q^\top k v^\top}{q^\top \cdot \text{mean}(k, \text{dim}=-1) + \epsilon}

    Example:
        q = torch.rand(4, 2) # [c, n]
        k = torch.rand(4, 2) # [c, n]
        v = torch.rand(4, 2) # [c, n]

        u = q.T @ (k @ v.T)                        # [n, c] @ ([c, n] @ [n, c]) -> [n, c]
        d = q.T @ k.mean(dim=-1, keepdims=True)    # [n, c] @ [c, 1] -> [n, 1]
        x = u / d                                  # [n, c] / [n, 1] -> [n, c]
    """

    def __init__(self, dim, kernel=Elu(), **kwargs):
        super().__init__()
        self.head_dim = dim // 2
        self.qk = nn.Conv2d(dim, dim, kernel_size=1)
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.kernel = kernel

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        s = n**-0.5

        qk = self.kernel(self.qk(x))
        (q, k), v = qk.view(b, 2, self.head_dim, n).unbind(dim=1), x

        q_t = q.transpose(-1, -2)                                   # [b, n, c]
        kv = (k * s) @ (v.view(b, -1, n).transpose(-1, -2) * s)     # [b, c, c]
        x = q_t @ kv / (q_t @ k.mean(dim=-1, keepdim=True) + 1e-6)  # [b, n, c]

        return x.transpose(-1, -2).reshape(b, c, h, w) + self.pe(v)


class Attention(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.head_dim = dim // 2
        self.qk = nn.Conv2d(dim, dim, kernel_size=1, groups=1)
        self.pe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w

        (q, k), v = self.qk(x).view(b, 2, 1, self.head_dim, n).transpose(-1, -2).unbind(dim=1), x
        x = scaled_dot_product_attention(q, k, v.view(b, 1, -1, n).transpose(-1, -2))

        return x.transpose(-1, -2).reshape(b, c, h, w) + self.pe(v)


def calculate_similarity(out1, out2, name1, name2):
    """Calculate similarity metrics between two outputs."""
    mse = torch.nn.functional.mse_loss(out1, out2).item()
    mae = torch.abs(out1 - out2).mean().item()
    cosine_sim = torch.nn.functional.cosine_similarity(out1.flatten(), out2.flatten(), dim=0).item()
    max_diff = torch.max(torch.abs(out1 - out2)).item()

    print(f"\n=== Similarity between {name1} and {name2} ===")
    print(f"MSE (Mean Squared Error): {mse:.6f}")
    print(f"MAE (Mean Absolute Error): {mae:.6f}")
    print(f"Cosine Similarity: {cosine_sim:.6f}")
    print(f"Max Absolute Difference: {max_diff:.6f}")
    if torch.allclose(out1, out2, atol=1e-3):
        print(f"✅ {name1} and {name2} outputs are numerically equivalent (within tolerance)")
    else:
        print(f"❌ {name1} and {name2} outputs are NOT numerically equivalent")
        print( "Note: Small differences are expected due to different attention implementations")

    return {
        "mse": mse,
        "mae": mae,
        "cosine_sim": cosine_sim,
        "max_diff": max_diff,
        "is_close": torch.allclose(out1, out2, atol=1e-3),
    }


def test_modules(
    resolution=(8, 8),
    seed=21,
    device=None,
    batch_size=2,
    dim=64,
    num_runs=1000,
    compile=False,
    kernel="softplus",
):
    """Test the three attention modules with specified parameters."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Resolution: {resolution}")
    print(f"Seed: {seed}")
    print(f"Batch size: {batch_size}")
    print(f"Dimension: {dim}")
    print(f"Number of runs: {num_runs}")
    print(f"Pytorch Compiled: {compile}")
    print(f"Kernel: {kernel}")

    kernel = {"elu": Elu(), "softplus": Softplus(), "relu": nn.ReLU()}[kernel]

    height, width = resolution

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    x = torch.randn(batch_size, dim, height, width).to(device)

    # LinearAttention3 and LinearAttention4 are equivalent
    # Attention may run faster with flash attention
    model3 = LinearAttention3(dim, kernel).to(device)
    model4 = LinearAttention4(dim, kernel).to(device)
    model_attention = Attention(dim).to(device)

    model4.load_state_dict(model3.state_dict())
    model_attention.load_state_dict(model3.state_dict())

    if compile:
        model3 = torch.compile(model3)
        model4 = torch.compile(model4)
        model_attention = torch.compile(model_attention)

    # Warmup runs to stabilize performance measurements
    with torch.no_grad():
        for _ in range(10):
            _ = model3(x)
            _ = model4(x)
            _ = model_attention(x)

    # Synchronize before timing
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Test LinearAttention3
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            out3 = model3(x)
    torch.cuda.synchronize() if device.type == "cuda" else None
    time3 = time.time() - start_time

    # Test LinearAttention4
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            out4 = model4(x)
    torch.cuda.synchronize() if device.type == "cuda" else None
    time4 = time.time() - start_time

    # Test Attention
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            out_attention = model_attention(x)
    torch.cuda.synchronize() if device.type == "cuda" else None
    time_attention = time.time() - start_time

    # Print speed test results
    print("\n=== Speed Test Results ===")
    print(f"LinearAttention3: {time3:.4f} sec for {num_runs} runs")
    print(f"LinearAttention4: {time4:.4f} sec for {num_runs} runs")
    print(f"Attention: {time_attention:.4f} sec for {num_runs} runs")
    print(f"Speed ratio (Attention3/Attention4): {time3 / time4:.2f}x")
    print(f"Speed ratio (Attention3/Attention): {time3 / time_attention:.2f}x")
    print(f"Speed ratio (Attention4/Attention): {time4 / time_attention:.2f}x")

    # Get final outputs for similarity comparison
    with torch.no_grad():
        out3 = model3(x)
        out4 = model4(x)
        out_attention = model_attention(x)

    # Calculate pairwise similarities
    sim_3_4 = calculate_similarity(out3, out4, "LinearAttention3", "LinearAttention4")
    sim_3_attention = calculate_similarity(out3, out_attention, "LinearAttention3", "Attention")
    sim_4_attention = calculate_similarity(out4, out_attention, "LinearAttention4", "Attention")

    return {
        "times": {
            "LinearAttention3": time3,
            "LinearAttention4": time4,
            "Attention": time_attention,
        },
        "similarities": {
            "LinearAttention3_vs_LinearAttention4": sim_3_4,
            "LinearAttention3_vs_Attention": sim_3_attention,
            "LinearAttention4_vs_Attention": sim_4_attention,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Test attention modules")
    parser.add_argument("--resolution", type=str, default="8,8", help="Resolution as 'height,width'")
    parser.add_argument("--seed", type=int, default=21, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--dim", type=int, default=64, help="Dimension")
    parser.add_argument("--num-runs", type=int, default=1000, help="Number of runs for speed test")
    parser.add_argument("--compile", action="store_true", help="Compile the model")
    parser.add_argument("--kernel", type=str, default="softplus", choices=["elu", "softplus", "relu"], help="Activation kernel")

    args = parser.parse_args()
    resolution = tuple(map(int, args.resolution.split(",")))
    test_modules(
        resolution=resolution,
        seed=args.seed,
        device=torch.device(args.device) if args.device else None,
        batch_size=args.batch_size,
        dim=args.dim,
        num_runs=args.num_runs,
        compile=args.compile,
        kernel=args.kernel,
    )


if __name__ == "__main__":
    main()
