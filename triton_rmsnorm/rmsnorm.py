"""Part 3: a fused RMSNorm forward kernel in Triton.

RMSNorm (used in Llama, Qwen, Mistral) for one row x of length N:
    y = x / sqrt(mean(x^2) + eps) * w

Why fuse it?  Written in plain PyTorch it is ~5 separate kernels
(square, mean, add eps, rsqrt, two multiplies), and each one reads and writes
the whole tensor to DRAM. RMSNorm does almost no math per byte, so it is
memory-bound: speed = how few bytes you move. The fused kernel reads x once,
reads w once, and writes y once. That is the minimum possible traffic.

Kernel layout: one Triton "program" (think: one thread block) per row.
The whole row fits in registers (BLOCK_N >= N), so the sum of squares and the
normalisation happen without ever going back to DRAM.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_kernel(
    X, W, Y,              # pointers to input, weight, output
    stride_x, stride_y,   # elements between the starts of consecutive rows
    N,                    # row length (hidden size)
    eps,
    BLOCK_N: tl.constexpr,  # power of 2 >= N, fixed at compile time
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N  # BLOCK_N may be bigger than N; masked lanes do nothing

    # 1 read of x. Accumulate in fp32: squaring fp16 values can overflow and
    # summing thousands of them in fp16 loses precision.
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    # 1 write of y, cast back to the input dtype.
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Triton RMSNorm over the last dimension. Works for any leading shape."""
    assert x.shape[-1] == weight.shape[0], "weight must match the last dim"
    orig_shape = x.shape
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    M, N = x2.shape
    y = torch.empty_like(x2)

    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N * x.element_size() > 64 * 1024:
        raise ValueError(f"row of {N} elements is too long for the one-row-per-program kernel")
    # More warps for wider rows so each thread handles a sensible chunk.
    num_warps = min(max(BLOCK_N // 256, 1), 16)

    _rmsnorm_fwd_kernel[(M,)](
        x2, weight, y, x2.stride(0), y.stride(0), N, eps,
        BLOCK_N=BLOCK_N, num_warps=num_warps,
    )
    return y.reshape(orig_shape)


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Plain PyTorch version, computed in fp32 like the kernel. Several kernels."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()
    return y.to(x.dtype)
