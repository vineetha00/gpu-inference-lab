"""Workloads for the roofline study, each with hand-counted FLOPs and bytes.

Arithmetic intensity (AI) = FLOPs / bytes moved to and from DRAM (HBM/GDDR).
    low AI  -> the GPU waits on memory   (memory-bound, left slope of the roof)
    high AI -> the GPU waits on its math (compute-bound, flat part of the roof)

Byte counts are the *ideal* minimum: every input read once, every output
written once. Nsight Compute's dram__bytes tells us what really happened.

The story the points tell:
  * elementwise add and softmax sit far left: nothing to reuse.
  * GEMV with batch 1 is LLM decode: every weight is read to do 2 FLOPs.
  * as the batch grows, each weight read is reused across more tokens, so
    the point slides right. This is *why* vLLM's throughput rises with
    concurrency until the GPU becomes compute-bound (or runs out of KV cache).
  * big square GEMMs are compute-bound.
"""

import torch

DTYPE = torch.float16
ELT = 2  # bytes per fp16 element


def _make(kind, size, device):
    """Return (name, fn, flops, bytes) for one workload."""
    g = torch.Generator(device="cpu").manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(device, DTYPE)

    if kind == "add":
        n = size
        a, b, out = rand(n), rand(n), torch.empty(n, device=device, dtype=DTYPE)
        return f"add N={n:,}", lambda: torch.add(a, b, out=out), n, 3 * n * ELT
    if kind == "softmax":
        rows, cols = size
        x = rand(rows, cols)
        # ~5 FLOPs per element: max, subtract, exp, sum, divide
        return (f"softmax {rows}x{cols}", lambda: torch.softmax(x, dim=-1),
                5 * rows * cols, 2 * rows * cols * ELT)
    if kind == "gemv":  # (batch x K) @ (K x N): a linear layer during decode
        batch, k, n = size
        x, w = rand(batch, k), rand(k, n)
        flops = 2 * batch * k * n
        bytes_ = (batch * k + k * n + batch * n) * ELT
        return f"linear B={batch} {k}x{n}", lambda: x @ w, flops, bytes_
    if kind == "gemm":
        n = size
        a, b = rand(n, n), rand(n, n)
        return f"gemm {n}^3", lambda: a @ b, 2 * n ** 3, 3 * n * n * ELT
    raise ValueError(kind)


def workload_specs(small=False):
    """(kind, size) pairs. small=True gives tiny sizes for a CPU smoke test."""
    if small:
        return [("add", 1 << 16), ("softmax", (64, 256)), ("gemv", (1, 256, 256)),
                ("gemv", (16, 256, 256)), ("gemm", 128)]
    return [
        ("add", 1 << 26),
        ("softmax", (8192, 4096)),
        ("gemv", (1, 4096, 4096)),
        ("gemv", (8, 4096, 4096)),
        ("gemv", (32, 4096, 4096)),
        ("gemv", (128, 4096, 4096)),
        ("gemv", (512, 4096, 4096)),
        ("gemm", 1024),
        ("gemm", 2048),
        ("gemm", 4096),
        ("gemm", 8192),
    ]


def build_all(device="cuda", small=False):
    return [_make(k, s, device) for k, s in workload_specs(small)]


def time_fn(fn, device="cuda", warmup=10, iters=50):
    """Median runtime in seconds. CUDA events time the GPU, not the Python call."""
    import statistics
    import time

    for _ in range(warmup):
        fn()
    if device != "cuda":
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return statistics.median(ts)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end) / 1e3)
    return statistics.median(ts)


def measure_roofs(device="cuda", small=False):
    """Measure the two roofs instead of trusting spec sheets.

    Bandwidth roof: copy a buffer far bigger than L2 (reads N + writes N bytes).
    Compute roof: a large fp16 matmul on tensor cores.
    """
    n_bytes = (1 << 20) if small else (1 << 30)
    src = torch.empty(n_bytes // ELT, device=device, dtype=DTYPE)
    dst = torch.empty_like(src)
    t = time_fn(lambda: dst.copy_(src), device, warmup=5, iters=20)
    bw = 2 * n_bytes / t

    n = 256 if small else 8192
    a = torch.randn(n, n, device=device, dtype=DTYPE)
    b = torch.randn(n, n, device=device, dtype=DTYPE)
    t = time_fn(lambda: a @ b, device, warmup=5, iters=20)
    flops = 2 * n ** 3 / t
    return {"peak_bw_Bps": bw, "peak_flops": flops}


def run_study(device="cuda", small=False):
    roofs = measure_roofs(device, small)
    rows = []
    for name, fn, flops, bytes_ in build_all(device, small):
        t = time_fn(fn, device)
        ai = flops / bytes_
        achieved = flops / t
        bound = min(roofs["peak_flops"], ai * roofs["peak_bw_Bps"])
        rows.append({
            "name": name, "flops": flops, "bytes": bytes_, "time_us": t * 1e6,
            "ai": ai, "achieved_tflops": achieved / 1e12,
            "achieved_GBps": bytes_ / t / 1e9,
            "roof_tflops": bound / 1e12,
            "pct_of_roof": 100 * achieved / bound,
            "regime": "memory-bound" if ai * roofs["peak_bw_Bps"] < roofs["peak_flops"] else "compute-bound",
        })
    return roofs, rows


def plot_roofline(roofs, rows, title, path, ncu_rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    bw, peak = roofs["peak_bw_Bps"], roofs["peak_flops"]
    ridge = peak / bw
    ai = np.logspace(-2, 4, 400)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.loglog(ai, np.minimum(peak, ai * bw) / 1e12, "k-", lw=2, label="measured roof")
    ax.axvline(ridge, color="gray", ls=":", lw=1)
    ax.text(ridge * 1.1, peak / 1e12 * 0.02, f"ridge point\nAI = {ridge:.0f} FLOP/B",
            fontsize=8, color="gray")
    colors = {"add": "tab:blue", "softmax": "tab:cyan", "linear": "tab:orange", "gemm": "tab:red"}
    for r in rows:
        c = colors[r["name"].split()[0]]
        ax.plot(r["ai"], r["achieved_tflops"], "o", color=c, ms=7)
        ax.annotate(r["name"], (r["ai"], r["achieved_tflops"]), fontsize=7,
                    textcoords="offset points", xytext=(5, -3))
    if ncu_rows:  # same FLOPs, but bytes as measured by the hardware counters
        for r in ncu_rows:
            if r.get("ncu_ai"):
                ax.plot(r["ncu_ai"], r["achieved_tflops"], "x", color="k", ms=6)
        ax.plot([], [], "x", color="k", label="AI from ncu dram bytes")
    ax.set(xlabel="Arithmetic intensity (FLOPs per DRAM byte)",
           ylabel="Achieved TFLOP/s", title=title)
    ax.text(0.012, peak / 1e12 * 1.15,
            f"bandwidth roof {bw / 1e9:.0f} GB/s   |   compute roof {peak / 1e12:.1f} TFLOP/s (fp16)",
            fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":  # CPU smoke test: python roofline/workloads.py
    roofs, rows = run_study("cpu", small=True)
    for r in rows:
        print(f"{r['name']:28s} AI={r['ai']:8.2f}  {r['achieved_tflops'] * 1e3:8.3f} GFLOP/s  {r['regime']}")
