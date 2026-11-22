"""Benchmark the Triton RMSNorm against PyTorch, in GB/s.

RMSNorm is memory-bound, so the honest score is effective bandwidth:
    bytes = read x + write y + read w = (2*M*N + N) * element_size
    GB/s  = bytes / time
and the ceiling is the GPU's measured copy bandwidth. "% of peak" says how
close each version gets to that ceiling.
"""

import torch
import torch.nn.functional as F
import triton

from rmsnorm import rmsnorm, rmsnorm_reference

M = 4096  # rows = tokens in a batch (e.g. 8 sequences x 512 tokens)
NS = [1024, 2048, 4096, 5120, 8192, 16384]  # hidden sizes (4096 = Llama-7B, 5120 = 13B)


def peak_copy_bandwidth():
    n = 1 << 29  # 1 GiB of fp16, far bigger than L2
    src = torch.empty(n, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(src)
    ms = triton.testing.do_bench(lambda: dst.copy_(src), warmup=10, rep=100)
    return 2 * n * 2 / (ms * 1e-3) / 1e9


def run(dtype=torch.float16):
    peak = peak_copy_bandwidth()
    compiled = torch.compile(rmsnorm_reference)
    providers = {
        "triton (ours)": lambda x, w: rmsnorm(x, w),
        "torch eager": lambda x, w: rmsnorm_reference(x, w),
        "F.rms_norm": lambda x, w: F.rms_norm(x, (x.shape[-1],), w, eps=1e-6),
        "torch.compile": lambda x, w: compiled(x, w),
    }
    rows = []
    for n in NS:
        x = torch.randn(M, n, device="cuda", dtype=dtype)
        w = torch.randn(n, device="cuda", dtype=dtype)
        bytes_ = (2 * M * n + n) * x.element_size()
        row = {"N": n}
        for name, fn in providers.items():
            ms = triton.testing.do_bench(lambda: fn(x, w), warmup=25, rep=100)
            row[name] = {"us": ms * 1e3, "GBps": bytes_ / (ms * 1e-3) / 1e9}
        rows.append(row)
        print(f"N={n:6d}  " + "  ".join(f"{k}: {v['GBps']:6.0f} GB/s"
                                       for k, v in row.items() if k != "N"))
    return peak, list(providers), rows


def plot(peak, providers, rows, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["N"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for p in providers:
        ax.plot(ns, [r[p]["GBps"] for r in rows], "o-", label=p, lw=2.5 if "ours" in p else 1.5)
    ax.axhline(peak, color="k", ls="--", lw=1, label=f"measured copy peak ({peak:.0f} GB/s)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set(xlabel="hidden size N", ylabel="effective bandwidth (GB/s)", title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def markdown(peak, providers, rows, gpu):
    head = "| N | " + " | ".join(f"{p} GB/s" for p in providers) + " | ours % of peak | speedup vs eager |"
    lines = [f"RMSNorm forward, fp16, M={M} rows, on **{gpu}** (copy peak {peak:.0f} GB/s)\n",
             head, "|" + "---:|" * (len(providers) + 3)]
    for r in rows:
        ours, eager = r["triton (ours)"], r["torch eager"]
        lines.append(f"| {r['N']} | " + " | ".join(f"{r[p]['GBps']:.0f}" for p in providers)
                     + f" | {100 * ours['GBps'] / peak:.0f}% | {eager['us'] / ours['us']:.1f}x |")
    return "\n".join(lines) + "\n"
