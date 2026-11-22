"""Part 2: roofline + Nsight study on a Modal GPU.

Run (from the repo root):
    modal run roofline/modal_roofline.py
    GPU=H100 modal run roofline/modal_roofline.py

What it does, in order:
  1. Measures the GPU's two roofs (copy bandwidth, fp16 matmul FLOP/s).
  2. Times 11 workloads and places each on the roofline (results/roofline_*.png).
  3. Records an Nsight Systems timeline of all workloads (.nsys-rep + kernel summary).
  4. Runs Nsight Compute on each workload to read real DRAM bytes from hardware
     counters, plus one full report on the batch-1 decode GEMV (.ncu-rep).
     Some cloud GPUs block hardware counters; if so, step 4 says so and the
     rest still runs.

Open the .nsys-rep in Nsight Systems and the .ncu-rep in Nsight Compute
(both free from NVIDIA and run on a Mac as viewers).
"""

import json
import os
import sys
from pathlib import Path

import modal

GPU = os.environ.get("GPU", "A10G")
HERE = Path(__file__).parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "apt-get update",
        # Nsight CLIs from NVIDIA's apt repo (already configured in this image).
        "apt-get install -y --no-install-recommends cuda-nsight-compute-12-8 cuda-nsight-systems-12-8"
        " || apt-get install -y --no-install-recommends nsight-compute nsight-systems || true",
    )
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("matplotlib", "numpy")
    .add_local_dir(HERE, "/root/roofline")
)
app = modal.App("roofline-nsight-study")


def _find(tool):
    import glob
    import shutil

    hit = shutil.which(tool)
    if hit:
        return hit
    for pat in [f"/usr/local/cuda*/bin/{tool}", f"/opt/nvidia/nsight-compute/*/{tool}",
                f"/opt/nvidia/nsight-systems/*/bin/{tool}", f"/usr/local/cuda*/nsight-*/{tool}",
                f"/usr/local/cuda*/nsight-systems*/bin/{tool}"]:
        found = sorted(glob.glob(pat))
        if found:
            return found[-1]
    return None


NCU_METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
]


def _parse_ncu_csv(path):
    """Sum each metric over all kernels inside the 'measure' range."""
    import csv

    totals, kernels = {}, set()
    lines = [ln for ln in open(path) if ln.startswith('"')]
    for row in csv.DictReader(lines):
        name, val = row.get("Metric Name"), row.get("Metric Value", "")
        if not name:
            continue
        kernels.add(row.get("Kernel Name", "?"))
        try:
            v = float(val.replace(",", ""))
        except ValueError:
            continue
        if name.endswith("pct_of_peak_sustained_elapsed"):
            totals[name] = max(totals.get(name, 0.0), v)  # report the busiest kernel
        else:
            totals[name] = totals.get(name, 0.0) + v
    return totals, sorted(kernels)


@app.function(gpu=GPU, image=image, timeout=60 * 45)
def study() -> dict:
    import subprocess

    sys.path.insert(0, "/root/roofline")
    from workloads import build_all, plot_roofline, run_study

    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    print("GPU:", gpu_name)
    out = {"gpu": gpu_name, "files": {}, "notes": []}

    # 1 + 2: roofs and timed workloads
    roofs, rows = run_study("cuda")
    out["roofs"], out["rows"] = roofs, rows
    print(f"Measured roofs: {roofs['peak_bw_Bps'] / 1e9:.0f} GB/s, "
          f"{roofs['peak_flops'] / 1e12:.1f} TFLOP/s fp16")
    for r in rows:
        print(f"  {r['name']:26s} AI={r['ai']:9.2f}  {r['achieved_tflops']:7.2f} TFLOP/s  "
              f"{r['pct_of_roof']:5.1f}% of roof  {r['regime']}")

    target = "/root/roofline/profile_target.py"

    # 3: Nsight Systems timeline
    nsys = _find("nsys")
    if nsys:
        p = subprocess.run([nsys, "profile", "-t", "cuda,nvtx", "--stats=true",
                            "-o", "/tmp/timeline", "--force-overwrite", "true",
                            sys.executable, target, "all"],
                           capture_output=True, text=True)
        out["nsys_stats"] = (p.stdout + p.stderr)[-20000:]
        if os.path.exists("/tmp/timeline.nsys-rep"):
            out["files"]["timeline.nsys-rep"] = open("/tmp/timeline.nsys-rep", "rb").read()
    else:
        out["notes"].append("nsys not found in image; skipped the timeline.")

    # 4: Nsight Compute, per workload
    ncu = _find("ncu")
    ncu_ok = False
    if ncu:
        for i, r in enumerate(rows):
            log = f"/tmp/ncu_{i}.csv"
            p = subprocess.run([ncu, "--nvtx", "--nvtx-include", "measure/",
                                "--metrics", ",".join(NCU_METRICS), "--csv",
                                "--print-units", "base", "--log-file", log,
                                sys.executable, target, str(i)],
                               capture_output=True, text=True)
            text = p.stdout + p.stderr + (open(log).read() if os.path.exists(log) else "")
            if "ERR_NVGPUCTRPERM" in text:
                out["notes"].append("Nsight Compute: this host blocks GPU performance counters "
                                    "(ERR_NVGPUCTRPERM). Roofline uses CUDA-event timing only.")
                break
            if not os.path.exists(log):
                out["notes"].append(f"ncu produced no output for {r['name']}: {text[-800:]}")
                continue
            totals, kernels = _parse_ncu_csv(log)
            dram = totals.get("dram__bytes_read.sum", 0) + totals.get("dram__bytes_write.sum", 0)
            if dram > 0:
                ncu_ok = True
                r["ncu_dram_bytes"] = dram
                r["ncu_ai"] = r["flops"] / dram
                r["ncu_kernel_time_us"] = totals.get("gpu__time_duration.sum", 0) / 1e3
                r["ncu_sm_pct"] = totals.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
                r["ncu_mem_pct"] = totals.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
                r["ncu_kernels"] = kernels
                print(f"  ncu {r['name']:22s} dram={dram / 1e6:9.2f} MB (ideal {r['bytes'] / 1e6:9.2f})  "
                      f"SM {r['ncu_sm_pct']}%  MEM {r['ncu_mem_pct']}%")
        if ncu_ok:
            # One full report on the batch-1 decode GEMV, to open in the Nsight Compute UI.
            gemv_idx = next(i for i, r in enumerate(rows) if r["name"].startswith("linear B=1 "))
            subprocess.run([ncu, "--set", "full", "--nvtx", "--nvtx-include", "measure/",
                            "-o", "/tmp/decode_gemv", "-f", sys.executable, target, str(gemv_idx)],
                           capture_output=True, text=True)
            if os.path.exists("/tmp/decode_gemv.ncu-rep"):
                out["files"]["decode_gemv_full.ncu-rep"] = open("/tmp/decode_gemv.ncu-rep", "rb").read()
    else:
        out["notes"].append("ncu not found in image; skipped hardware counters.")
    out["ncu_ok"] = ncu_ok

    plot_roofline(roofs, rows, f"Roofline on {gpu_name} (fp16, measured roofs)",
                  "/tmp/roofline.png", ncu_rows=rows if ncu_ok else None)
    out["files"]["roofline.png"] = open("/tmp/roofline.png", "rb").read()
    for n in out["notes"]:
        print("NOTE:", n)
    return out


@app.local_entrypoint()
def main():
    out = study.remote()
    res = HERE.parent / "results"
    res.mkdir(exist_ok=True)
    tag = f"roofline_{GPU}"
    for fname, data in out.pop("files").items():
        (res / f"{tag}_{fname}").write_bytes(data)
    if "nsys_stats" in out:
        (res / f"{tag}_nsys_stats.txt").write_text(out.pop("nsys_stats"))
    (res / f"{tag}.json").write_text(json.dumps(out, indent=2, default=str))

    roofs = out["roofs"]
    lines = [f"Roofline on **{out['gpu']}**: measured {roofs['peak_bw_Bps'] / 1e9:.0f} GB/s, "
             f"{roofs['peak_flops'] / 1e12:.1f} TFLOP/s fp16, ridge at "
             f"{roofs['peak_flops'] / roofs['peak_bw_Bps']:.0f} FLOP/byte\n",
             "| Workload | AI (ideal) | AI (ncu) | Time (us) | TFLOP/s | GB/s | % of roof | Regime |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in out["rows"]:
        ncu_ai = f"{r['ncu_ai']:.2f}" if r.get("ncu_ai") else "-"
        lines.append(f"| {r['name']} | {r['ai']:.2f} | {ncu_ai} | {r['time_us']:.1f} | "
                     f"{r['achieved_tflops']:.2f} | {r['achieved_GBps']:.0f} | "
                     f"{r['pct_of_roof']:.0f}% | {r['regime']} |")
    md = "\n".join(lines) + "\n"
    if out["notes"]:
        md += "\nNotes:\n" + "\n".join(f"- {n}" for n in out["notes"]) + "\n"
    (res / f"{tag}.md").write_text(md)
    print("\n" + md)
    print(f"Saved results/{tag}_*")
