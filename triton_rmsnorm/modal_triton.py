"""Part 3 runner: test and benchmark the Triton RMSNorm on a Modal GPU.

Run (from the repo root):
    modal run triton_rmsnorm/modal_triton.py
    GPU=H100 modal run triton_rmsnorm/modal_triton.py
"""

import os
import sys
from pathlib import Path

import modal

GPU = os.environ.get("GPU", "A10G")
HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("pytest", "matplotlib")
    .add_local_dir(HERE, "/root/triton_rmsnorm")
)
app = modal.App("triton-rmsnorm")


@app.function(gpu=GPU, image=image, timeout=60 * 30)
def test_and_bench() -> dict:
    import subprocess

    os.chdir("/root/triton_rmsnorm")
    sys.path.insert(0, "/root/triton_rmsnorm")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()

    # Correctness first: never benchmark a kernel that gives wrong answers.
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "test_rmsnorm.py"],
                       capture_output=True, text=True)
    print(t.stdout[-3000:])
    if t.returncode != 0:
        return {"gpu": gpu, "tests_passed": False, "pytest": t.stdout + t.stderr}

    import bench_rmsnorm as b
    peak, providers, rows = b.run()
    b.plot(peak, providers, rows, f"Fused RMSNorm (Triton) vs PyTorch on {gpu}", "/tmp/rmsnorm.png")
    return {"gpu": gpu, "tests_passed": True, "pytest": t.stdout[-3000:],
            "markdown": b.markdown(peak, providers, rows, gpu),
            "rows": rows, "png": open("/tmp/rmsnorm.png", "rb").read()}


@app.local_entrypoint()
def main():
    import json

    out = test_and_bench.remote()
    res = HERE.parent / "results"
    res.mkdir(exist_ok=True)
    tag = f"rmsnorm_{GPU}"
    if not out["tests_passed"]:
        print(out["pytest"])
        raise SystemExit("Tests failed on the GPU, benchmark skipped.")
    (res / f"{tag}.png").write_bytes(out.pop("png"))
    (res / f"{tag}.md").write_text(out["markdown"])
    (res / f"{tag}.json").write_text(json.dumps(out, indent=2))
    print("\n" + out["markdown"])
    print(f"Saved results/{tag}.png, .md, .json")
