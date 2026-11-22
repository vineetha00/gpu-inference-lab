# GPU Inference Lab

Three hands-on studies of LLM inference performance, each run on a cloud GPU through [Modal](https://modal.com):

1. **vLLM under load.** Serve a model with vLLM and measure time-to-first-token (TTFT), time-per-output-token (TPOT) and throughput as concurrent users go from 1 to 128.
2. **Roofline + Nsight.** Measure the GPU's real bandwidth and compute ceilings, place 11 workloads on a roofline (from batch-1 decode GEMV to large GEMMs), and confirm byte counts with Nsight Compute hardware counters. Also records an Nsight Systems timeline.
3. **Fused RMSNorm in Triton.** A one-pass RMSNorm kernel, tested against PyTorch and benchmarked in GB/s against eager PyTorch, `F.rms_norm` and `torch.compile`.

The three connect: the roofline shows *why* vLLM's throughput rises with concurrency (batching moves decode GEMVs right, out of the memory-bound zone), and the Triton kernel is graded against the same bandwidth roof.

## Setup (once)

```bash
pip install modal
modal setup          # opens a browser to log in and saves a token
```

## Run

From the repo root. Each command builds its container image on the first run (a few minutes), then runs on the GPU and writes results into `results/`.

```bash
modal run triton_rmsnorm/modal_triton.py     # ~5 min. Tests, then the benchmark
modal run roofline/modal_roofline.py         # ~10 min. Roofline, nsys, ncu
modal run vllm_bench/modal_vllm_bench.py     # ~15 min first time (downloads the model)
```

Options:

```bash
GPU=H100 modal run roofline/modal_roofline.py                     # any Modal GPU: T4, L4, A10G (default), L40S, A100, H100
modal run vllm_bench/modal_vllm_bench.py --model Qwen/Qwen2.5-0.5B-Instruct --levels 1,4,16,64,256
modal run vllm_bench/modal_vllm_bench.py --max-tokens 512 --prompt-words 1500
```

## Results

Filled in after the first runs. Each run also writes a `.md` table, a `.png` chart and a `.json` with every number into `results/`.

| Study | GPU | Headline |
|---|---|---|
| vLLM, Qwen2.5-1.5B | | |
| Roofline | | |
| Triton RMSNorm | | |

## Layout

```
vllm_bench/      loadgen.py (async streaming load generator), modal_vllm_bench.py
roofline/        workloads.py (FLOP/byte counts, timing, plot), profile_target.py, modal_roofline.py
triton_rmsnorm/  rmsnorm.py (kernel), test_rmsnorm.py, bench_rmsnorm.py, modal_triton.py
results/         outputs
```

## Testing without a GPU

```bash
TRITON_INTERPRET=1 pytest triton_rmsnorm -q   # runs the kernel in Triton's CPU interpreter
python roofline/workloads.py                  # tiny CPU version of the roofline study
```
