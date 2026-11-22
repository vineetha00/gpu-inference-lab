"""Part 1: serve a model with vLLM on a Modal GPU and load-test it.

Run (from the repo root):
    modal run vllm_bench/modal_vllm_bench.py
    GPU=L4 modal run vllm_bench/modal_vllm_bench.py --model Qwen/Qwen2.5-0.5B-Instruct
    modal run vllm_bench/modal_vllm_bench.py --levels 1,4,16,64 --max-tokens 256

Outputs land in results/: a JSON with every number, a Markdown table, and a PNG.
The vLLM server and the load generator run inside the same container, so the
numbers measure the server, not your home internet.
"""

import json
import os
import sys
from pathlib import Path

import modal

GPU = os.environ.get("GPU", "A10G")
HERE = Path(__file__).parent
VLLM_VERSION = "0.11.0"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # vLLM 0.11 expects transformers 4.x; transformers 5 removed a tokenizer
    # attribute it uses, so pin the older line.
    .pip_install(f"vllm=={VLLM_VERSION}", "transformers==4.57.1", "aiohttp", "matplotlib")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_dir(HERE, "/root/vllm_bench")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
app = modal.App("vllm-load-bench")


def _plot(results, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = [r["concurrency"] for r in results]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(c, [r["ttft_p50_ms"] for r in results], "o-", label="p50")
    ax[0, 0].plot(c, [r["ttft_p99_ms"] for r in results], "s--", label="p99")
    ax[0, 0].set(title="Time to first token", ylabel="ms")
    ax[0, 0].legend()
    ax[0, 1].plot(c, [r["output_tok_per_s"] for r in results], "o-", color="tab:green")
    ax[0, 1].set(title="Output throughput", ylabel="tokens / s")
    ax[1, 0].plot(c, [r["tpot_p50_ms"] for r in results], "o-", color="tab:red")
    ax[1, 0].set(title="Time per output token (p50)", ylabel="ms")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xscale("log", base=2)
        a.set_xlabel("concurrent requests")
        a.grid(alpha=0.3)
    # The trade-off curve: each point is one concurrency level.
    ax[1, 1].plot([r["output_tok_per_s"] for r in results],
                  [r["tpot_p50_ms"] for r in results], "o-", color="tab:purple")
    for r in results:
        ax[1, 1].annotate(str(r["concurrency"]),
                          (r["output_tok_per_s"], r["tpot_p50_ms"]),
                          textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax[1, 1].set(title="Throughput vs per-user speed (labels = concurrency)",
                 xlabel="output tokens / s", ylabel="TPOT p50 (ms)")
    ax[1, 1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


@app.function(gpu=GPU, image=image, volumes={"/root/.cache/huggingface": hf_cache},
              timeout=60 * 60)
def bench(model: str, levels: list, prompt_words: int, max_tokens: int,
          max_model_len: int) -> dict:
    import asyncio
    import subprocess
    import time
    import urllib.request

    sys.path.insert(0, "/root/vllm_bench")
    from loadgen import sweep

    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    print("GPU:", gpu_name)

    log = open("/tmp/vllm_server.log", "w")
    server = subprocess.Popen(
        ["vllm", "serve", model, "--port", "8000",
         "--max-model-len", str(max_model_len),
         "--gpu-memory-utilization", "0.90"],
        stdout=log, stderr=subprocess.STDOUT)

    # Wait for the server (first run also downloads the weights into the volume).
    t0 = time.time()
    while True:
        if server.poll() is not None:
            raise RuntimeError("vLLM exited early:\n" + open("/tmp/vllm_server.log").read()[-4000:])
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            break
        except Exception:
            if time.time() - t0 > 20 * 60:
                raise TimeoutError("vLLM did not come up in 20 minutes")
            time.sleep(3)
    startup_s = time.time() - t0
    print(f"vLLM ready after {startup_s:.0f}s")

    results = asyncio.run(sweep("http://localhost:8000", model, levels,
                                prompt_words=prompt_words, max_tokens=max_tokens))
    server.terminate()
    hf_cache.commit()

    title = f"vLLM {VLLM_VERSION} | {model} | {gpu_name} | ~{prompt_words}-word prompt, {max_tokens} output tokens"
    _plot(results, title, "/tmp/vllm_bench.png")
    log.close()
    server_log = open("/tmp/vllm_server.log").read()
    return {
        "meta": {"model": model, "gpu": gpu_name, "vllm": VLLM_VERSION,
                 "prompt_words": prompt_words, "max_tokens": max_tokens,
                 "max_model_len": max_model_len, "startup_s": startup_s},
        "results": results,
        "png": open("/tmp/vllm_bench.png", "rb").read(),
        # The server log shows KV-cache size ("# GPU blocks") and max concurrency,
        # which explain where the throughput curve flattens.
        "server_log_tail": server_log[-6000:],
    }


def _markdown(meta, results):
    lines = [f"**{meta['model']}** on **{meta['gpu']}** (vLLM {meta['vllm']}), "
             f"~{meta['prompt_words']}-word prompts, {meta['max_tokens']} output tokens\n",
             "| Concurrency | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | Output tok/s | Req/s |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['concurrency']} | {r['ttft_p50_ms']:.1f} | {r['ttft_p99_ms']:.1f} | "
                     f"{r['tpot_p50_ms']:.2f} | {r['output_tok_per_s']:.0f} | {r['req_per_s']:.2f} |")
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct",
         levels: str = "1,2,4,8,16,32,64,128",
         prompt_words: int = 400, max_tokens: int = 128, max_model_len: int = 4096):
    out = bench.remote(model, [int(x) for x in levels.split(",")],
                       prompt_words, max_tokens, max_model_len)
    res_dir = HERE.parent / "results"
    res_dir.mkdir(exist_ok=True)
    tag = f"vllm_{GPU}_{model.split('/')[-1]}".replace(" ", "")
    (res_dir / f"{tag}.png").write_bytes(out.pop("png"))
    (res_dir / f"{tag}_server.log").write_text(out.pop("server_log_tail"))
    (res_dir / f"{tag}.json").write_text(json.dumps(out, indent=2))
    md = _markdown(out["meta"], out["results"])
    (res_dir / f"{tag}.md").write_text(md)
    print("\n" + md)
    print(f"Saved results/{tag}.png, .json, .md, _server.log")
