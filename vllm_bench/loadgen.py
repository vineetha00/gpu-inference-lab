"""Closed-loop load generator for an OpenAI-compatible streaming server (vLLM).

How it works, in one picture:
    concurrency = C  ->  C workers, each sends a request, waits for the full
    streamed answer, then sends the next one, until N requests are done.

For every request we record three timestamps:
    t_send   : request leaves the client
    t_first  : first generated token arrives   -> TTFT = t_first - t_send
    t_end    : last token arrives              -> E2E  = t_end   - t_send
    TPOT (time per output token, after the first) = (t_end - t_first) / (tokens - 1)

Throughput for a level = total output tokens / wall time of the whole level.
"""

import asyncio
import json
import random
import statistics
import time

import aiohttp

# A small word list. Each prompt is a random shuffle of these words, so no two
# prompts share a prefix and vLLM's prefix cache can't make prefill look free.
_WORDS = (
    "the model reads every token of the prompt before it can write anything "
    "batching lets many users share one forward pass on the gpu memory "
    "bandwidth decides how fast weights stream into the cores each step"
).split()


def make_prompt(approx_words: int, rng: random.Random) -> str:
    words = [rng.choice(_WORDS) for _ in range(approx_words)]
    return "Summarise the following notes.\n" + " ".join(words) + "\nSummary:"


def percentile(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


async def one_request(session, url, model, prompt, max_tokens):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # vLLM extension: keep generating until max_tokens, so every request
        # produces the same number of tokens and runs are comparable.
        "ignore_eos": True,
    }
    t_send = time.perf_counter()
    t_first = None
    n_tokens = 0
    n_chunks = 0
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("choices") and chunk["choices"][0].get("text"):
                n_chunks += 1
                if t_first is None:
                    t_first = time.perf_counter()
            if chunk.get("usage"):
                n_tokens = chunk["usage"]["completion_tokens"]
    t_end = time.perf_counter()
    if n_tokens == 0:  # server didn't send usage; fall back to chunk count
        n_tokens = n_chunks
    if t_first is None:
        t_first = t_end
    return {
        "ttft": t_first - t_send,
        "e2e": t_end - t_send,
        "tpot": (t_end - t_first) / max(n_tokens - 1, 1),
        "tokens": n_tokens,
    }


async def run_level(base_url, model, concurrency, num_requests,
                    prompt_words, max_tokens, seed=0):
    url = base_url.rstrip("/") + "/v1/completions"
    rng = random.Random(seed + concurrency)
    prompts = [make_prompt(prompt_words, rng) for _ in range(num_requests)]
    queue = asyncio.Queue()
    for p in prompts:
        queue.put_nowait(p)
    records = []

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def worker():
            while True:
                try:
                    prompt = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                records.append(await one_request(session, url, model, prompt, max_tokens))

        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = time.perf_counter() - t0

    ttft = [r["ttft"] for r in records]
    tpot = [r["tpot"] for r in records]
    e2e = [r["e2e"] for r in records]
    total_tokens = sum(r["tokens"] for r in records)
    return {
        "concurrency": concurrency,
        "num_requests": len(records),
        "wall_s": wall,
        "ttft_p50_ms": percentile(ttft, 50) * 1e3,
        "ttft_p90_ms": percentile(ttft, 90) * 1e3,
        "ttft_p99_ms": percentile(ttft, 99) * 1e3,
        "tpot_p50_ms": percentile(tpot, 50) * 1e3,
        "tpot_p99_ms": percentile(tpot, 99) * 1e3,
        "e2e_p50_s": percentile(e2e, 50),
        "output_tok_per_s": total_tokens / wall,
        "req_per_s": len(records) / wall,
        "avg_output_tokens": statistics.mean(r["tokens"] for r in records),
    }


async def sweep(base_url, model, levels, prompt_words=400, max_tokens=128,
                min_requests=32, requests_per_worker=4, log=print):
    # Warm-up: the first requests pay for CUDA graph capture, allocator growth, etc.
    await run_level(base_url, model, 4, 8, prompt_words, max_tokens, seed=999)
    results = []
    for c in levels:
        n = max(min_requests, requests_per_worker * c)
        r = await run_level(base_url, model, c, n, prompt_words, max_tokens)
        log(f"conc={c:4d}  TTFT p50={r['ttft_p50_ms']:8.1f} ms  p99={r['ttft_p99_ms']:8.1f} ms  "
            f"TPOT p50={r['tpot_p50_ms']:6.2f} ms  out={r['output_tok_per_s']:8.1f} tok/s")
        results.append(r)
    return results
