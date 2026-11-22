# Study notes: learning this repo end to end

Read one part per sitting, in this order: roofline, then Triton, then vLLM. The roofline is the idea the other two depend on.

---

## Part 2 first: the roofline

**The idea.** A GPU can be slow for two reasons. Either it is waiting for data to arrive from memory, or its maths units are full. Arithmetic intensity (AI) tells you which. AI is the number of FLOPs you do for each byte you move. Low AI means waiting on memory. High AI means waiting on maths.

**The two roofs.** We measure them instead of trusting the spec sheet. Bandwidth comes from copying a 1 GB buffer, which is far bigger than the L2 cache, so every byte really comes from DRAM. Compute comes from a big fp16 matmul. The ridge point is compute ÷ bandwidth: a kernel needs more AI than that to become compute-bound.

**What the points show.**
- `add` and `softmax` sit far left, because each element is used once.
- `linear B=1` is one decode step of an LLM. Every weight is read to do just 2 FLOPs (a multiply and an add), so AI is about 1 and it is badly memory-bound.
- As B grows (8, 32, 128, 512), each weight you read is reused for B tokens, so AI grows roughly with B and the point slides right. **This is why batching users together makes LLM serving efficient.**
- Big GEMMs sit on the flat roof, compute-bound.

**Read in the code:** `workloads.py` → `_make` (how FLOPs and bytes are counted), `time_fn` (why CUDA events), `measure_roofs`.

**Nsight.** Nsight Systems gives a timeline of the whole program: which kernels ran, for how long, and where the gaps are. Nsight Compute zooms into one kernel and reads hardware counters. We use its `dram__bytes` counter to check our hand-counted bytes. If the measured bytes are much higher than the ideal, the kernel is re-reading data. `profile_target.py` wraps the call we care about in an NVTX label called "measure", so ncu skips the warm-up launches.

**Interviewer questions**
1. *"You profile a kernel and it's at 30% of peak FLOPs. Is it a bad kernel?"* Not necessarily. First compute its AI. If it sits left of the ridge, the real ceiling is bandwidth × AI, so compare against that. It is only "bad" if it sits well below the roof at its AI.
2. *"Why is LLM decode memory-bound but prefill compute-bound?"* Prefill processes all prompt tokens at once, so it is a GEMM with high AI. Decode makes one token per sequence, so it is a GEMV that reads every weight for very little maths.
3. *"How would you time a CUDA kernel correctly?"* Warm up first, use CUDA events or synchronize before reading the clock (GPU launches are asynchronous), and take the median of many runs.
4. *"Your measured DRAM bytes are 3× your estimate. What's happening?"* Data is being re-read: poor tiling, cache thrashing, or the operation being split into several kernels that each go back to DRAM.

---

## Part 3: the Triton RMSNorm

**The idea.** RMSNorm divides each row by its root-mean-square, then scales by a weight. In plain PyTorch that is about five kernels, and each one reads and writes the whole tensor. Our kernel reads x once, reads w once and writes y once. RMSNorm does very little maths per byte, so fewer bytes means faster. That is the whole point of fusion.

**How the kernel works.** One program per row. `tl.arange(0, BLOCK_N)` makes a vector of column indices. `BLOCK_N` is the next power of two above N, and `mask = cols < N` switches off the extra lanes. The row is loaded into registers, squared and summed in fp32, then normalised and stored, all without going back to DRAM.

**Why fp32 accumulation.** 300² = 90,000, which is bigger than fp16's maximum (65,504). The test `test_large_values_do_not_overflow_fp16` checks exactly this.

**How it's graded.** In GB/s, against the measured copy bandwidth. A memory-bound kernel can't beat a plain copy, so "% of peak" is an honest score.

**Read in the code:** `rmsnorm.py` (kernel, then wrapper), `test_rmsnorm.py` (what each test protects against), `bench_rmsnorm.py` (the bytes formula).

**Interviewer questions**
1. *"Why did you write this kernel when PyTorch already has RMSNorm?"* Eager PyTorch launches several kernels and moves the tensor through DRAM several times. Fusing makes it one pass, which is the minimum traffic for a memory-bound op.
2. *"What happens if the hidden size is 5120, not a power of 2?"* BLOCK_N becomes 8192 and the mask turns off the extra lanes for both loads and stores. There is a test for 5120 and 1000.
3. *"What breaks if the hidden size is huge, like 64K?"* The row no longer fits in registers. You would loop over the row in chunks, with two passes: one for the sum of squares, one for the normalisation.
4. *"How did torch.compile compare, and why?"* (Answer from your results. torch.compile also fuses into a Triton kernel, so it should land close to ours. Eager should be clearly slower.)

---

## Part 1: vLLM under load

**The idea.** We start a vLLM server and send it streaming requests from C simulated users at once. Each user sends a request, waits for the full answer, then sends the next one (a "closed loop"). C goes 1, 2, 4 … 128.

**The three numbers.**
- **TTFT** (time to first token): how long a user waits before anything appears. This is mostly prefill, plus queueing.
- **TPOT** (time per output token): how fast the text streams after that. One decode step.
- **Throughput:** total output tokens per second across all users.

**What you should see.** As C goes up, throughput climbs a lot at first, because decode steps are memory-bound and batching more sequences into one step costs almost nothing extra (that is the roofline's GEMV story). TPOT rises slowly. Then throughput flattens, either because the GPU becomes compute-bound or because the KV cache is full and new requests have to queue. Once requests queue, TTFT p99 jumps. The server log shows the KV-cache size ("GPU blocks") and maximum concurrency, which explain where the knee is.

**Why random prompts and ignore_eos.** Random prompts stop vLLM's prefix cache from reusing earlier prompts, which would make prefill look free. `ignore_eos` makes every answer the same length, so each level is a fair comparison.

**Read in the code:** `loadgen.py` → `one_request` (streaming and timestamps), `run_level` (workers and queue), then `modal_vllm_bench.py` → `bench`.

**Interviewer questions**
1. *"Throughput keeps rising with concurrency. Why not just run at 128 users?"* Per-user speed (TPOT) and tail TTFT get worse. The right operating point depends on the latency target (SLO), for example TTFT p99 under 500 ms.
2. *"What is continuous batching, and where does it show up in your numbers?"* vLLM adds new requests to the running batch at every decode step instead of waiting for a whole batch to finish. That is why throughput scales smoothly with concurrency rather than in steps.
3. *"What limits the maximum batch size?"* KV-cache memory. Each running sequence holds keys and values for every token. PagedAttention stores them in fixed-size blocks so memory isn't wasted, but the total is still capped by what's left of GPU memory after the weights.
4. *"How would you lower TTFT at high load?"* Chunked prefill (splitting long prompts so they don't stall decodes), prefix caching for shared system prompts, more GPUs, or disaggregating prefill and decode onto separate GPUs.

---

## After the runs

Put your numbers into the README results table. Then write one resume bullet per part, using real numbers, in this shape: *what you built → what you measured → what you found.*
