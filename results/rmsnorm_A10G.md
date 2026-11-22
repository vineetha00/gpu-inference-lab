RMSNorm forward, fp16, M=4096 rows, on **NVIDIA A10** (copy peak 485 GB/s)

| N | triton (ours) GB/s | torch eager GB/s | F.rms_norm GB/s | torch.compile GB/s | ours % of peak | speedup vs eager |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 439 | 45 | 45 | 431 | 90% | 9.8x |
| 2048 | 464 | 47 | 47 | 476 | 96% | 9.9x |
| 4096 | 478 | 48 | 48 | 463 | 98% | 10.0x |
| 5120 | 478 | 48 | 48 | 463 | 99% | 10.0x |
| 8192 | 481 | 48 | 48 | 446 | 99% | 9.9x |
| 16384 | 477 | 49 | 49 | 366 | 98% | 9.8x |
