Roofline on **NVIDIA A10**: measured 483 GB/s, 73.2 TFLOP/s fp16, ridge at 152 FLOP/byte

| Workload | AI (ideal) | AI (ncu) | Time (us) | TFLOP/s | GB/s | % of roof | Regime |
|---|---:|---:|---:|---:|---:|---:|---|
| add N=67,108,864 | 0.17 | - | 834.5 | 0.08 | 482 | 100% | memory-bound |
| softmax 8192x4096 | 1.25 | - | 367.6 | 0.46 | 365 | 76% | memory-bound |
| linear B=1 4096x4096 | 1.00 | - | 80.9 | 0.41 | 415 | 86% | memory-bound |
| linear B=8 4096x4096 | 7.97 | - | 84.0 | 3.20 | 401 | 83% | memory-bound |
| linear B=32 4096x4096 | 31.51 | - | 84.1 | 12.77 | 405 | 84% | memory-bound |
| linear B=128 4096x4096 | 120.47 | - | 86.1 | 49.86 | 414 | 86% | memory-bound |
| linear B=512 4096x4096 | 409.60 | - | 202.8 | 84.73 | 207 | 116% | compute-bound |
| gemm 1024^3 | 341.33 | - | 47.1 | 45.59 | 134 | 62% | compute-bound |
| gemm 2048^3 | 682.67 | - | 236.5 | 72.63 | 106 | 99% | compute-bound |
| gemm 4096^3 | 1365.33 | - | 2480.6 | 55.40 | 41 | 76% | compute-bound |
| gemm 8192^3 | 2730.67 | - | 15176.7 | 72.45 | 27 | 99% | compute-bound |
