"""Tiny program for the profilers to watch.

    python profile_target.py 3      -> run workload #3 (warm up, then one call
                                       inside an NVTX range called "measure")
    python profile_target.py all    -> every workload, each in its own NVTX
                                       range, for an Nsight Systems timeline

NVTX ranges are labels on the timeline. Nsight Compute uses the "measure"
label to profile only the kernels we care about, skipping warm-up launches.
"""

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from workloads import build_all  # noqa: E402


def main():
    which = sys.argv[1]
    work = build_all("cuda")
    picked = range(len(work)) if which == "all" else [int(which)]
    for i in picked:
        name, fn, _, _ = work[i]
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        label = "measure" if which != "all" else name
        torch.cuda.nvtx.range_push(label)
        fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()


if __name__ == "__main__":
    main()
