"""Correctness tests: the Triton kernel must match the PyTorch reference.

On a GPU:   pytest triton_rmsnorm -q
On a CPU:   TRITON_INTERPRET=1 pytest triton_rmsnorm -q   (Triton's interpreter, slow but exact)
"""

import os

import pytest
import torch

from rmsnorm import rmsnorm, rmsnorm_reference

DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
TOL = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}

# Odd sizes on purpose: 1000 and 5120 are not powers of 2, so the mask matters.
SHAPES = [(1, 128), (7, 1000), (64, 4096), (33, 5120)]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_reference(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    w = torch.randn(shape[-1], device=DEVICE, dtype=dtype)
    torch.testing.assert_close(rmsnorm(x, w), rmsnorm_reference(x, w),
                               atol=TOL[dtype], rtol=TOL[dtype])


def test_3d_input_batch_by_seq_by_hidden():
    torch.manual_seed(0)
    x = torch.randn(2, 9, 768, device=DEVICE, dtype=torch.float16)
    w = torch.ones(768, device=DEVICE, dtype=torch.float16)
    y = rmsnorm(x, w)
    assert y.shape == x.shape
    torch.testing.assert_close(y, rmsnorm_reference(x, w), atol=2e-3, rtol=2e-3)


def test_non_contiguous_rows():
    torch.manual_seed(0)
    big = torch.randn(16, 2048, device=DEVICE, dtype=torch.float32)
    x = big[:, :1024]  # each row starts 2048 elements apart: stride must be honoured
    w = torch.randn(1024, device=DEVICE, dtype=torch.float32)
    torch.testing.assert_close(rmsnorm(x, w), rmsnorm_reference(x, w), atol=1e-5, rtol=1e-5)


def test_large_values_do_not_overflow_fp16():
    # 300^2 = 90,000 > fp16 max (65,504). fp32 accumulation keeps this finite.
    x = torch.full((4, 1024), 300.0, device=DEVICE, dtype=torch.float16)
    w = torch.ones(1024, device=DEVICE, dtype=torch.float16)
    y = rmsnorm(x, w)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.ones_like(y), atol=2e-3, rtol=2e-3)


def test_unit_rms_after_norm():
    torch.manual_seed(0)
    x = torch.randn(8, 2048, device=DEVICE, dtype=torch.float32) * 5 + 3
    y = rmsnorm(x, torch.ones(2048, device=DEVICE))
    rms = y.pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=1e-4)
