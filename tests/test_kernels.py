"""Unit tests for SIMD transformer kernels."""
from __future__ import annotations
import math, sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import transformer_kernels as tk
    _HAS_EXT = True
except ImportError:
    _HAS_EXT = False

from python.transformer_kernels.inference import (
    softmax as softmax_py,
    rmsnorm as rmsnorm_py,
    rope    as rope_py,
    build_rope_cache,
)

RNG = np.random.default_rng(42)
RTOL, ATOL = 1e-4, 1e-4

# ── reference implementations ─────────────────────────────────────────────────
def ref_softmax(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()

def ref_rmsnorm(x, w, eps=1e-6):
    return x / math.sqrt((x ** 2).mean() + eps) * w

def ref_rope(q, k, cos, sin):
    half = q.shape[-1] // 2
    def _r(h):
        return np.concatenate([h[..., :half] * cos - h[..., half:] * sin,
                                h[..., :half] * sin + h[..., half:] * cos], axis=-1)
    return _r(q), _r(k)

# ── softmax ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [1, 7, 8, 9, 63, 64, 128, 256, 1024, 4096])
def test_softmax_python(n):
    x = RNG.standard_normal(n).astype(np.float32)
    np.testing.assert_allclose(softmax_py(x.copy()), ref_softmax(x), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize("n", [64, 256, 1024, 4096])
@pytest.mark.skipif(not _HAS_EXT, reason="C extension not built")
def test_softmax_ext(n):
    x = RNG.standard_normal(n).astype(np.float32)
    np.testing.assert_allclose(tk.softmax(x), ref_softmax(x), rtol=RTOL, atol=ATOL)

def test_softmax_sums_to_one():
    x = RNG.standard_normal(512).astype(np.float32)
    assert abs(softmax_py(x).sum() - 1.0) < 1e-5

def test_softmax_numerical_stability():
    x = np.array([1e10, 2e10, 3e10], dtype=np.float32)
    out = softmax_py(x)
    assert np.all(np.isfinite(out)) and abs(out.sum() - 1.0) < 1e-5

# ── rmsnorm ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [1, 7, 8, 9, 63, 64, 128, 256, 1024, 4096])
def test_rmsnorm_python(n):
    x = RNG.standard_normal(n).astype(np.float32)
    w = RNG.uniform(0.5, 1.5, n).astype(np.float32)
    np.testing.assert_allclose(rmsnorm_py(x, w), ref_rmsnorm(x, w), rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize("n", [64, 256, 1024, 4096])
@pytest.mark.skipif(not _HAS_EXT, reason="C extension not built")
def test_rmsnorm_ext(n):
    x = RNG.standard_normal(n).astype(np.float32)
    w = np.ones(n, dtype=np.float32)
    np.testing.assert_allclose(tk.rmsnorm(x, w), ref_rmsnorm(x, w), rtol=RTOL, atol=ATOL)

def test_rmsnorm_unit_weights():
    n = 256
    x = RNG.standard_normal(n).astype(np.float32)
    w = np.ones(n, dtype=np.float32)
    out = rmsnorm_py(x, w)
    rms = math.sqrt((x ** 2).mean() + 1e-6)
    np.testing.assert_allclose(out, x / rms, rtol=RTOL, atol=ATOL)

# ── rope ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seq,nh,hd", [(1, 1, 64), (4, 8, 128), (16, 32, 64)])
def test_rope_python(seq, nh, hd):
    q = RNG.standard_normal((seq, nh, hd)).astype(np.float32)
    k = RNG.standard_normal((seq, nh, hd)).astype(np.float32)
    cos, sin = build_rope_cache(seq, hd)
    cos_b, sin_b = cos[:, np.newaxis, :], sin[:, np.newaxis, :]
    ref_q, ref_k = ref_rope(q, k, cos_b, sin_b)
    out_q, out_k = rope_py(q, k, cos, sin)
    np.testing.assert_allclose(out_q, ref_q, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(out_k, ref_k, rtol=RTOL, atol=ATOL)

def test_rope_preserves_norm():
    """RoPE is a rotation – must preserve the L2 norm of each token."""
    seq, nh, hd = 4, 2, 64
    q = RNG.standard_normal((seq, nh, hd)).astype(np.float32)
    k = q.copy()
    cos, sin = build_rope_cache(seq, hd)
    q_rot, _ = rope_py(q, k, cos, sin)
    np.testing.assert_allclose(
        np.linalg.norm(q_rot.reshape(seq, -1), axis=-1),
        np.linalg.norm(q.reshape(seq, -1),     axis=-1),
        rtol=1e-4,
    )

# ── inference smoke tests ──────────────────────────────────────────────────────
def test_inference_forward_shape():
    from python.transformer_kernels.inference import TransformerInference
    m = TransformerInference(n_layers=2, n_heads=4, head_dim=32,
                              vocab_size=100, max_seq=64)
    logits = m.forward([1, 2, 3, 4])
    assert logits.shape == (4, 100) and np.all(np.isfinite(logits))

def test_inference_generate():
    from python.transformer_kernels.inference import TransformerInference
    m = TransformerInference(n_layers=1, n_heads=2, head_dim=16,
                              vocab_size=50, max_seq=32)
    ids = m.generate([1, 2], max_new_tokens=5)
    assert len(ids) == 7 and all(0 <= i < 50 for i in ids)
