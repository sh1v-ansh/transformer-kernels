"""End-to-end transformer inference loop using SIMD-accelerated kernels."""
from __future__ import annotations
import math
import numpy as np

try:
    from transformer_kernels.transformer_kernels_ext import (
        softmax as _softmax_c, rmsnorm as _rmsnorm_c, rope as _rope_c,
    )
    _HAS_EXT = True
except ImportError:
    _HAS_EXT = False

# ── numpy fallbacks ───────────────────────────────────────────────────────────
def _softmax_np(x):
    x = x - x.max(axis=-1, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def _rmsnorm_np(x, w, eps=1e-6):
    return x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps) * w

# ── dispatch ──────────────────────────────────────────────────────────────────
def softmax(x):
    if _HAS_EXT:
        flat = np.ascontiguousarray(x.reshape(-1), dtype=np.float32)
        return _softmax_c(flat, inplace=True).reshape(x.shape)
    return _softmax_np(x.astype(np.float32))

def rmsnorm(x, w, eps=1e-6):
    if _HAS_EXT:
        return _rmsnorm_c(np.ascontiguousarray(x, dtype=np.float32),
                          np.ascontiguousarray(w, dtype=np.float32), eps)
    return _rmsnorm_np(x.astype(np.float32), w.astype(np.float32), eps)

def rope(q, k, cos_cache, sin_cache):
    if _HAS_EXT:
        return _rope_c(np.ascontiguousarray(q, dtype=np.float32),
                       np.ascontiguousarray(k, dtype=np.float32),
                       np.ascontiguousarray(cos_cache, dtype=np.float32),
                       np.ascontiguousarray(sin_cache, dtype=np.float32))
    half = q.shape[-1] // 2
    cos_b = cos_cache[:, np.newaxis, :]
    sin_b = sin_cache[:, np.newaxis, :]
    def _r(h): return np.concatenate([h[...,:half]*cos_b - h[...,half:]*sin_b,
                                       h[...,:half]*sin_b + h[...,half:]*cos_b], axis=-1)
    return _r(q), _r(k)

def build_rope_cache(max_seq_len, head_dim, base=10_000.0):
    half  = head_dim // 2
    theta = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / half))
    t     = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, theta)
    return np.cos(freqs), np.sin(freqs)

def _attention(q, k, v, mask=None):
    seq, nh, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    scores = np.einsum("shd,thd->hst", q, k) * scale
    if mask is not None:
        scores += mask[np.newaxis]
    probs = np.array([softmax(scores[h]) for h in range(nh)])
    return np.einsum("hst,thd->shd", probs, v)

class TransformerInference:
    """Minimal decoder-only transformer wired to SIMD kernels."""
    def __init__(self, n_layers=12, n_heads=12, head_dim=64,
                 vocab_size=32_000, max_seq=2048):
        self.n_layers, self.n_heads = n_layers, n_heads
        self.head_dim, self.d_model = head_dim, n_heads * head_dim
        self.vocab_size, self.max_seq = vocab_size, max_seq
        rng = np.random.default_rng(0)
        self._init_weights(rng)
        self.cos_cache, self.sin_cache = build_rope_cache(max_seq, head_dim)

    def _init_weights(self, rng):
        d, nh, hd, s = self.d_model, self.n_heads, self.head_dim, 0.02
        self.embed   = rng.standard_normal((self.vocab_size, d)).astype(np.float32) * s
        self.unembed = rng.standard_normal((d, self.vocab_size)).astype(np.float32) * s
        self.norm_f_w = np.ones(d, dtype=np.float32)
        self.layers = [{"attn_norm_w": np.ones(d, np.float32),
                        "ffn_norm_w":  np.ones(d, np.float32),
                        "Wq": rng.standard_normal((d, nh*hd)).astype(np.float32)*s,
                        "Wk": rng.standard_normal((d, nh*hd)).astype(np.float32)*s,
                        "Wv": rng.standard_normal((d, nh*hd)).astype(np.float32)*s,
                        "Wo": rng.standard_normal((nh*hd, d)).astype(np.float32)*s,
                        "W1": rng.standard_normal((d, 4*d)).astype(np.float32)*s,
                        "W2": rng.standard_normal((4*d, d)).astype(np.float32)*s}
                       for _ in range(self.n_layers)]

    def forward(self, token_ids):
        seq  = len(token_ids)
        x    = self.embed[token_ids]
        mask = np.triu(np.full((seq, seq), -1e9, np.float32), k=1)
        for layer in self.layers:
            h = np.stack([rmsnorm(x[s], layer["attn_norm_w"]) for s in range(seq)])
            Q = (h @ layer["Wq"]).reshape(seq, self.n_heads, self.head_dim)
            K = (h @ layer["Wk"]).reshape(seq, self.n_heads, self.head_dim)
            V = (h @ layer["Wv"]).reshape(seq, self.n_heads, self.head_dim)
            Q, K = rope(Q, K, self.cos_cache[:seq], self.sin_cache[:seq])
            x = x + (_attention(Q, K, V, mask).reshape(seq, -1) @ layer["Wo"])
            h = np.stack([rmsnorm(x[s], layer["ffn_norm_w"]) for s in range(seq)])
            x = x + (np.maximum(h @ layer["W1"], 0) @ layer["W2"])
        out = np.stack([rmsnorm(x[s], self.norm_f_w) for s in range(seq)])
        return out @ self.unembed

    def generate(self, prompt_ids, max_new_tokens=20):
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            ids.append(int(np.argmax(self.forward(ids)[-1])))
        return ids
