"""Benchmark SIMD kernels against scalar/numpy baseline.

Usage: python benchmarks/bench_kernels.py [--sizes 128 256 512 1024 2048 4096]
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

try:
    import torch; _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import transformer_kernels as tk; _HAS_EXT = True
except ImportError:
    _HAS_EXT = False

def _timeit(fn, warmup=10, runs=200):
    for _ in range(warmup): fn()
    times = sorted((time.perf_counter(), fn(), time.perf_counter())[2] -
                   (time.perf_counter(), fn(), time.perf_counter())[0]
                   for _ in range(runs))
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter()-t0)*1e6)
    t.sort(); return t[len(t)//2]

def bench_softmax(sizes):
    rows = []
    for n in sizes:
        x = np.random.randn(n).astype(np.float32)
        row = {"op": "softmax", "n": n}
        def scalar(): y=x-x.max(); e=np.exp(y); return e/e.sum()
        row["scalar_us"] = _timeit(scalar)
        if _HAS_EXT:
            row["simd_us"] = _timeit(lambda: tk.softmax(x.copy()))
            row["speedup"] = row["scalar_us"] / row["simd_us"]
        if _HAS_TORCH:
            xt = torch.from_numpy(x)
            row["torch_us"] = _timeit(lambda: torch.softmax(xt, dim=0))
        rows.append(row)
    return rows

def bench_rmsnorm(sizes):
    rows = []
    for n in sizes:
        x = np.random.randn(n).astype(np.float32)
        w = np.ones(n, dtype=np.float32)
        row = {"op": "rmsnorm", "n": n}
        def scalar(): return x / np.sqrt((x**2).mean()+1e-6) * w
        row["scalar_us"] = _timeit(scalar)
        if _HAS_EXT:
            row["simd_us"] = _timeit(lambda: tk.rmsnorm(x, w))
            row["speedup"] = row["scalar_us"] / row["simd_us"]
        if _HAS_TORCH:
            xt, wt = torch.from_numpy(x), torch.from_numpy(w)
            row["torch_us"] = _timeit(lambda: torch.nn.functional.rms_norm(xt,(n,),wt))
        rows.append(row)
    return rows

def bench_rope(seq_lens, n_heads=32, head_dim=128):
    from python.transformer_kernels.inference import build_rope_cache
    rows = []
    for s in seq_lens:
        q = np.random.randn(s,n_heads,head_dim).astype(np.float32)
        k = q.copy()
        cos, sin = build_rope_cache(s, head_dim)
        row = {"op":"rope","seq_len":s,"n_heads":n_heads,"head_dim":head_dim}
        half = head_dim//2
        def scalar():
            q0,q1=q[...,:half],q[...,half:]
            return np.concatenate([q0*cos[:,None]-q1*sin[:,None],
                                   q0*sin[:,None]+q1*cos[:,None]],axis=-1)
        row["scalar_us"] = _timeit(scalar)
        if _HAS_EXT:
            row["simd_us"] = _timeit(lambda: tk.rope(q,k,cos,sin))
            row["speedup"] = row["scalar_us"] / row["simd_us"]
        rows.append(row)
    return rows

def _print_table(rows):
    print(f"\n{'op':<10} {'size':>6}  {'scalar µs':>10}  {'simd µs':>10}  "
          f"{'speedup':>8}  {'torch µs':>10}")
    print("-"*65)
    for r in rows:
        sz = r.get("n", r.get("seq_len","?"))
        print(f"{r['op']:<10} {sz:>6}  "
              f"{r.get('scalar_us',float('nan')):>10.2f}  "
              f"{r.get('simd_us',float('nan')):>10.2f}  "
              f"{r.get('speedup',float('nan')):>8.2f}x  "
              f"{r.get('torch_us',float('nan')):>10.2f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[128,256,512,1024,2048,4096,8192])
    args = p.parse_args()
    rows = bench_softmax(args.sizes) + bench_rmsnorm(args.sizes) + \
           bench_rope([s for s in args.sizes if s<=2048])
    _print_table(rows)
    out = RESULTS_DIR / "bench_results.json"
    with open(out,"w") as f: json.dump(rows,f,indent=2)
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()


# ── oneDNN baseline via Intel Extension for PyTorch (optional) ───────────────
def _try_onednn_softmax(n: int):
    try:
        import intel_extension_for_pytorch  # noqa: F401
        import torch
        xt = torch.randn(n)
        with torch.no_grad():
            return _timeit(lambda: torch.softmax(xt, 0))
    except ImportError:
        return None


def _append_onednn(rows: list[dict]):
    for r in rows:
        if r["op"] != "softmax":
            continue
        t = _try_onednn_softmax(r["n"])
        if t is not None:
            r["onednn_us"] = t
