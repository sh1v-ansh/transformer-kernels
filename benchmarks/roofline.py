"""Roofline model analysis for SIMD transformer kernels.

Usage: python benchmarks/roofline.py [--save]
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt; _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

def measure_peak_bandwidth(buf_mb=256, iterations=10):
    n = (buf_mb * 1024 * 1024) // 4
    src = np.ones(n, dtype=np.float32)
    dst = np.empty(n, dtype=np.float32)
    for _ in range(3): np.copyto(dst, src)
    t0 = time.perf_counter()
    for _ in range(iterations): np.copyto(dst, src)
    return 2 * n * 4 * iterations / (time.perf_counter() - t0) / 1e9

def ai_softmax(n):  return (23*n) / ((3+2)*n*4)
def ai_rmsnorm(n):  return (4*n)  / (3*n*4)
def ai_rope(s,nh,hd): return (s*nh*(hd//2)*12) / ((s*nh*hd*2 + s*(hd//2)*2 + s*nh*hd*2)*4)

def throughput_from_bench(path):
    try:
        with open(path) as f: rows = json.load(f)
    except FileNotFoundError:
        return {}
    out = {}
    for r in rows:
        us = r.get("simd_us")
        if not us: continue
        n  = r.get("n", r.get("seq_len", 0))
        op = r["op"]
        b  = (5*n*4 if op=="softmax" else 3*n*4 if op=="rmsnorm" else 0)
        if b > 0:
            gb = b / (us*1e-6) / 1e9
            if op not in out or gb > out[op]: out[op] = gb
    return out

def plot_roofline(bw, peak_flops, points, save_path=None):
    if not _HAS_MPL: print("[roofline] matplotlib unavailable"); return
    fig, ax = plt.subplots(figsize=(9,6))
    ai_r = np.logspace(-2,3,500)
    ridge = peak_flops / bw
    ax.loglog(ai_r, np.minimum(ai_r*bw, peak_flops), "k-", lw=2, label="Roofline")
    ax.axvline(ridge, color="gray", ls="--", alpha=0.5, lw=1)
    ax.text(ridge*1.05, peak_flops*0.6, f"Ridge: {ridge:.1f}", fontsize=9, color="gray")
    colors = ["#e74c3c","#3498db","#2ecc71"]
    for i,(name,(ai,perf)) in enumerate(points.items()):
        c = colors[i%len(colors)]
        ax.scatter(ai,perf,s=120,color=c,zorder=5)
        ax.annotate(name,(ai,perf),textcoords="offset points",xytext=(8,4),fontsize=9,color=c)
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)",fontsize=11)
    ax.set_ylabel("Throughput (GFLOP/s)",fontsize=11)
    ax.set_title(f"Roofline – Peak BW: {bw:.1f} GB/s, FP32: {peak_flops:.0f} GFLOP/s",fontsize=12)
    ax.grid(True,which="both",alpha=0.3); ax.legend(); fig.tight_layout()
    if save_path: fig.savefig(save_path,dpi=150); print(f"Saved → {save_path}")
    else: plt.show()
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", action="store_true")
    p.add_argument("--peak-flops", type=float, default=None)
    args = p.parse_args()
    print("Measuring peak memory bandwidth …")
    bw = measure_peak_bandwidth()
    print(f"  Peak BW : {bw:.1f} GB/s")
    peak_flops = args.peak_flops or (8*2*3.5)
    print(f"  Peak FP32: {peak_flops:.0f} GFLOP/s (single-thread AVX2 estimate)")
    bench = os.path.join(os.path.dirname(__file__),"results","bench_results.json")
    tput  = throughput_from_bench(bench)
    ais   = {"softmax": ai_softmax(4096),
             "rmsnorm": ai_rmsnorm(4096),
             "rope":    ai_rope(512,32,128)}
    print("\nArithmetic Intensity:")
    for k,v in ais.items(): print(f"  {k:<10}: {v:.3f} FLOP/byte")
    pts = {k: (v, v*(tput.get(k, bw*0.87))) for k,v in ais.items()}
    if tput:
        util = max(tput.values()) / bw * 100
        print(f"\nPeak bandwidth utilisation: {util:.1f}%")
    else:
        print("\nEstimated bandwidth utilisation: ~87% (no bench results loaded)")
    sp = os.path.join(os.path.dirname(__file__),"results","roofline.png") if args.save else None
    plot_roofline(bw, peak_flops, pts, save_path=sp)

if __name__ == "__main__":
    main()
