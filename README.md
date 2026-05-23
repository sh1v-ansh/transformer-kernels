# SIMD Transformer Kernels

AVX2/AVX-512 vectorized kernels for softmax, RoPE, and RMSNorm in C++, with Python bindings via pybind11 and an end-to-end transformer inference loop.

## Results

| Kernel   | Scalar (µs) | SIMD (µs) | Speedup  | Notes                              |
|----------|-------------|-----------|----------|------------------------------------|
| softmax  | 42.1        | 12.4      | **3.4×** | n=4096, AVX2, Cephes fast-exp poly |
| rmsnorm  | 18.7        |  5.7      | **3.3×** | n=4096, AVX2 FMA sq-sum + scale    |
| rope     | 31.2        |  9.6      | **3.3×** | seq=512, nh=32, hd=128, AVX2       |

Roofline analysis confirms **87% of peak memory bandwidth utilisation** (47 GB/s measured, DDR5-4800).
Benchmarked against PyTorch CPU (`torch.softmax`, `torch.nn.functional.rms_norm`) and oneDNN (Intel Extension for PyTorch).

## Requirements

- GCC ≥ 12 or Clang ≥ 15 with AVX2 support (`-mavx2 -mfma`)
- CMake ≥ 3.16
- Python ≥ 3.9, pybind11 ≥ 2.11, numpy ≥ 1.24
- PyTorch ≥ 2.1 (optional, baseline comparison)

## Build

### Python package (recommended)

```bash
pip install pybind11 numpy
pip install -e .
```

### CMake

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Usage

```python
import numpy as np
import transformer_kernels as tk

print(tk.cpu_features())   # {'avx2': True, 'avx512f': False}

x   = np.random.randn(1024).astype(np.float32)
p   = tk.softmax(x)

inp = np.random.randn(512).astype(np.float32)
w   = np.ones(512, dtype=np.float32)
out = tk.rmsnorm(inp, w)

from python.transformer_kernels.inference import build_rope_cache
q   = np.random.randn(32, 16, 128).astype(np.float32)
k   = q.copy()
cos, sin = build_rope_cache(32, 128)
q_rot, k_rot = tk.rope(q, k, cos, sin)
```

### End-to-end inference loop

```python
from transformer_kernels import TransformerInference

model = TransformerInference(n_layers=12, n_heads=12, head_dim=64,
                              vocab_size=32000, max_seq=2048)
logits = model.forward([1, 42, 17, 8])   # [seq_len, vocab_size]
ids    = model.generate([1, 42], max_new_tokens=20)
```

## Benchmarks

```bash
python benchmarks/bench_kernels.py --sizes 128 256 512 1024 2048 4096
```

## Roofline Analysis

```bash
python benchmarks/roofline.py --save
# saves benchmarks/results/roofline.png
```

See `benchmarks/results/` for saved benchmark tables and roofline plots.

## Tests

```bash
pytest tests/ -v
```

## Architecture

```
include/
  kernels.h          public API (scalar / AVX2 / AVX-512 declarations + dispatch)
  utils.h            horizontal reductions (hsum, hmax), Cephes fast exp(x)
src/
  softmax.cpp        3-pass softmax: max-scan → exp(x-max) → normalise
  rmsnorm.cpp        2-pass RMSNorm: sq-sum-reduce → scale × weight
  rope.cpp           RoPE: paired (x0, x1) half-rotation over head dim
  bindings.cpp       pybind11 module (softmax / rmsnorm / rope / cpu_features)
python/
  transformer_kernels/
    __init__.py      public Python API
    inference.py     end-to-end transformer inference loop + RoPE cache builder
benchmarks/
  bench_kernels.py   wall-clock benchmarks vs scalar / PyTorch CPU / oneDNN
  roofline.py        roofline model, peak BW probe, bandwidth utilisation plot
  results/           bench_summary.txt, roofline_summary.txt
tests/
  test_kernels.py    numerical correctness (parametrised) + inference smoke tests
```
