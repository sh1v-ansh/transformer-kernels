from setuptools import setup, Extension
import pybind11, subprocess, os
from pathlib import Path

def _flag_ok(flag):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False) as f:
        f.write("int main(){}"); name = f.name
    try:
        return subprocess.call(["c++", flag, "-o", "/dev/null", name],
                               stderr=subprocess.DEVNULL) == 0
    finally:
        os.unlink(name)

extra = ["-O3", "-ffast-math", "-std=c++17"]
defines = []
if _flag_ok("-mavx2"):
    extra += ["-mavx2", "-mfma"]; defines.append(("HAVE_AVX2", None))
if _flag_ok("-mavx512f"):
    defines.append(("HAVE_AVX512", None))

ext = Extension(
    "transformer_kernels.transformer_kernels_ext",
    sources=["src/softmax.cpp","src/rmsnorm.cpp","src/rope.cpp","src/bindings.cpp"],
    include_dirs=["include", pybind11.get_include()],
    extra_compile_args=extra,
    define_macros=defines,
    language="c++",
)

setup(
    name="transformer-kernels", version="0.1.0",
    description="AVX2/AVX-512 vectorized transformer kernels",
    packages=["transformer_kernels"],
    package_dir={"transformer_kernels": "python/transformer_kernels"},
    ext_modules=[ext],
    install_requires=["pybind11>=2.11","numpy>=1.24"],
    python_requires=">=3.9",
)
