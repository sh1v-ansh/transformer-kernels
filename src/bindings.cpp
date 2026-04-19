#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include "kernels.h"

namespace py = pybind11;

static float* mptr(py::array_t<float>& a) {
    return reinterpret_cast<float*>(a.request().ptr);
}
static const float* cptr(const py::array_t<float>& a) {
    return reinterpret_cast<const float*>(a.request().ptr);
}
static size_t numel(const py::array_t<float>& a) {
    size_t n = 1;
    for (auto s : a.shape()) n *= static_cast<size_t>(s);
    return n;
}

py::array_t<float> softmax(py::array_t<float> x, bool inplace = false) {
    if (!inplace) x = x.attr("copy")().cast<py::array_t<float>>();
    tk::softmax(mptr(x), numel(x));
    return x;
}

py::array_t<float> rmsnorm(const py::array_t<float>& inp,
                            const py::array_t<float>& weight,
                            float eps = 1e-6f) {
    if (numel(inp) != numel(weight))
        throw std::invalid_argument("input and weight size mismatch");
    auto out = py::array_t<float>(inp.request().shape);
    tk::rmsnorm(mptr(out), cptr(inp), cptr(weight), numel(inp), eps);
    return out;
}

std::pair<py::array_t<float>, py::array_t<float>>
rope(py::array_t<float> q, py::array_t<float> k,
     const py::array_t<float>& cos_cache,
     const py::array_t<float>& sin_cache) {
    auto qr = q.request();
    if (qr.ndim != 3)
        throw std::invalid_argument("q/k must be 3-D [seq_len, n_heads, head_dim]");
    size_t seq_len  = static_cast<size_t>(qr.shape[0]);
    size_t n_heads  = static_cast<size_t>(qr.shape[1]);
    size_t head_dim = static_cast<size_t>(qr.shape[2]);
    q = q.attr("copy")().cast<py::array_t<float>>();
    k = k.attr("copy")().cast<py::array_t<float>>();
    tk::rope(mptr(q), mptr(k), cptr(cos_cache), cptr(sin_cache),
             seq_len, n_heads, head_dim);
    return {q, k};
}

py::dict cpu_features() {
    py::dict d;
    d["avx2"]    = tk::cpu_has_avx2();
    d["avx512f"] = tk::cpu_has_avx512f();
    return d;
}

PYBIND11_MODULE(transformer_kernels_ext, m) {
    m.doc() = "AVX2/AVX-512 transformer kernels";
    m.def("softmax",      &softmax,      py::arg("x"), py::arg("inplace") = false);
    m.def("rmsnorm",      &rmsnorm,      py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-6f);
    m.def("rope",         &rope,         py::arg("q"), py::arg("k"),
                                         py::arg("cos_cache"), py::arg("sin_cache"));
    m.def("cpu_features", &cpu_features);
}
