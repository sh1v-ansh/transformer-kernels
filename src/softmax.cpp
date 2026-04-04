#include "kernels.h"
#include <algorithm>
#include <cmath>

namespace tk {

bool cpu_has_avx2()   { return false; }
bool cpu_has_avx512f(){ return false; }

void softmax_scalar(float* x, size_t n) {
    float max_val = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) { x[i] = std::exp(x[i] - max_val); sum += x[i]; }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; ++i) x[i] *= inv;
}

void softmax(float* x, size_t n) { softmax_scalar(x, n); }

} // namespace tk
