#include "kernels.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef HAVE_AVX2
#  include <immintrin.h>
#endif

namespace tk {

bool cpu_has_avx2() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
bool cpu_has_avx512f() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

void softmax_scalar(float* x, size_t n) {
    float max_val = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) { x[i] = std::exp(x[i] - max_val); sum += x[i]; }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; ++i) x[i] *= inv;
}

#ifdef HAVE_AVX2
void softmax_avx2(float* x, size_t n) {
    const size_t step = 8;
    __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    size_t i = 0;
    for (; i + step <= n; i += step)
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
    float max_val = hmax256_ps(vmax);
    for (; i < n; ++i) max_val = std::max(max_val, x[i]);

    __m256 vsum  = _mm256_setzero_ps();
    __m256 vmaxv = _mm256_set1_ps(max_val);
    i = 0;
    for (; i + step <= n; i += step) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmaxv);
        v = exp_avx2(v);
        _mm256_storeu_ps(x + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }
    float sum = hsum256_ps(vsum);
    for (; i < n; ++i) { x[i] = std::exp(x[i] - max_val); sum += x[i]; }

    __m256 vinv = _mm256_set1_ps(1.0f / sum);
    i = 0;
    for (; i + step <= n; i += step)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
    for (; i < n; ++i) x[i] /= sum;
}
#else
void softmax_avx2(float* x, size_t n) { softmax_scalar(x, n); }
#endif

void softmax_avx512(float* x, size_t n) { softmax_avx2(x, n); }

void softmax(float* x, size_t n) {
    if (cpu_has_avx2()) softmax_avx2(x, n);
    else softmax_scalar(x, n);
}

} // namespace tk
