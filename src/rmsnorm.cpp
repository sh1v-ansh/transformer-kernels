#include "kernels.h"
#include "utils.h"
#include <cmath>
#ifdef HAVE_AVX2
#  include <immintrin.h>
#endif

namespace tk {

void rmsnorm_scalar(float* out, const float* in, const float* weight,
                    size_t n, float eps) {
    float ss = 0.0f;
    for (size_t i = 0; i < n; ++i) ss += in[i] * in[i];
    float inv_rms = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (size_t i = 0; i < n; ++i) out[i] = in[i] * inv_rms * weight[i];
}

#ifdef HAVE_AVX2
void rmsnorm_avx2(float* out, const float* in, const float* weight,
                  size_t n, float eps) {
    const size_t step = 8;
    __m256 vss = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + step <= n; i += step) {
        __m256 v = _mm256_loadu_ps(in + i);
        vss = _mm256_fmadd_ps(v, v, vss);
    }
    float ss = hsum256_ps(vss);
    for (; i < n; ++i) ss += in[i] * in[i];

    float inv_rms = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    __m256 scale  = _mm256_set1_ps(inv_rms);
    i = 0;
    for (; i + step <= n; i += step) {
        __m256 v = _mm256_mul_ps(_mm256_loadu_ps(in + i), scale);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(v, _mm256_loadu_ps(weight + i)));
    }
    for (; i < n; ++i) out[i] = in[i] * inv_rms * weight[i];
}
#else
void rmsnorm_avx2(float* out, const float* in, const float* weight,
                  size_t n, float eps) { rmsnorm_scalar(out, in, weight, n, eps); }
#endif

void rmsnorm_avx512(float* out, const float* in, const float* weight,
                    size_t n, float eps) { rmsnorm_avx2(out, in, weight, n, eps); }

void rmsnorm(float* out, const float* in, const float* weight,
             size_t n, float eps) {
    if (cpu_has_avx2()) rmsnorm_avx2(out, in, weight, n, eps);
    else rmsnorm_scalar(out, in, weight, n, eps);
}

} // namespace tk
