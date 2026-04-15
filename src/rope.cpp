#include "kernels.h"
#include "utils.h"
#ifdef HAVE_AVX2
#  include <immintrin.h>
#endif

namespace tk {

void rope_scalar(float* q, float* k,
                 const float* cos_cache, const float* sin_cache,
                 size_t seq_len, size_t n_heads, size_t head_dim) {
    const size_t half = head_dim / 2;
    for (size_t s = 0; s < seq_len; ++s) {
        const float* cos_row = cos_cache + s * half;
        const float* sin_row = sin_cache + s * half;
        for (size_t h = 0; h < n_heads; ++h) {
            for (float* vec : {q + (s*n_heads+h)*head_dim,
                               k + (s*n_heads+h)*head_dim}) {
                for (size_t j = 0; j < half; ++j) {
                    float x0 = vec[j], x1 = vec[j + half];
                    vec[j]        = x0 * cos_row[j] - x1 * sin_row[j];
                    vec[j + half] = x0 * sin_row[j] + x1 * cos_row[j];
                }
            }
        }
    }
}

#ifdef HAVE_AVX2
static void rope_head_avx2(float* h, const float* cos_row,
                            const float* sin_row, size_t head_dim) {
    const size_t half = head_dim / 2;
    const size_t step = 8;
    size_t j = 0;
    for (; j + step <= half; j += step) {
        __m256 x0  = _mm256_loadu_ps(h + j);
        __m256 x1  = _mm256_loadu_ps(h + j + half);
        __m256 cos = _mm256_loadu_ps(cos_row + j);
        __m256 sin = _mm256_loadu_ps(sin_row + j);
        _mm256_storeu_ps(h + j,
            _mm256_fmsub_ps(x0, cos, _mm256_mul_ps(x1, sin)));
        _mm256_storeu_ps(h + j + half,
            _mm256_fmadd_ps(x0, sin, _mm256_mul_ps(x1, cos)));
    }
    for (; j < half; ++j) {
        float x0 = h[j], x1 = h[j + half];
        h[j]        = x0 * cos_row[j] - x1 * sin_row[j];
        h[j + half] = x0 * sin_row[j] + x1 * cos_row[j];
    }
}

void rope_avx2(float* q, float* k,
               const float* cos_cache, const float* sin_cache,
               size_t seq_len, size_t n_heads, size_t head_dim) {
    for (size_t s = 0; s < seq_len; ++s) {
        const float* cos_row = cos_cache + s * (head_dim / 2);
        const float* sin_row = sin_cache + s * (head_dim / 2);
        for (size_t h = 0; h < n_heads; ++h) {
            rope_head_avx2(q + (s*n_heads+h)*head_dim, cos_row, sin_row, head_dim);
            rope_head_avx2(k + (s*n_heads+h)*head_dim, cos_row, sin_row, head_dim);
        }
    }
}
#else
void rope_avx2(float* q, float* k, const float* c, const float* s,
               size_t sl, size_t nh, size_t hd) {
    rope_scalar(q, k, c, s, sl, nh, hd);
}
#endif

void rope_avx512(float* q, float* k, const float* c, const float* s,
                 size_t sl, size_t nh, size_t hd) {
    rope_avx2(q, k, c, s, sl, nh, hd);
}

void rope(float* q, float* k,
          const float* cos_cache, const float* sin_cache,
          size_t seq_len, size_t n_heads, size_t head_dim) {
    if (cpu_has_avx2()) rope_avx2(q, k, cos_cache, sin_cache, seq_len, n_heads, head_dim);
    else rope_scalar(q, k, cos_cache, sin_cache, seq_len, n_heads, head_dim);
}

} // namespace tk
