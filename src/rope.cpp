#include "kernels.h"

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

void rope(float* q, float* k,
          const float* cos_cache, const float* sin_cache,
          size_t seq_len, size_t n_heads, size_t head_dim) {
    rope_scalar(q, k, cos_cache, sin_cache, seq_len, n_heads, head_dim);
}

} // namespace tk
