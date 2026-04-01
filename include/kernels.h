#pragma once
#include <cstddef>

namespace tk {

void softmax_scalar(float* x, size_t n);
void rmsnorm_scalar(float* out, const float* in, const float* weight,
                    size_t n, float eps = 1e-6f);
void rope_scalar(float* q, float* k,
                 const float* cos_cache, const float* sin_cache,
                 size_t seq_len, size_t n_heads, size_t head_dim);

// dispatch wrappers (initially just call scalar)
void softmax(float* x, size_t n);
void rmsnorm(float* out, const float* in, const float* weight,
             size_t n, float eps = 1e-6f);
void rope(float* q, float* k,
          const float* cos_cache, const float* sin_cache,
          size_t seq_len, size_t n_heads, size_t head_dim);

bool cpu_has_avx2();
bool cpu_has_avx512f();

} // namespace tk
