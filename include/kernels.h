#pragma once
#include <cstddef>

namespace tk {

void softmax_scalar (float* x, size_t n);
void softmax_avx2   (float* x, size_t n);
void softmax_avx512 (float* x, size_t n);
void softmax        (float* x, size_t n);

void rmsnorm_scalar (float* out, const float* in, const float* w, size_t n, float eps = 1e-6f);
void rmsnorm_avx2   (float* out, const float* in, const float* w, size_t n, float eps = 1e-6f);
void rmsnorm_avx512 (float* out, const float* in, const float* w, size_t n, float eps = 1e-6f);
void rmsnorm        (float* out, const float* in, const float* w, size_t n, float eps = 1e-6f);

void rope_scalar (float* q, float* k, const float* cos, const float* sin,
                  size_t seq_len, size_t n_heads, size_t head_dim);
void rope_avx2   (float* q, float* k, const float* cos, const float* sin,
                  size_t seq_len, size_t n_heads, size_t head_dim);
void rope_avx512 (float* q, float* k, const float* cos, const float* sin,
                  size_t seq_len, size_t n_heads, size_t head_dim);
void rope        (float* q, float* k, const float* cos, const float* sin,
                  size_t seq_len, size_t n_heads, size_t head_dim);

bool cpu_has_avx2();
bool cpu_has_avx512f();

} // namespace tk
