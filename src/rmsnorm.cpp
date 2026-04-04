#include "kernels.h"
#include <cmath>

namespace tk {

void rmsnorm_scalar(float* out, const float* in, const float* weight,
                    size_t n, float eps) {
    float ss = 0.0f;
    for (size_t i = 0; i < n; ++i) ss += in[i] * in[i];
    float inv_rms = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (size_t i = 0; i < n; ++i) out[i] = in[i] * inv_rms * weight[i];
}

void rmsnorm(float* out, const float* in, const float* weight,
             size_t n, float eps) {
    rmsnorm_scalar(out, in, weight, n, eps);
}

} // namespace tk
