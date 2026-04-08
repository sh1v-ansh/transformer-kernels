#pragma once
#include <cmath>
#include <cstddef>

#ifdef HAVE_AVX2
#  include <immintrin.h>

namespace tk {

inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline float hmax256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_max_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo  = _mm_max_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}

// Cephes-style minimax exp(x) approximation (~1 ULP, valid [-87, 88])
inline __m256 exp_avx2(__m256 x) {
    const __m256 ln2_rcp = _mm256_set1_ps(1.4426950408889634f);
    const __m256 half    = _mm256_set1_ps(0.5f);
    const __m256 ln2_hi  = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo  = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 p0 = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 p1 = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 p2 = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 p3 = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 p4 = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 p5 = _mm256_set1_ps(5.0000001201E-1f);
    const __m256 one = _mm256_set1_ps(1.0f);

    x = _mm256_min_ps(x, _mm256_set1_ps( 88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));

    __m256 fx = _mm256_fmadd_ps(x, ln2_rcp, half);
    fx = _mm256_floor_ps(fx);
    __m256 r = _mm256_fnmadd_ps(fx, ln2_hi, x);
    r = _mm256_fnmadd_ps(fx, ln2_lo, r);

    __m256 y = p0;
    y = _mm256_fmadd_ps(y, r, p1);
    y = _mm256_fmadd_ps(y, r, p2);
    y = _mm256_fmadd_ps(y, r, p3);
    y = _mm256_fmadd_ps(y, r, p4);
    y = _mm256_fmadd_ps(y, r, p5);
    y = _mm256_fmadd_ps(y, r, one);
    y = _mm256_fmadd_ps(y, r, one);

    __m256i n = _mm256_cvttps_epi32(fx);
    n = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    n = _mm256_slli_epi32(n, 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}

} // namespace tk
#endif // HAVE_AVX2
