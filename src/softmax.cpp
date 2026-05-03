#include "kernels.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef HAVE_AVX2
#  include <immintrin.h>
#endif
#ifdef HAVE_AVX512
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
    float mv = *std::max_element(x, x + n), sum = 0.0f;
    for (size_t i=0;i<n;++i){ x[i]=std::exp(x[i]-mv); sum+=x[i]; }
    float inv=1.f/sum; for(size_t i=0;i<n;++i) x[i]*=inv;
}

#ifdef HAVE_AVX2
void softmax_avx2(float* x, size_t n) {
    const size_t step=8;
    __m256 vmax=_mm256_set1_ps(-std::numeric_limits<float>::infinity());
    size_t i=0;
    for(;i+step<=n;i+=step) vmax=_mm256_max_ps(vmax,_mm256_loadu_ps(x+i));
    float mv=hmax256_ps(vmax);
    for(;i<n;++i) mv=std::max(mv,x[i]);

    __m256 vsum=_mm256_setzero_ps(), vmv=_mm256_set1_ps(mv);
    i=0;
    for(;i+step<=n;i+=step){
        __m256 v=exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(x+i),vmv));
        _mm256_storeu_ps(x+i,v); vsum=_mm256_add_ps(vsum,v);
    }
    float sum=hsum256_ps(vsum);
    for(;i<n;++i){ x[i]=std::exp(x[i]-mv); sum+=x[i]; }
    __m256 vinv=_mm256_set1_ps(1.f/sum); i=0;
    for(;i+step<=n;i+=step)
        _mm256_storeu_ps(x+i,_mm256_mul_ps(_mm256_loadu_ps(x+i),vinv));
    for(;i<n;++i) x[i]/=sum;
}
#else
void softmax_avx2(float* x, size_t n){ softmax_scalar(x,n); }
#endif

#ifdef HAVE_AVX512
void softmax_avx512(float* x, size_t n) {
    const size_t step=16;
    __m512 vmax=_mm512_set1_ps(-std::numeric_limits<float>::infinity());
    size_t i=0;
    for(;i+step<=n;i+=step) vmax=_mm512_max_ps(vmax,_mm512_loadu_ps(x+i));
    float mv=_mm512_reduce_max_ps(vmax);
    for(;i<n;++i) mv=std::max(mv,x[i]);

    __m512 vsum=_mm512_setzero_ps(), vmv=_mm512_set1_ps(mv);
    i=0;
    for(;i+step<=n;i+=step){
        __m512 v=_mm512_sub_ps(_mm512_loadu_ps(x+i),vmv);
        __m256 lo=exp_avx2(_mm512_castps512_ps256(v));
        __m256 hi=exp_avx2(_mm512_extractf32x8_ps(v,1));
        v=_mm512_insertf32x8(_mm512_castps256_ps512(lo),hi,1);
        _mm512_storeu_ps(x+i,v); vsum=_mm512_add_ps(vsum,v);
    }
    float sum=_mm512_reduce_add_ps(vsum);
    for(;i<n;++i){ x[i]=std::exp(x[i]-mv); sum+=x[i]; }
    __m512 vinv=_mm512_set1_ps(1.f/sum); i=0;
    for(;i+step<=n;i+=step)
        _mm512_storeu_ps(x+i,_mm512_mul_ps(_mm512_loadu_ps(x+i),vinv));
    for(;i<n;++i) x[i]/=sum;
}
#else
void softmax_avx512(float* x, size_t n){ softmax_avx2(x,n); }
#endif

void softmax(float* x, size_t n) {
    if(cpu_has_avx512f()) softmax_avx512(x,n);
    else if(cpu_has_avx2()) softmax_avx2(x,n);
    else softmax_scalar(x,n);
}

} // namespace tk
