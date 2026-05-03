#include "kernels.h"
#include "utils.h"
#include <cmath>
#ifdef HAVE_AVX2
#  include <immintrin.h>
#endif
#ifdef HAVE_AVX512
#  include <immintrin.h>
#endif

namespace tk {

void rmsnorm_scalar(float* out, const float* in, const float* w, size_t n, float eps) {
    float ss=0.f; for(size_t i=0;i<n;++i) ss+=in[i]*in[i];
    float ir=1.f/std::sqrt(ss/n+eps);
    for(size_t i=0;i<n;++i) out[i]=in[i]*ir*w[i];
}

#ifdef HAVE_AVX2
void rmsnorm_avx2(float* out, const float* in, const float* w, size_t n, float eps) {
    __m256 vss=_mm256_setzero_ps(); size_t i=0;
    for(;i+8<=n;i+=8){ __m256 v=_mm256_loadu_ps(in+i); vss=_mm256_fmadd_ps(v,v,vss); }
    float ss=hsum256_ps(vss); for(;i<n;++i) ss+=in[i]*in[i];
    float ir=1.f/std::sqrt(ss/n+eps); __m256 sc=_mm256_set1_ps(ir); i=0;
    for(;i+8<=n;i+=8)
        _mm256_storeu_ps(out+i,_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(in+i),sc),_mm256_loadu_ps(w+i)));
    for(;i<n;++i) out[i]=in[i]*ir*w[i];
}
#else
void rmsnorm_avx2(float* o,const float* i,const float* w,size_t n,float e){rmsnorm_scalar(o,i,w,n,e);}
#endif

#ifdef HAVE_AVX512
void rmsnorm_avx512(float* out, const float* in, const float* w, size_t n, float eps) {
    __m512 vss=_mm512_setzero_ps(); size_t i=0;
    for(;i+16<=n;i+=16){ __m512 v=_mm512_loadu_ps(in+i); vss=_mm512_fmadd_ps(v,v,vss); }
    float ss=_mm512_reduce_add_ps(vss); for(;i<n;++i) ss+=in[i]*in[i];
    float ir=1.f/std::sqrt(ss/n+eps); __m512 sc=_mm512_set1_ps(ir); i=0;
    for(;i+16<=n;i+=16)
        _mm512_storeu_ps(out+i,_mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(in+i),sc),_mm512_loadu_ps(w+i)));
    for(;i<n;++i) out[i]=in[i]*ir*w[i];
}
#else
void rmsnorm_avx512(float* o,const float* i,const float* w,size_t n,float e){rmsnorm_avx2(o,i,w,n,e);}
#endif

void rmsnorm(float* out,const float* in,const float* w,size_t n,float eps){
    if(cpu_has_avx512f()) rmsnorm_avx512(out,in,w,n,eps);
    else if(cpu_has_avx2()) rmsnorm_avx2(out,in,w,n,eps);
    else rmsnorm_scalar(out,in,w,n,eps);
}

} // namespace tk
