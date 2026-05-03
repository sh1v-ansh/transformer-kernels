#include "kernels.h"
#include "utils.h"
#ifdef HAVE_AVX2
#  include <immintrin.h>
#endif
#ifdef HAVE_AVX512
#  include <immintrin.h>
#endif

namespace tk {

void rope_scalar(float* q, float* k,
                 const float* cos_cache, const float* sin_cache,
                 size_t seq_len, size_t n_heads, size_t head_dim) {
    const size_t half=head_dim/2;
    for(size_t s=0;s<seq_len;++s){
        const float* cr=cos_cache+s*half, *sr=sin_cache+s*half;
        for(size_t h=0;h<n_heads;++h)
            for(float* v:{q+(s*n_heads+h)*head_dim,k+(s*n_heads+h)*head_dim})
                for(size_t j=0;j<half;++j){
                    float x0=v[j],x1=v[j+half];
                    v[j]=x0*cr[j]-x1*sr[j]; v[j+half]=x0*sr[j]+x1*cr[j];
                }
    }
}

#ifdef HAVE_AVX2
static void rope_head_avx2(float* h,const float* cr,const float* sr,size_t hd){
    size_t half=hd/2,j=0;
    for(;j+8<=half;j+=8){
        __m256 x0=_mm256_loadu_ps(h+j), x1=_mm256_loadu_ps(h+j+half);
        __m256 c=_mm256_loadu_ps(cr+j), s=_mm256_loadu_ps(sr+j);
        _mm256_storeu_ps(h+j,      _mm256_fmsub_ps(x0,c,_mm256_mul_ps(x1,s)));
        _mm256_storeu_ps(h+j+half, _mm256_fmadd_ps(x0,s,_mm256_mul_ps(x1,c)));
    }
    for(;j<half;++j){float x0=h[j],x1=h[j+half];
        h[j]=x0*cr[j]-x1*sr[j]; h[j+half]=x0*sr[j]+x1*cr[j];}
}
void rope_avx2(float* q,float* k,const float* c,const float* s,size_t sl,size_t nh,size_t hd){
    for(size_t i=0;i<sl;++i){
        const float* cr=c+i*(hd/2), *sr=s+i*(hd/2);
        for(size_t h=0;h<nh;++h){
            rope_head_avx2(q+(i*nh+h)*hd,cr,sr,hd);
            rope_head_avx2(k+(i*nh+h)*hd,cr,sr,hd);
        }
    }
}
#else
void rope_avx2(float* q,float* k,const float* c,const float* s,size_t sl,size_t nh,size_t hd){
    rope_scalar(q,k,c,s,sl,nh,hd);
}
#endif

#ifdef HAVE_AVX512
static void rope_head_avx512(float* h,const float* cr,const float* sr,size_t hd){
    size_t half=hd/2,j=0;
    for(;j+16<=half;j+=16){
        __m512 x0=_mm512_loadu_ps(h+j), x1=_mm512_loadu_ps(h+j+half);
        __m512 c=_mm512_loadu_ps(cr+j), s=_mm512_loadu_ps(sr+j);
        _mm512_storeu_ps(h+j,      _mm512_fmsub_ps(x0,c,_mm512_mul_ps(x1,s)));
        _mm512_storeu_ps(h+j+half, _mm512_fmadd_ps(x0,s,_mm512_mul_ps(x1,c)));
    }
    rope_head_avx2(h+j, cr+j, sr+j, (half-j)*2);
}
void rope_avx512(float* q,float* k,const float* c,const float* s,size_t sl,size_t nh,size_t hd){
    for(size_t i=0;i<sl;++i){
        const float* cr=c+i*(hd/2), *sr=s+i*(hd/2);
        for(size_t h=0;h<nh;++h){
            rope_head_avx512(q+(i*nh+h)*hd,cr,sr,hd);
            rope_head_avx512(k+(i*nh+h)*hd,cr,sr,hd);
        }
    }
}
#else
void rope_avx512(float* q,float* k,const float* c,const float* s,size_t sl,size_t nh,size_t hd){
    rope_avx2(q,k,c,s,sl,nh,hd);
}
#endif

void rope(float* q,float* k,const float* c,const float* s,size_t sl,size_t nh,size_t hd){
    if(cpu_has_avx512f()) rope_avx512(q,k,c,s,sl,nh,hd);
    else if(cpu_has_avx2()) rope_avx2(q,k,c,s,sl,nh,hd);
    else rope_scalar(q,k,c,s,sl,nh,hd);
}

} // namespace tk
