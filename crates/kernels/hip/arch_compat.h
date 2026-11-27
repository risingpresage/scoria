#pragma once
#include <hip/hip_runtime.h>

#if defined(__gfx1200__) || defined(__gfx1201__)
#define SCORIA_RDNA4 1
#define SCORIA_WAVE_SIZE 32

__device__ __forceinline__ uint4 scoria_load_nt_b128(const uint4* addr) {
    uint4 v;
    asm volatile("global_load_b128 %0, %1, off th:TH_LOAD_NT"
                 : "=&v"(v)
                 : "v"(addr)
                 : "memory");
    return v;
}

__device__ __forceinline__ uint4 scoria_load_ht_b128(const uint4* addr) {
    uint4 v;
    asm volatile("global_load_b128 %0, %1, off th:TH_LOAD_HT"
                 : "=&v"(v)
                 : "v"(addr)
                 : "memory");
    return v;
}

__device__ __forceinline__ void scoria_wait_loads() {
    asm volatile("s_wait_loadcnt 0" ::: "memory");
}

__device__ __forceinline__ unsigned scoria_lut16(unsigned lut_bits, unsigned nib) {
    return __builtin_amdgcn_permlane16_var(0u, lut_bits, nib, false, false);
}

#elif defined(__gfx942__) || defined(__gfx950__) || defined(__gfx90a__)
#define SCORIA_CDNA 1
#define SCORIA_WAVE_SIZE 64

__device__ __forceinline__ uint4 scoria_load_nt_b128(const uint4* addr) {
    uint4 v;
#if defined(__gfx942__) || defined(__gfx950__)
    asm volatile("global_load_dwordx4 %0, %1, off nt"
                 : "=&v"(v)
                 : "v"(addr)
                 : "memory");
#else
    asm volatile("global_load_dwordx4 %0, %1, off glc slc"
                 : "=&v"(v)
                 : "v"(addr)
                 : "memory");
#endif
    return v;
}

__device__ __forceinline__ uint4 scoria_load_ht_b128(const uint4* addr) {
    uint4 v;
    asm volatile("global_load_dwordx4 %0, %1, off"
                 : "=&v"(v)
                 : "v"(addr)
                 : "memory");
    return v;
}

__device__ __forceinline__ void scoria_wait_loads() {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

__device__ __forceinline__ unsigned scoria_lut16(unsigned lut_bits, unsigned nib) {
    return (unsigned)__builtin_amdgcn_ds_bpermute((int)((nib & 15u) << 2), (int)lut_bits);
}

#else
#error "Unsupported AMDGPU target: scoria requires gfx1200/gfx1201 (RDNA4) or gfx90a/gfx942/gfx950 (CDNA)"
#endif

typedef unsigned scoria_u32x4_t __attribute__((ext_vector_type(4)));
typedef unsigned scoria_u32x2_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint4 scoria_nt_u4(const void* ptr) {
    scoria_u32x4_t v = __builtin_nontemporal_load((const scoria_u32x4_t*)ptr);
    return make_uint4(v[0], v[1], v[2], v[3]);
}

__device__ __forceinline__ uint2 scoria_nt_u2(const void* ptr) {
    scoria_u32x2_t v = __builtin_nontemporal_load((const scoria_u32x2_t*)ptr);
    return make_uint2(v[0], v[1]);
}

__device__ __forceinline__ unsigned short scoria_nt_u16(const unsigned short* ptr) {
    return __builtin_nontemporal_load(ptr);
}

__device__ __forceinline__ int scoria_dot4_u8_i8(unsigned a_u8x4, int b_i8x4, int acc) {
#ifdef SCORIA_RDNA4
    return __builtin_amdgcn_sudot4(false, (int)a_u8x4, true, b_i8x4, acc, false);
#else
    return __builtin_amdgcn_sdot4((int)a_u8x4, b_i8x4, acc, false);
#endif
}

__device__ __forceinline__ int scoria_dot4_i8_i8(int a_i8x4, int b_i8x4, int acc) {
#ifdef SCORIA_RDNA4
    return __builtin_amdgcn_sudot4(true, a_i8x4, true, b_i8x4, acc, false);
#else
    return __builtin_amdgcn_sdot4(a_i8x4, b_i8x4, acc, false);
#endif
}

__device__ __forceinline__ float scoria_wave_sum(float v) {
    for (int offset = SCORIA_WAVE_SIZE >> 1; offset > 0; offset >>= 1) {
        v += __shfl_xor(v, offset, SCORIA_WAVE_SIZE);
    }
    return v;
}

__device__ __forceinline__ float scoria_f16_bits_to_f32(unsigned short bits) {
    union {
        unsigned short u;
        _Float16 h;
    } cvt;
    cvt.u = bits;
    return (float)cvt.h;
}

__device__ __forceinline__ unsigned short scoria_read_u16_unaligned(const unsigned char* p) {
    return (unsigned short)p[0] | ((unsigned short)p[1] << 8);
}

__device__ __forceinline__ unsigned scoria_read_u32_unaligned(const unsigned char* p) {
    return (unsigned)p[0]
        | ((unsigned)p[1] << 8)
        | ((unsigned)p[2] << 16)
        | ((unsigned)p[3] << 24);
}
