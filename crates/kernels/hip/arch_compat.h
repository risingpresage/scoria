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
