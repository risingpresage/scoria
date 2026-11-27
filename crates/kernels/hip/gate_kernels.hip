#include "arch_compat.h"

typedef float float8_t __attribute__((ext_vector_type(8)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float float4_t __attribute__((ext_vector_type(4)));
typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef int int4_t_v __attribute__((ext_vector_type(4)));
typedef int int8_t_v __attribute__((ext_vector_type(8)));

extern "C" __global__ void soak_touch(unsigned* buf, unsigned n_words, unsigned magic) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 && n_words > 0) {
        buf[0] ^= magic;
        buf[n_words - 1] ^= (magic + 1u);
    }
}

extern "C" __global__ void stream_bw_default(const uint4* __restrict__ in,
                                              uint4* __restrict__ out,
                                              unsigned n_vec4) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (unsigned i = tid; i < n_vec4; i += stride) {
        uint4 v = in[i];
        acc.x ^= v.x;
        acc.y ^= v.y;
        acc.z ^= v.z;
        acc.w ^= v.w;
    }
    if ((threadIdx.x & (SCORIA_WAVE_SIZE - 1)) == 0) {
        out[tid / SCORIA_WAVE_SIZE] = acc;
    }
}

extern "C" __global__ void stream_bw_nt(const uint4* __restrict__ in,
                                         uint4* __restrict__ out,
                                         unsigned n_vec4) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    unsigned i = tid;
    unsigned stride4 = stride * 4u;
    for (; i + stride * 3u < n_vec4; i += stride4) {
        uint4 v0 = scoria_load_nt_b128(in + i);
        uint4 v1 = scoria_load_nt_b128(in + i + stride);
        uint4 v2 = scoria_load_nt_b128(in + i + stride * 2u);
        uint4 v3 = scoria_load_nt_b128(in + i + stride * 3u);
        scoria_wait_loads();
        acc.x ^= v0.x ^ v1.x ^ v2.x ^ v3.x;
        acc.y ^= v0.y ^ v1.y ^ v2.y ^ v3.y;
        acc.z ^= v0.z ^ v1.z ^ v2.z ^ v3.z;
        acc.w ^= v0.w ^ v1.w ^ v2.w ^ v3.w;
    }
    for (; i < n_vec4; i += stride) {
        uint4 v = scoria_load_nt_b128(in + i);
        scoria_wait_loads();
        acc.x ^= v.x;
        acc.y ^= v.y;
        acc.z ^= v.z;
        acc.w ^= v.w;
    }
    if ((threadIdx.x & (SCORIA_WAVE_SIZE - 1)) == 0) {
        out[tid / SCORIA_WAVE_SIZE] = acc;
    }
}

extern "C" __global__ void l3_probe_ht(const uint4* __restrict__ hot_buf,
                                        unsigned hot_n_vec4,
                                        uint4* __restrict__ out,
                                        unsigned iters) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;
    unsigned stride4 = stride * 4u;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (unsigned r = 0; r < iters; ++r) {
        unsigned i = tid;
        for (; i + stride * 3u < hot_n_vec4; i += stride4) {
            uint4 v0 = scoria_load_ht_b128(hot_buf + i);
            uint4 v1 = scoria_load_ht_b128(hot_buf + i + stride);
            uint4 v2 = scoria_load_ht_b128(hot_buf + i + stride * 2u);
            uint4 v3 = scoria_load_ht_b128(hot_buf + i + stride * 3u);
            scoria_wait_loads();
            acc.x += v0.x + v1.x + v2.x + v3.x;
            acc.y += v0.y + v1.y + v2.y + v3.y;
            acc.z += v0.z + v1.z + v2.z + v3.z;
            acc.w += v0.w + v1.w + v2.w + v3.w;
        }
        for (; i < hot_n_vec4; i += stride) {
            uint4 v = scoria_load_ht_b128(hot_buf + i);
            scoria_wait_loads();
            acc.x += v.x;
            acc.y += v.y;
            acc.z += v.z;
            acc.w += v.w;
        }
    }
    if ((threadIdx.x & (SCORIA_WAVE_SIZE - 1)) == 0) {
        out[tid / SCORIA_WAVE_SIZE] = acc;
    }
}

#ifdef SCORIA_RDNA4
extern "C" __global__ void wmma_bench_f16(float* __restrict__ out, unsigned iters, float seed) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    _Float16 s = (_Float16)(seed + (float)(tid & 7) * 0.001f);
    half8_t a = {s, s, s, s, s, s, s, s};
    half8_t b = {s, s, s, s, s, s, s, s};

    float8_t c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    float8_t c4 = {0}, c5 = {0}, c6 = {0}, c7 = {0};

    #pragma unroll 1
    for (unsigned i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c3);
        c4 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c4);
        c5 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c5);
        c6 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c6);
        c7 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c7);
    }

    float sum = c0[0] + c1[1] + c2[2] + c3[3] + c4[4] + c5[5] + c6[6] + c7[7];
    if (tid < 256) {
        out[tid] = sum;
    }
}

extern "C" __global__ void wmma_bench_fp8(float* __restrict__ out, unsigned iters, unsigned seed) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    int packed = (int)(seed ^ (tid & 15u) ^ 0x3c3c3c3cu);
    int2_t a = {packed, packed};
    int2_t b = {packed, packed};

    float8_t c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    float8_t c4 = {0}, c5 = {0}, c6 = {0}, c7 = {0};

    #pragma unroll 1
    for (unsigned i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c3);
        c4 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c4);
        c5 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c5);
        c6 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c6);
        c7 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c7);
    }

    float sum = c0[0] + c1[1] + c2[2] + c3[3] + c4[4] + c5[5] + c6[6] + c7[7];
    if (tid < 256) {
        out[tid] = sum;
    }
}

extern "C" __global__ void wmma_bench_iu4(int* __restrict__ out, unsigned iters, unsigned seed) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    int packed = (int)(seed ^ (tid & 15u) ^ 0x11111111u);
    int2_t a = {packed, packed};
    int2_t b = {packed, packed};

    int8_t_v c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    int8_t_v c4 = {0}, c5 = {0}, c6 = {0}, c7 = {0};

    #pragma unroll 1
    for (unsigned i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c0, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c1, false);
        c2 = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c2, false);
        c3 = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c3, false);
        c4 = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c4, false);
        c5 = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c5, false);
        c6 = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c6, false);
        c7 = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c7, false);
    }

    int sum = c0[0] + c1[1] + c2[2] + c3[3] + c4[4] + c5[5] + c6[6] + c7[7];
    if (tid < 256) {
        out[tid] = sum;
    }
}

extern "C" __global__ void swmmac_bench_f16(float* __restrict__ out, unsigned iters, float seed) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    _Float16 s = (_Float16)(seed + (float)(tid & 7) * 0.001f);
    half8_t a = {s, s, s, s, s, s, s, s};
    half16_t b = {s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s};
    int idx = (int)(tid & 31u);

    float8_t c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    float8_t c4 = {0}, c5 = {0}, c6 = {0}, c7 = {0};

    #pragma unroll 1
    for (unsigned i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c0, idx);
        c1 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c1, idx);
        c2 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c2, idx);
        c3 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c3, idx);
        c4 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c4, idx);
        c5 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c5, idx);
        c6 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c6, idx);
        c7 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c7, idx);
    }

    float sum = c0[0] + c1[1] + c2[2] + c3[3] + c4[4] + c5[5] + c6[6] + c7[7];
    if (tid < 256) {
        out[tid] = sum;
    }
}

extern "C" __global__ void swmmac_bench_fp8(float* __restrict__ out, unsigned iters, unsigned seed) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    int packed = (int)(seed ^ (tid & 15u) ^ 0x3c3c3c3cu);
    int2_t a = {packed, packed};
    int4_t_v b = {packed, packed, packed, packed};
    int idx = (int)(tid & 31u);

    float8_t c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    float8_t c4 = {0}, c5 = {0}, c6 = {0}, c7 = {0};

    #pragma unroll 1
    for (unsigned i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c0, idx);
        c1 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c1, idx);
        c2 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c2, idx);
        c3 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c3, idx);
        c4 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c4, idx);
        c5 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c5, idx);
        c6 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c6, idx);
        c7 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c7, idx);
    }

    float sum = c0[0] + c1[1] + c2[2] + c3[3] + c4[4] + c5[5] + c6[6] + c7[7];
    if (tid < 256) {
        out[tid] = sum;
    }
}
#endif

#ifdef __gfx942__
extern "C" __global__ void mfma_bench_f16(float* __restrict__ out, unsigned iters, float seed) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    _Float16 s = (_Float16)(seed + (float)(tid & 7) * 0.001f);
    half4_t a = {s, s, s, s};
    half4_t b = {s, s, s, s};

    float4_t c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    float4_t c4 = {0}, c5 = {0}, c6 = {0}, c7 = {0};

    #pragma unroll 1
    for (unsigned i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c0, 0, 0, 0);
        c1 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c1, 0, 0, 0);
        c2 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c2, 0, 0, 0);
        c3 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c3, 0, 0, 0);
        c4 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c4, 0, 0, 0);
        c5 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c5, 0, 0, 0);
        c6 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c6, 0, 0, 0);
        c7 = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c7, 0, 0, 0);
    }

    float sum = c0[0] + c1[1] + c2[2] + c3[3] + c4[0] + c5[1] + c6[2] + c7[3];
    if (tid < 256) {
        out[tid] = sum;
    }
}

extern "C" __global__ void mfma_bench_fp8(float* __restrict__ out, unsigned iters, unsigned seed) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lo = seed ^ (tid & 15u) ^ 0x3c3c3c3cu;
    long long a = ((long long)lo << 32) | (long long)lo;
    long long b = a;

    float4_t c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    float4_t c4 = {0}, c5 = {0}, c6 = {0}, c7 = {0};

    #pragma unroll 1
    for (unsigned i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c0, 0, 0, 0);
        c1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c1, 0, 0, 0);
        c2 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c2, 0, 0, 0);
        c3 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c3, 0, 0, 0);
        c4 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c4, 0, 0, 0);
        c5 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c5, 0, 0, 0);
        c6 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c6, 0, 0, 0);
        c7 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c7, 0, 0, 0);
    }

    float sum = c0[0] + c1[1] + c2[2] + c3[3] + c4[0] + c5[1] + c6[2] + c7[3];
    if (tid < 256) {
        out[tid] = sum;
    }
}

extern "C" __global__ void mfma_bench_i8(int* __restrict__ out, unsigned iters, unsigned seed) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lo = seed ^ (tid & 15u) ^ 0x11111111u;
    long long a = ((long long)lo << 32) | (long long)lo;
    long long b = a;

    int4_t_v c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    int4_t_v c4 = {0}, c5 = {0}, c6 = {0}, c7 = {0};

    #pragma unroll 1
    for (unsigned i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c0, 0, 0, 0);
        c1 = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c1, 0, 0, 0);
        c2 = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c2, 0, 0, 0);
        c3 = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c3, 0, 0, 0);
        c4 = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c4, 0, 0, 0);
        c5 = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c5, 0, 0, 0);
        c6 = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c6, 0, 0, 0);
        c7 = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, c7, 0, 0, 0);
    }

    int sum = c0[0] + c1[1] + c2[2] + c3[3] + c4[0] + c5[1] + c6[2] + c7[3];
    if (tid < 256) {
        out[tid] = sum;
    }
}
#endif

extern "C" __global__ void lut_bench_permlane(const unsigned* __restrict__ packed_nibbles,
                                              float* __restrict__ out,
                                              unsigned n_words,
                                              unsigned iters) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;
    unsigned lane = threadIdx.x & 15u;
    float lut_val = ((float)lane - 7.5f) * 0.125f;
    unsigned lut_bits = __float_as_uint(lut_val);

    float acc = 0.0f;
    for (unsigned r = 0; r < iters; ++r) {
        for (unsigned i = tid; i < n_words; i += stride) {
            unsigned w = packed_nibbles[i];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                unsigned nib = (w >> (k * 4)) & 0xfu;
                unsigned gathered = scoria_lut16(lut_bits, nib);
                acc += __uint_as_float(gathered);
            }
        }
    }
    if ((threadIdx.x & (SCORIA_WAVE_SIZE - 1)) == 0) {
        out[tid / SCORIA_WAVE_SIZE] = acc;
    }
}

extern "C" __global__ void lut_bench_lds(const unsigned* __restrict__ packed_nibbles,
                                         float* __restrict__ out,
                                         unsigned n_words,
                                         unsigned iters) {
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) {
        s_lut[threadIdx.x] = ((float)threadIdx.x - 7.5f) * 0.125f;
    }
    __syncthreads();

    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (unsigned r = 0; r < iters; ++r) {
        for (unsigned i = tid; i < n_words; i += stride) {
            unsigned w = packed_nibbles[i];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                unsigned nib = (w >> (k * 4)) & 0xfu;
                acc += s_lut[nib];
            }
        }
    }
    if ((threadIdx.x & (SCORIA_WAVE_SIZE - 1)) == 0) {
        out[tid / SCORIA_WAVE_SIZE] = acc;
    }
}

extern "C" __global__ void bitcos_bench(const unsigned* __restrict__ bitmaps,
                                        const unsigned* __restrict__ signs,
                                        float* __restrict__ out,
                                        unsigned n_words,
                                        unsigned iters) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;
    int acc = 0;
    for (unsigned r = 0; r < iters; ++r) {
        for (unsigned i = tid; i < n_words; i += stride) {
            unsigned bm = bitmaps[i];
            unsigned sg = signs[i];
            unsigned nz = __popc(bm);
            unsigned ones = __popc(bm & sg);
            int neg = (int)ones;
            int pos = (int)(nz - ones);
            acc += (pos - neg);
        }
    }
    if ((threadIdx.x & (SCORIA_WAVE_SIZE - 1)) == 0) {
        out[tid / SCORIA_WAVE_SIZE] = (float)acc;
    }
}
