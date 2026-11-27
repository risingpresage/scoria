#include "arch_compat.h"

__device__ __forceinline__ void scoria_get_scale_min_k4(int j,
                                                        const unsigned char* q,
                                                        unsigned char* d_out,
                                                        unsigned char* m_out) {
    if (j < 4) {
        *d_out = q[j] & 63u;
        *m_out = q[j + 4] & 63u;
    } else {
        *d_out = (q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4);
        *m_out = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

extern "C" __global__ void embed_lookup_f32(const unsigned char* __restrict__ table,
                                            const unsigned* __restrict__ token_ids,
                                            float* __restrict__ out,
                                            unsigned n_tokens,
                                            unsigned n_embd) {
    unsigned tok = blockIdx.x;
    if (tok >= n_tokens) return;
    unsigned tid = threadIdx.x;
    const float* row = (const float*)table + (size_t)token_ids[tok] * n_embd;
    float* dst = out + (size_t)tok * n_embd;
    for (unsigned i = tid; i < n_embd; i += blockDim.x) {
        dst[i] = row[i];
    }
}

extern "C" __global__ void embed_lookup_f16(const unsigned char* __restrict__ table,
                                            const unsigned* __restrict__ token_ids,
                                            float* __restrict__ out,
                                            unsigned n_tokens,
                                            unsigned n_embd) {
    unsigned tok = blockIdx.x;
    if (tok >= n_tokens) return;
    unsigned tid = threadIdx.x;
    const unsigned char* row = table + (size_t)token_ids[tok] * n_embd * 2u;
    float* dst = out + (size_t)tok * n_embd;
    for (unsigned i = tid; i < n_embd; i += blockDim.x) {
        dst[i] = scoria_f16_bits_to_f32(scoria_read_u16_unaligned(row + (size_t)i * 2u));
    }
}

extern "C" __global__ void embed_lookup_q8_0(const unsigned char* __restrict__ table,
                                             const unsigned* __restrict__ token_ids,
                                             float* __restrict__ out,
                                             unsigned n_tokens,
                                             unsigned n_embd) {
    unsigned tok = blockIdx.x;
    if (tok >= n_tokens) return;
    unsigned tid = threadIdx.x;
    unsigned nb = n_embd / 32u;
    const unsigned char* row = table + (size_t)token_ids[tok] * nb * 34u;
    float* dst = out + (size_t)tok * n_embd;
    for (unsigned idx = tid; idx < n_embd; idx += blockDim.x) {
        unsigned b = idx >> 5;
        unsigned j = idx & 31u;
        const unsigned char* blk = row + (size_t)b * 34u;
        float d = scoria_f16_bits_to_f32(scoria_read_u16_unaligned(blk));
        signed char q = (signed char)blk[2u + j];
        dst[idx] = d * (float)q;
    }
}

extern "C" __global__ void embed_lookup_q4_k(const unsigned char* __restrict__ table,
                                             const unsigned* __restrict__ token_ids,
                                             float* __restrict__ out,
                                             unsigned n_tokens,
                                             unsigned n_embd) {
    unsigned tok = blockIdx.x;
    if (tok >= n_tokens) return;
    unsigned tid = threadIdx.x;
    unsigned nb = n_embd / 256u;
    const unsigned char* row = table + (size_t)token_ids[tok] * nb * 144u;
    float* dst = out + (size_t)tok * n_embd;
    for (unsigned b = tid; b < nb; b += blockDim.x) {
        const unsigned char* blk = row + (size_t)b * 144u;
        float d = scoria_f16_bits_to_f32(scoria_read_u16_unaligned(blk));
        float dmin = scoria_f16_bits_to_f32(scoria_read_u16_unaligned(blk + 2));
        const unsigned char* scales = blk + 4;
        const unsigned char* qs = blk + 16;
        float* y = dst + (size_t)b * 256u;
        int is = 0;
        for (int c = 0; c < 4; ++c) {
            unsigned char sc1, m1, sc2, m2;
            scoria_get_scale_min_k4(is, scales, &sc1, &m1);
            scoria_get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1 = d * (float)sc1;
            float min1 = dmin * (float)m1;
            float d2 = d * (float)sc2;
            float min2 = dmin * (float)m2;
            const unsigned char* q = qs + c * 32;
            for (int l = 0; l < 32; ++l) {
                y[c * 64 + l] = d1 * (float)(q[l] & 0x0Fu) - min1;
                y[c * 64 + 32 + l] = d2 * (float)(q[l] >> 4) - min2;
            }
            is += 2;
        }
    }
}

extern "C" __global__ void embed_lookup_q6_k(const unsigned char* __restrict__ table,
                                             const unsigned* __restrict__ token_ids,
                                             float* __restrict__ out,
                                             unsigned n_tokens,
                                             unsigned n_embd) {
    unsigned tok = blockIdx.x;
    if (tok >= n_tokens) return;
    unsigned tid = threadIdx.x;
    unsigned nb = n_embd / 256u;
    const unsigned char* row = table + (size_t)token_ids[tok] * nb * 210u;
    float* dst = out + (size_t)tok * n_embd;
    for (unsigned b = tid; b < nb; b += blockDim.x) {
        const unsigned char* blk = row + (size_t)b * 210u;
        float d = scoria_f16_bits_to_f32(scoria_read_u16_unaligned(blk + 208u));
        const signed char* sc = (const signed char*)(blk + 192u);
        float* y = dst + (size_t)b * 256u;
        for (unsigned half = 0; half < 2u; ++half) {
            const unsigned char* ql = blk + (size_t)half * 64u;
            const unsigned char* qh = blk + 128u + (size_t)half * 32u;
            const signed char* s = sc + (size_t)half * 8u;
            float* yy = y + (size_t)half * 128u;
            for (unsigned l = 0; l < 32u; ++l) {
                unsigned is = l >> 4;
                int q1 = (int)((ql[l] & 0x0Fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                int q2 = (int)((ql[l + 32u] & 0x0Fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                int q4 = (int)((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
                yy[l] = d * (float)s[is + 0u] * (float)q1;
                yy[l + 32u] = d * (float)s[is + 2u] * (float)q2;
                yy[l + 64u] = d * (float)s[is + 4u] * (float)q3;
                yy[l + 96u] = d * (float)s[is + 6u] * (float)q4;
            }
        }
    }
}

extern "C" __global__ void rmsnorm_f32(const float* __restrict__ x,
                                       const float* __restrict__ w,
                                       float* __restrict__ out,
                                       unsigned n_tokens,
                                       unsigned n_embd,
                                       float eps) {
    unsigned tok = blockIdx.x;
    if (tok >= n_tokens) return;
    unsigned tid = threadIdx.x;
    unsigned lane = tid & (SCORIA_WAVE_SIZE - 1u);
    unsigned wave_id = tid / SCORIA_WAVE_SIZE;
    unsigned n_waves = blockDim.x / SCORIA_WAVE_SIZE;

    const float* x_row = x + (size_t)tok * n_embd;
    float* out_row = out + (size_t)tok * n_embd;

    float ss = 0.0f;
    if ((n_embd & 3u) == 0u) {
        unsigned n4 = n_embd >> 2;
        const float4* x4 = (const float4*)x_row;
        for (unsigned i = tid; i < n4; i += blockDim.x) {
            float4 v = x4[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (unsigned i = tid; i < n_embd; i += blockDim.x) {
            float v = x_row[i];
            ss += v * v;
        }
    }
    float wsum = scoria_wave_sum(ss);
    __shared__ float s_sums[32];
    __shared__ float s_inv;
    if (lane == 0) {
        s_sums[wave_id] = wsum;
    }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (unsigned k = 0; k < n_waves; ++k) {
            total += s_sums[k];
        }
        s_inv = rsqrtf(total / (float)n_embd + eps);
    }
    __syncthreads();
    float inv = s_inv;
    if ((n_embd & 3u) == 0u) {
        unsigned n4 = n_embd >> 2;
        const float4* x4 = (const float4*)x_row;
        const float4* w4 = (const float4*)w;
        float4* out4 = (float4*)out_row;
        for (unsigned i = tid; i < n4; i += blockDim.x) {
            float4 xv = x4[i];
            float4 wv = w4[i];
            float4 ov = {xv.x * inv * wv.x, xv.y * inv * wv.y, xv.z * inv * wv.z, xv.w * inv * wv.w};
            out4[i] = ov;
        }
    } else {
        for (unsigned i = tid; i < n_embd; i += blockDim.x) {
            out_row[i] = x_row[i] * inv * w[i];
        }
    }
}

// Qwen3 per-head RMSNorm over `head_dim` for Q and K, applied before RoPE.
// One wave per (token, head); head_dim is typically 128 so each lane owns one float4.
extern "C" __global__ void qk_rmsnorm_f32(float* __restrict__ q,
                                          float* __restrict__ k,
                                          const float* __restrict__ q_w,
                                          const float* __restrict__ k_w,
                                          unsigned m,
                                          unsigned n_heads,
                                          unsigned n_kv_heads,
                                          unsigned head_dim,
                                          float eps) {
    unsigned tok = blockIdx.x;
    unsigned h = blockIdx.y;
    if (tok >= m || h >= (n_heads + n_kv_heads)) return;
    unsigned lane = threadIdx.x;
    if (lane >= SCORIA_WAVE_SIZE) return;

    float* vec;
    const float* w;
    if (h < n_heads) {
        vec = q + ((size_t)tok * n_heads + h) * head_dim;
        w = q_w;
    } else {
        vec = k + ((size_t)tok * n_kv_heads + (h - n_heads)) * head_dim;
        w = k_w;
    }

    float ss = 0.0f;
    if ((head_dim & 3u) == 0u) {
        unsigned n4 = head_dim >> 2;
        const float4* v4 = (const float4*)vec;
        for (unsigned i = lane; i < n4; i += SCORIA_WAVE_SIZE) {
            float4 v = v4[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        float total = scoria_wave_sum(ss);
        float inv = rsqrtf(total / (float)head_dim + eps);
        float4* o4 = (float4*)vec;
        const float4* w4 = (const float4*)w;
        for (unsigned i = lane; i < n4; i += SCORIA_WAVE_SIZE) {
            float4 v = o4[i];
            float4 wv = w4[i];
            float4 ov = {v.x * inv * wv.x, v.y * inv * wv.y, v.z * inv * wv.z, v.w * inv * wv.w};
            o4[i] = ov;
        }
    } else {
        for (unsigned i = lane; i < head_dim; i += SCORIA_WAVE_SIZE) {
            float v = vec[i];
            ss += v * v;
        }
        float total = scoria_wave_sum(ss);
        float inv = rsqrtf(total / (float)head_dim + eps);
        for (unsigned i = lane; i < head_dim; i += SCORIA_WAVE_SIZE) {
            vec[i] = vec[i] * inv * w[i];
        }
    }
}

extern "C" __global__ void linear_f32(const unsigned char* __restrict__ weight,
                                      const float* __restrict__ x,
                                      const float* __restrict__ bias,
                                      float* __restrict__ y,
                                      unsigned m,
                                      unsigned n,
                                      unsigned k) {
    unsigned wave_id = threadIdx.x / SCORIA_WAVE_SIZE;
    unsigned lane = threadIdx.x & (SCORIA_WAVE_SIZE - 1u);
    unsigned waves_per_wg = blockDim.x / SCORIA_WAVE_SIZE;
    unsigned row = blockIdx.x * waves_per_wg + wave_id;
    unsigned tok = blockIdx.y;
    if (row >= n || tok >= m) return;

    const float* w_row = (const float*)weight + (size_t)row * k;
    const float* x_row = x + (size_t)tok * k;
    float acc = 0.0f;
    for (unsigned i = lane; i < k; i += SCORIA_WAVE_SIZE) {
        acc += w_row[i] * x_row[i];
    }
    float sum = scoria_wave_sum(acc);
    if (lane == 0) {
        y[(size_t)tok * n + row] = sum + (bias ? bias[row] : 0.0f);
    }
}

extern "C" __global__ void linear_f16(const unsigned char* __restrict__ weight,
                                      const float* __restrict__ x,
                                      const float* __restrict__ bias,
                                      float* __restrict__ y,
                                      unsigned m,
                                      unsigned n,
                                      unsigned k) {
    unsigned wave_id = threadIdx.x / SCORIA_WAVE_SIZE;
    unsigned lane = threadIdx.x & (SCORIA_WAVE_SIZE - 1u);
    unsigned waves_per_wg = blockDim.x / SCORIA_WAVE_SIZE;
    unsigned row = blockIdx.x * waves_per_wg + wave_id;
    unsigned tok = blockIdx.y;
    if (row >= n || tok >= m) return;

    const unsigned char* w_row = weight + (size_t)row * k * 2u;
    const float* x_row = x + (size_t)tok * k;
    float acc = 0.0f;
    for (unsigned i = lane; i < k; i += SCORIA_WAVE_SIZE) {
        float w = scoria_f16_bits_to_f32(scoria_read_u16_unaligned(w_row + (size_t)i * 2u));
        acc += w * x_row[i];
    }
    float sum = scoria_wave_sum(acc);
    if (lane == 0) {
        y[(size_t)tok * n + row] = sum + (bias ? bias[row] : 0.0f);
    }
}

extern "C" __global__ void linear_q5_0(const unsigned char* __restrict__ weight,
                                       const float* __restrict__ x,
                                       const float* __restrict__ bias,
                                       float* __restrict__ y,
                                       unsigned m,
                                       unsigned n,
                                       unsigned k) {
    unsigned wave_id = threadIdx.x / SCORIA_WAVE_SIZE;
    unsigned lane = threadIdx.x & (SCORIA_WAVE_SIZE - 1u);
    unsigned waves_per_wg = blockDim.x / SCORIA_WAVE_SIZE;
    unsigned row = blockIdx.x * waves_per_wg + wave_id;
    unsigned tok = blockIdx.y;
    if (row >= n || tok >= m) return;

    unsigned nb = k / 32u;
    const unsigned char* w_row = weight + (size_t)row * nb * 22u;
    const float* x_row = x + (size_t)tok * k;
    float acc = 0.0f;
    for (unsigned b = lane; b < nb; b += SCORIA_WAVE_SIZE) {
        const unsigned char* blk = w_row + (size_t)b * 22u;
        float d = scoria_f16_bits_to_f32(scoria_read_u16_unaligned(blk));
        unsigned qh = scoria_read_u32_unaligned(blk + 2);
        const unsigned char* qs = blk + 6;
        const float* xp = x_row + (size_t)b * 32u;
        float blk_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            unsigned xh_0 = ((qh >> j) << 4) & 0x10u;
            unsigned xh_1 = (qh >> (j + 12)) & 0x10u;
            int x0 = (int)((qs[j] & 0x0Fu) | xh_0) - 16;
            int x1 = (int)((qs[j] >> 4) | xh_1) - 16;
            blk_sum += (float)x0 * xp[j] + (float)x1 * xp[j + 16];
        }
        acc += d * blk_sum;
    }
    float sum = scoria_wave_sum(acc);
    if (lane == 0) {
        y[(size_t)tok * n + row] = sum + (bias ? bias[row] : 0.0f);
    }
}

extern "C" __global__ void linear_q8_0(const unsigned char* __restrict__ weight,
                                       const float* __restrict__ x,
                                       const float* __restrict__ bias,
                                       float* __restrict__ y,
                                       unsigned m,
                                       unsigned n,
                                       unsigned k) {
    unsigned wave_id = threadIdx.x / SCORIA_WAVE_SIZE;
    unsigned lane = threadIdx.x & (SCORIA_WAVE_SIZE - 1u);
    unsigned waves_per_wg = blockDim.x / SCORIA_WAVE_SIZE;
    unsigned row = blockIdx.x * waves_per_wg + wave_id;
    unsigned tok = blockIdx.y;
    if (row >= n || tok >= m) return;

    unsigned nb = k / 32u;
    const unsigned char* w_row = weight + (size_t)row * nb * 34u;
    const float* x_row = x + (size_t)tok * k;
    float acc = 0.0f;
    for (unsigned b = lane; b < nb; b += SCORIA_WAVE_SIZE) {
        const unsigned char* blk = w_row + (size_t)b * 34u;
        float d = scoria_f16_bits_to_f32(scoria_read_u16_unaligned(blk));
        const signed char* qs = (const signed char*)(blk + 2);
        const float* xp = x_row + (size_t)b * 32u;
        float blk_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            blk_sum += (float)qs[j] * xp[j];
        }
        acc += d * blk_sum;
    }
    float sum = scoria_wave_sum(acc);
    if (lane == 0) {
        y[(size_t)tok * n + row] = sum + (bias ? bias[row] : 0.0f);
    }
}

extern "C" __global__ void linear_q4_k(const unsigned char* __restrict__ weight,
                                       const float* __restrict__ x,
                                       const float* __restrict__ bias,
                                       float* __restrict__ y,
                                       unsigned m,
                                       unsigned n,
                                       unsigned k) {
    unsigned wave_id = threadIdx.x / SCORIA_WAVE_SIZE;
    unsigned lane = threadIdx.x & (SCORIA_WAVE_SIZE - 1u);
    unsigned waves_per_wg = blockDim.x / SCORIA_WAVE_SIZE;
    unsigned row = blockIdx.x * waves_per_wg + wave_id;
    unsigned tok = blockIdx.y;
    if (row >= n || tok >= m) return;

    unsigned nb = k / 256u;
    const unsigned char* w_row = weight + (size_t)row * nb * 144u;
    const float* x_row = x + (size_t)tok * k;
    float acc = 0.0f;
    for (unsigned b = lane; b < nb; b += SCORIA_WAVE_SIZE) {
        const unsigned char* blk = w_row + (size_t)b * 144u;
        float d = scoria_f16_bits_to_f32(scoria_read_u16_unaligned(blk));
        float dmin = scoria_f16_bits_to_f32(scoria_read_u16_unaligned(blk + 2));
        const unsigned char* scales = blk + 4;
        const unsigned char* qs = blk + 16;
        const float* xp = x_row + (size_t)b * 256u;
        int is = 0;
        for (int c = 0; c < 4; ++c) {
            unsigned char sc1, m1, sc2, m2;
            scoria_get_scale_min_k4(is, scales, &sc1, &m1);
            scoria_get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1 = d * (float)sc1;
            float min1 = dmin * (float)m1;
            float d2 = d * (float)sc2;
            float min2 = dmin * (float)m2;
            const unsigned char* q = qs + c * 32;
            const float* x_c = xp + c * 64;
            float s1 = 0.0f, s2 = 0.0f;
            float xs1 = 0.0f, xs2 = 0.0f;
            #pragma unroll
            for (int l = 0; l < 32; ++l) {
                float xl1 = x_c[l];
                float xl2 = x_c[l + 32];
                s1 += (float)(q[l] & 0x0Fu) * xl1;
                s2 += (float)(q[l] >> 4) * xl2;
                xs1 += xl1;
                xs2 += xl2;
            }
            acc += (d1 * s1 - min1 * xs1) + (d2 * s2 - min2 * xs2);
            is += 2;
        }
    }
    float sum = scoria_wave_sum(acc);
    if (lane == 0) {
        y[(size_t)tok * n + row] = sum + (bias ? bias[row] : 0.0f);
    }
}

extern "C" __global__ void linear_q6_k(const unsigned char* __restrict__ weight,
                                       const float* __restrict__ x,
                                       const float* __restrict__ bias,
                                       float* __restrict__ y,
                                       unsigned m,
                                       unsigned n,
                                       unsigned k) {
    unsigned wave_id = threadIdx.x / SCORIA_WAVE_SIZE;
    unsigned lane = threadIdx.x & (SCORIA_WAVE_SIZE - 1u);
    unsigned waves_per_wg = blockDim.x / SCORIA_WAVE_SIZE;
    unsigned row = blockIdx.x * waves_per_wg + wave_id;
    unsigned tok = blockIdx.y;
    if (row >= n || tok >= m) return;

    unsigned nb = k / 256u;
    const unsigned char* w_row = weight + (size_t)row * nb * 210u;
    const float* x_row = x + (size_t)tok * k;
    float acc = 0.0f;
    for (unsigned b = lane; b < nb; b += SCORIA_WAVE_SIZE) {
        const unsigned char* blk = w_row + (size_t)b * 210u;
        const unsigned char* ql_all = blk;
        const unsigned char* qh_all = blk + 128;
        const signed char* sc_all = (const signed char*)(blk + 192);
        float d = scoria_f16_bits_to_f32(scoria_read_u16_unaligned(blk + 208));
        const float* xp = x_row + (size_t)b * 256u;

        float blk_acc = 0.0f;
        for (int h = 0; h < 2; ++h) {
            const unsigned char* ql = ql_all + h * 64;
            const unsigned char* qh = qh_all + h * 32;
            const signed char* sc = sc_all + h * 8;
            const float* xh = xp + h * 128;
            for (int l = 0; l < 32; ++l) {
                int is = l >> 4;
                int q1 = (int)((ql[l] & 0x0Fu) | ((qh[l] & 0x03u) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0x0Fu) | (((qh[l] >> 2) & 0x03u) << 4)) - 32;
                int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 0x03u) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03u) << 4)) - 32;
                blk_acc += (float)sc[is + 0] * (float)q1 * xh[l + 0]
                         + (float)sc[is + 2] * (float)q2 * xh[l + 32]
                         + (float)sc[is + 4] * (float)q3 * xh[l + 64]
                         + (float)sc[is + 6] * (float)q4 * xh[l + 96];
            }
        }
        acc += d * blk_acc;
    }
    float sum = scoria_wave_sum(acc);
    if (lane == 0) {
        y[(size_t)tok * n + row] = sum + (bias ? bias[row] : 0.0f);
    }
}

extern "C" __global__ void rope_neox_f32(float* __restrict__ q,
                                         float* __restrict__ k,
                                         const unsigned* __restrict__ pos_ptr,
                                         unsigned m,
                                         unsigned n_heads,
                                         unsigned n_kv_heads,
                                         unsigned head_dim,
                                         float freq_base) {
    unsigned tok = blockIdx.x;
    unsigned h = blockIdx.y;
    if (tok >= m || h >= (n_heads + n_kv_heads)) return;
    unsigned start_pos = *pos_ptr;

    unsigned half_dim = head_dim >> 1;
    unsigned i = threadIdx.x;
    if (i >= half_dim) return;

    float* vec;
    if (h < n_heads) {
        vec = q + ((size_t)tok * n_heads + h) * head_dim;
    } else {
        unsigned kvh = h - n_heads;
        vec = k + ((size_t)tok * n_kv_heads + kvh) * head_dim;
    }

    float inv_freq = exp2f((-2.0f / (float)head_dim) * (float)i * log2f(freq_base));
    float theta = (float)(start_pos + tok) * inv_freq;
    float s, c;
    sincosf(theta, &s, &c);

    float x0 = vec[i];
    float x1 = vec[i + half_dim];
    vec[i] = x0 * c - x1 * s;
    vec[i + half_dim] = x0 * s + x1 * c;
}

extern "C" __global__ void kv_cache_append_f32(const float* __restrict__ k_in,
                                               const float* __restrict__ v_in,
                                               float* __restrict__ k_cache,
                                               float* __restrict__ v_cache,
                                               const unsigned* __restrict__ pos_ptr,
                                               unsigned m,
                                               unsigned kv_dim) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = m * kv_dim;
    if (idx >= total) return;
    size_t dst_idx = (size_t)(*pos_ptr) * kv_dim + idx;
    k_cache[dst_idx] = k_in[idx];
    v_cache[dst_idx] = v_in[idx];
}

// Fused optional per-head Q/K RMSNorm + NeoX RoPE + KV cache append.
// Uses float2 vectorized loads and warp reductions on Wave32 when head_dim == 128.
// Stores two adjacent cache elements, keeping the 64-bit `float2` store on the FP32
// path and letting the compiler merge the two half stores on the FP16 path.
template <typename KV>
__device__ __forceinline__ void scoria_store2(KV* p, float2 v) {
    if constexpr (sizeof(KV) == 4) {
        *(float2*)p = v;
    } else {
        p[0] = (KV)v.x;
        p[1] = (KV)v.y;
    }
}

template <typename KV>
__device__ __forceinline__ void qk_norm_rope_kv_impl(float* __restrict__ q,
                                                     const float* __restrict__ k_in,
                                                     const float* __restrict__ v_in,
                                                     const float* __restrict__ q_norm_w,
                                                     const float* __restrict__ k_norm_w,
                                                     KV* __restrict__ k_cache,
                                                     KV* __restrict__ v_cache,
                                               const unsigned* __restrict__ pos_ptr,
                                               unsigned m,
                                               unsigned n_heads,
                                               unsigned n_kv_heads,
                                               unsigned head_dim,
                                               float freq_base,
                                               float eps) {
    unsigned tok = blockIdx.x;
    unsigned h = blockIdx.y;
    if (tok >= m || h >= (n_heads + n_kv_heads)) return;
    unsigned lane = threadIdx.x;
    if (lane >= SCORIA_WAVE_SIZE) return;

    unsigned pos = *pos_ptr + tok;
    bool is_q = h < n_heads;

#ifdef SCORIA_RDNA4
    if (head_dim == 128u) {
        unsigned i0 = lane * 2u;
        float log2_base = log2f(freq_base);
        float inv_f0 = exp2f((-2.0f / 128.0f) * (float)i0 * log2_base);
        float inv_f1 = exp2f((-2.0f / 128.0f) * (float)(i0 + 1u) * log2_base);
        float s0, c0, s1, c1;
        sincosf((float)pos * inv_f0, &s0, &c0);
        sincosf((float)pos * inv_f1, &s1, &c1);

        if (is_q) {
            float* qv = q + ((size_t)tok * n_heads + h) * 128u;
            float2 lo = *(const float2*)(qv + i0);
            float2 hi = *(const float2*)(qv + 64u + i0);
            if (q_norm_w != nullptr) {
                float ss = lo.x * lo.x + lo.y * lo.y + hi.x * hi.x + hi.y * hi.y;
                float inv = rsqrtf(scoria_wave_sum(ss) * (1.0f / 128.0f) + eps);
                float2 wlo = *(const float2*)(q_norm_w + i0);
                float2 whi = *(const float2*)(q_norm_w + 64u + i0);
                lo.x *= inv * wlo.x;
                lo.y *= inv * wlo.y;
                hi.x *= inv * whi.x;
                hi.y *= inv * whi.y;
            }
            float2 out_lo = {lo.x * c0 - hi.x * s0, lo.y * c1 - hi.y * s1};
            float2 out_hi = {lo.x * s0 + hi.x * c0, lo.y * s1 + hi.y * c1};
            *(float2*)(qv + i0) = out_lo;
            *(float2*)(qv + 64u + i0) = out_hi;
        } else {
            unsigned kvh = h - n_heads;
            const float* kv = k_in + ((size_t)tok * n_kv_heads + kvh) * 128u;
            const float* vv = v_in + ((size_t)tok * n_kv_heads + kvh) * 128u;
            KV* kc = k_cache + ((size_t)pos * n_kv_heads + kvh) * 128u;
            KV* vc = v_cache + ((size_t)pos * n_kv_heads + kvh) * 128u;

            float2 lo = *(const float2*)(kv + i0);
            float2 hi = *(const float2*)(kv + 64u + i0);
            float2 vlo = *(const float2*)(vv + i0);
            float2 vhi = *(const float2*)(vv + 64u + i0);
            scoria_store2(vc + i0, vlo);
            scoria_store2(vc + 64u + i0, vhi);

            if (k_norm_w != nullptr) {
                float ss = lo.x * lo.x + lo.y * lo.y + hi.x * hi.x + hi.y * hi.y;
                float inv = rsqrtf(scoria_wave_sum(ss) * (1.0f / 128.0f) + eps);
                float2 wlo = *(const float2*)(k_norm_w + i0);
                float2 whi = *(const float2*)(k_norm_w + 64u + i0);
                lo.x *= inv * wlo.x;
                lo.y *= inv * wlo.y;
                hi.x *= inv * whi.x;
                hi.y *= inv * whi.y;
            }
            float2 out_lo = {lo.x * c0 - hi.x * s0, lo.y * c1 - hi.y * s1};
            float2 out_hi = {lo.x * s0 + hi.x * c0, lo.y * s1 + hi.y * c1};
            scoria_store2(kc + i0, out_lo);
            scoria_store2(kc + 64u + i0, out_hi);
        }
        return;
    }
#endif

    unsigned half_dim = head_dim >> 1;
    const float* src = is_q
        ? (q + ((size_t)tok * n_heads + h) * head_dim)
        : (k_in + ((size_t)tok * n_kv_heads + (h - n_heads)) * head_dim);
    const float* norm_w = is_q ? q_norm_w : k_norm_w;
    float* q_dst = is_q ? (q + ((size_t)tok * n_heads + h) * head_dim) : nullptr;
    KV* kv_dst = is_q
        ? nullptr
        : (k_cache + ((size_t)pos * n_kv_heads + (h - n_heads)) * head_dim);

    float inv = 1.0f;
    if (norm_w != nullptr) {
        float ss = 0.0f;
        for (unsigned i = lane; i < half_dim; i += SCORIA_WAVE_SIZE) {
            float x0 = src[i];
            float x1 = src[i + half_dim];
            ss += x0 * x0 + x1 * x1;
        }
        inv = rsqrtf(scoria_wave_sum(ss) / (float)head_dim + eps);
    }

    float log2_base = log2f(freq_base);
    float inv_step = -2.0f / (float)head_dim;
    for (unsigned i = lane; i < half_dim; i += SCORIA_WAVE_SIZE) {
        float x0 = src[i];
        float x1 = src[i + half_dim];
        if (norm_w != nullptr) {
            x0 *= inv * norm_w[i];
            x1 *= inv * norm_w[i + half_dim];
        }
        float inv_freq = exp2f(inv_step * (float)i * log2_base);
        float theta = (float)pos * inv_freq;
        float s, c;
        sincosf(theta, &s, &c);
        float o0 = x0 * c - x1 * s;
        float o1 = x0 * s + x1 * c;
        if (is_q) {
            q_dst[i] = o0;
            q_dst[i + half_dim] = o1;
        } else {
            kv_dst[i] = (KV)o0;
            kv_dst[i + half_dim] = (KV)o1;
        }
    }

    if (!is_q) {
        unsigned kvh = h - n_heads;
        const float* vv = v_in + ((size_t)tok * n_kv_heads + kvh) * head_dim;
        KV* vc = v_cache + ((size_t)pos * n_kv_heads + kvh) * head_dim;
        for (unsigned i = lane; i < head_dim; i += SCORIA_WAVE_SIZE) {
            vc[i] = (KV)vv[i];
        }
    }
}

extern "C" __global__ void gqa_attn_f32(const float* __restrict__ q,
                                        const float* __restrict__ k_cache,
                                        const float* __restrict__ v_cache,
                                        float* __restrict__ out,
                                        const unsigned* __restrict__ pos_ptr,
                                        unsigned m,
                                        unsigned n_heads,
                                        unsigned n_kv_heads,
                                        unsigned head_dim) {
    unsigned q_head = blockIdx.x;
    unsigned tok = blockIdx.y;
    if (q_head >= n_heads || tok >= m) return;
    unsigned start_pos = *pos_ptr;

    unsigned tid = threadIdx.x;
    unsigned lane = tid & (SCORIA_WAVE_SIZE - 1u);
    unsigned wave_id = tid / SCORIA_WAVE_SIZE;
    unsigned n_waves = blockDim.x / SCORIA_WAVE_SIZE;

    unsigned kv_group = n_heads / n_kv_heads;
    unsigned kv_head = q_head / kv_group;
    unsigned kv_len = start_pos + tok + 1u;
    float scale = rsqrtf((float)head_dim);

    const float* q_base = q + ((size_t)tok * n_heads + q_head) * head_dim;
    float* out_base = out + ((size_t)tok * n_heads + q_head) * head_dim;

    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_o[4][256];

    float m_i = -1e38f;
    float l_i = 0.0f;

#ifdef SCORIA_RDNA4
    if (head_dim == 128u) {
        unsigned d = lane * 4u;
        float4 qv = *(const float4*)(q_base + d);
        float4 av = {0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned t = wave_id; t < kv_len; t += n_waves) {
            const float* k_ptr = k_cache + ((size_t)t * n_kv_heads + kv_head) * 128u;
            const float* v_ptr = v_cache + ((size_t)t * n_kv_heads + kv_head) * 128u;
            float4 kv = *(const float4*)(k_ptr + d);
            float p = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            float score = scoria_wave_sum(p) * scale;
            float m_new = fmaxf(m_i, score);
            float alpha = (l_i == 0.0f) ? 0.0f : expf(m_i - m_new);
            float beta = expf(score - m_new);
            l_i = l_i * alpha + beta;
            float4 vv = *(const float4*)(v_ptr + d);
            av.x = av.x * alpha + beta * vv.x;
            av.y = av.y * alpha + beta * vv.y;
            av.z = av.z * alpha + beta * vv.z;
            av.w = av.w * alpha + beta * vv.w;
            m_i = m_new;
        }
        if (lane == 0) {
            s_m[wave_id] = m_i;
            s_l[wave_id] = l_i;
        }
        *(float4*)(&s_o[wave_id][d]) = av;
        __syncthreads();

        if (wave_id == 0) {
            float m_g = -1e38f;
            for (unsigned w = 0; w < n_waves; ++w) {
                if (s_l[w] > 0.0f) {
                    m_g = fmaxf(m_g, s_m[w]);
                }
            }
            float l_g = 0.0f;
            float4 out_v = {0.0f, 0.0f, 0.0f, 0.0f};
            for (unsigned w = 0; w < n_waves; ++w) {
                if (s_l[w] > 0.0f) {
                    float bw = expf(s_m[w] - m_g);
                    l_g += s_l[w] * bw;
                    float4 ov = *(const float4*)(&s_o[w][d]);
                    out_v.x += ov.x * bw;
                    out_v.y += ov.y * bw;
                    out_v.z += ov.z * bw;
                    out_v.w += ov.w * bw;
                }
            }
            float inv_l = 1.0f / l_g;
            out_v.x *= inv_l;
            out_v.y *= inv_l;
            out_v.z *= inv_l;
            out_v.w *= inv_l;
            *(float4*)(out_base + d) = out_v;
        }
        return;
    }

    if (head_dim == 64u) {
        unsigned d = lane * 2u;
        float2 qv = *(const float2*)(q_base + d);
        float2 av = {0.0f, 0.0f};
        for (unsigned t = wave_id; t < kv_len; t += n_waves) {
            const float* k_ptr = k_cache + ((size_t)t * n_kv_heads + kv_head) * 64u;
            const float* v_ptr = v_cache + ((size_t)t * n_kv_heads + kv_head) * 64u;
            float2 kv = *(const float2*)(k_ptr + d);
            float p = qv.x * kv.x + qv.y * kv.y;
            float score = scoria_wave_sum(p) * scale;
            float m_new = fmaxf(m_i, score);
            float alpha = (l_i == 0.0f) ? 0.0f : expf(m_i - m_new);
            float beta = expf(score - m_new);
            l_i = l_i * alpha + beta;
            float2 vv = *(const float2*)(v_ptr + d);
            av.x = av.x * alpha + beta * vv.x;
            av.y = av.y * alpha + beta * vv.y;
            m_i = m_new;
        }
        if (lane == 0) {
            s_m[wave_id] = m_i;
            s_l[wave_id] = l_i;
        }
        *(float2*)(&s_o[wave_id][d]) = av;
        __syncthreads();

        if (wave_id == 0) {
            float m_g = -1e38f;
            for (unsigned w = 0; w < n_waves; ++w) {
                if (s_l[w] > 0.0f) {
                    m_g = fmaxf(m_g, s_m[w]);
                }
            }
            float l_g = 0.0f;
            float2 out_v = {0.0f, 0.0f};
            for (unsigned w = 0; w < n_waves; ++w) {
                if (s_l[w] > 0.0f) {
                    float bw = expf(s_m[w] - m_g);
                    l_g += s_l[w] * bw;
                    float2 ov = *(const float2*)(&s_o[w][d]);
                    out_v.x += ov.x * bw;
                    out_v.y += ov.y * bw;
                }
            }
            float inv_l = 1.0f / l_g;
            out_v.x *= inv_l;
            out_v.y *= inv_l;
            *(float2*)(out_base + d) = out_v;
        }
        return;
    }
#endif

    float q_r[8];
    float a_r[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
        q_r[e] = (d < head_dim) ? q_base[d] : 0.0f;
        a_r[e] = 0.0f;
    }
    for (unsigned t = wave_id; t < kv_len; t += n_waves) {
        const float* k_ptr = k_cache + ((size_t)t * n_kv_heads + kv_head) * head_dim;
        const float* v_ptr = v_cache + ((size_t)t * n_kv_heads + kv_head) * head_dim;
        float p = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
            if (d < head_dim) p += q_r[e] * k_ptr[d];
        }
        float score = scoria_wave_sum(p) * scale;
        float m_new = fmaxf(m_i, score);
        float alpha = (l_i == 0.0f) ? 0.0f : expf(m_i - m_new);
        float beta = expf(score - m_new);
        l_i = l_i * alpha + beta;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
            if (d < head_dim) a_r[e] = a_r[e] * alpha + beta * v_ptr[d];
        }
        m_i = m_new;
    }
    if (lane == 0) {
        s_m[wave_id] = m_i;
        s_l[wave_id] = l_i;
    }
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
        if (d < head_dim) s_o[wave_id][d] = a_r[e];
    }
    __syncthreads();

    if (wave_id == 0) {
        float m_g = -1e38f;
        for (unsigned w = 0; w < n_waves; ++w) {
            if (s_l[w] > 0.0f) {
                m_g = fmaxf(m_g, s_m[w]);
            }
        }
        float l_g = 0.0f;
        float acc[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
        for (unsigned w = 0; w < n_waves; ++w) {
            if (s_l[w] > 0.0f) {
                float bw = expf(s_m[w] - m_g);
                l_g += s_l[w] * bw;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
                    if (d < head_dim) acc[e] += s_o[w][d] * bw;
                }
            }
        }
        float inv_l = 1.0f / l_g;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
            if (d < head_dim) out_base[d] = acc[e] * inv_l;
        }
    }
}

extern "C" __global__ void qk_norm_rope_kv_f32(float* __restrict__ q,
                                               const float* __restrict__ k_in,
                                               const float* __restrict__ v_in,
                                               const float* __restrict__ q_norm_w,
                                               const float* __restrict__ k_norm_w,
                                               float* __restrict__ k_cache,
                                               float* __restrict__ v_cache,
                                               const unsigned* __restrict__ pos_ptr,
                                               unsigned m,
                                               unsigned n_heads,
                                               unsigned n_kv_heads,
                                               unsigned head_dim,
                                               float freq_base,
                                               float eps) {
    qk_norm_rope_kv_impl<float>(q, k_in, v_in, q_norm_w, k_norm_w, k_cache, v_cache, pos_ptr, m,
                                n_heads, n_kv_heads, head_dim, freq_base, eps);
}

extern "C" __global__ void qk_norm_rope_kv_f16(float* __restrict__ q,
                                               const float* __restrict__ k_in,
                                               const float* __restrict__ v_in,
                                               const float* __restrict__ q_norm_w,
                                               const float* __restrict__ k_norm_w,
                                               _Float16* __restrict__ k_cache,
                                               _Float16* __restrict__ v_cache,
                                               const unsigned* __restrict__ pos_ptr,
                                               unsigned m,
                                               unsigned n_heads,
                                               unsigned n_kv_heads,
                                               unsigned head_dim,
                                               float freq_base,
                                               float eps) {
    qk_norm_rope_kv_impl<_Float16>(q, k_in, v_in, q_norm_w, k_norm_w, k_cache, v_cache, pos_ptr, m,
                                   n_heads, n_kv_heads, head_dim, freq_base, eps);
}



// inkling2_loop projection post-processing. Each kernel mirrors the corresponding helper in
// `inkling2_attn.rs`, which is the CPU oracle: a 4-tap causal depthwise convolution on K
// and V, a learned per-head-per-dim scale plus L2 clip on Q, an L2 clip on K, an
// unweighted RMSNorm on V, and a sigmoid gate on the attention output.
extern "C" __global__ void inkling2_conv4_f32(float* __restrict__ x,
                                              const float* __restrict__ weight,
                                              float* __restrict__ cache,
                                              unsigned channels) {
    unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    // PyTorch Conv1d cross-correlation: tap 0 multiplies the *oldest* input and tap 3 the
    // current one. Layout is weight[channel * 4 + tap]. The pre-conv value is added back
    // as a residual and the sum is divided by sqrt(kernel + 1) = sqrt(5).
    float current = x[c];
    float lag1 = cache[c];
    float lag2 = cache[channels + c];
    float lag3 = cache[2u * channels + c];
    const float* w = weight + c * 4u;
    float acc = w[0] * lag3 + w[1] * lag2 + w[2] * lag1 + w[3] * current;
    cache[2u * channels + c] = lag2;
    cache[channels + c] = lag1;
    cache[c] = current;
    x[c] = (acc + current) * 0.447213595499958f;  // 1/sqrt(5)
}

// Per-head `l2_clip(x * scale)` with an optional per-element scale (pass `scale == nullptr`
// for K). One wave per head.
extern "C" __global__ void inkling2_head_clip_f32(float* __restrict__ x,
                                                  const float* __restrict__ scale,
                                                  unsigned n_heads,
                                                  unsigned head_dim,
                                                  float threshold) {
    unsigned h = blockIdx.x;
    if (h >= n_heads) return;
    unsigned lane = threadIdx.x;
    float* vec = x + (size_t)h * head_dim;
    float local[4];
    float ss = 0.0f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
        float v = (d < head_dim) ? vec[d] : 0.0f;
        if (d < head_dim && scale != nullptr) v *= scale[(size_t)h * head_dim + d];
        local[e] = v;
        ss += v * v;
    }
    float norm = sqrtf(scoria_wave_sum(ss));
    float factor = fminf(threshold / (norm + 1e-6f), 1.0f);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
        if (d < head_dim) vec[d] = local[e] * factor;
    }
}

// Per-head unweighted RMSNorm (V). One wave per head, eps keeps a zero head at zero.
extern "C" __global__ void inkling2_head_rmsnorm_f32(float* __restrict__ x,
                                                     unsigned n_heads,
                                                     unsigned head_dim,
                                                     float eps) {
    unsigned h = blockIdx.x;
    if (h >= n_heads) return;
    unsigned lane = threadIdx.x;
    float* vec = x + (size_t)h * head_dim;
    float ss = 0.0f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
        if (d < head_dim) {
            float v = vec[d];
            ss += v * v;
        }
    }
    float mean_sq = scoria_wave_sum(ss) / (float)head_dim;
    float inv = 1.0f / sqrtf(mean_sq + eps);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
        if (d < head_dim) vec[d] *= inv;
    }
}

extern "C" __global__ void inkling2_sigmoid_f32(float* __restrict__ x, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = 1.0f / (1.0f + expf(-x[i]));
}

// inkling2_loop decode attention: one wave per query head, online softmax over the KV
// cache with a hidden-states-conditioned relative bias, a causal sliding-window mask, and
// a per-head sink logit that absorbs probability mass before normalisation.
//
// This mirrors `inkling2_attn::decode_head` exactly; that CPU reference is the oracle.
// There is no RoPE here: position enters only through `bias`, which the host computes as
// `gather((W_rel h) proj, q_pos - k_pos)` and zeroes outside the relative extent.
//
// `window == 0` hides every position (only the sink has mass), `window == UINT_MAX` is
// unbounded, and anything between keeps positions with `pos - t < window`.
extern "C" __global__ void inkling2_attn_decode_f32(const float* __restrict__ q,
                                                    const float* __restrict__ k_cache,
                                                    const float* __restrict__ v_cache,
                                                    const float* __restrict__ bias,
                                                    const float* __restrict__ sink,
                                                    float* __restrict__ out,
                                                    unsigned pos,
                                                    unsigned n_heads,
                                                    unsigned n_kv_heads,
                                                    unsigned head_dim,
                                                    unsigned window) {
    unsigned q_head = blockIdx.x;
    if (q_head >= n_heads) return;
    unsigned lane = threadIdx.x;
    if (lane >= SCORIA_WAVE_SIZE) return;

    unsigned kv_group = n_heads / n_kv_heads;
    unsigned kv_head = q_head / kv_group;
    float scale = rsqrtf((float)head_dim);

    float q_r[4];
    float acc[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
        q_r[e] = (d < head_dim) ? q[(size_t)q_head * head_dim + d] : 0.0f;
        acc[e] = 0.0f;
    }

    // Treat the sink as a virtual score: starting the online pass at m = sink with l = 1
    // is exactly `softmax([scores, sink])` with the sink column dropped, and leaves the
    // output at zero when every position is masked.
    float sink_v = sink[q_head];
    float m_i = sink_v;
    float l_i = 1.0f;
    const float* bias_row = bias + (size_t)q_head * (pos + 1u);

    for (unsigned t = 0; t <= pos; ++t) {
        if ((pos - t) >= window) continue;
        const float* k_ptr = k_cache + ((size_t)t * n_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
            if (d < head_dim) dot += q_r[e] * k_ptr[d];
        }
        float score = scoria_wave_sum(dot) * scale + bias_row[t];
        float m_new = fmaxf(m_i, score);
        float alpha = expf(m_i - m_new);
        float beta = expf(score - m_new);
        l_i = l_i * alpha + beta;
        const float* v_ptr = v_cache + ((size_t)t * n_kv_heads + kv_head) * head_dim;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
            if (d < head_dim) acc[e] = acc[e] * alpha + beta * v_ptr[d];
        }
        m_i = m_new;
    }

    float inv_l = (l_i > 0.0f) ? 1.0f / l_i : 0.0f;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
        if (d < head_dim) out[(size_t)q_head * head_dim + d] = acc[e] * inv_l;
    }
}

// FP16 KV cache: the decode loop reads K and V once per position, so halving their
// size halves long-context bandwidth and VRAM. Activations stay FP32; only the cached
// keys and values are rounded, which is what llama.cpp does by default (`-ctk f16`).
__device__ __forceinline__ float4 scoria_load_f16x4(const _Float16* p) {
    float4 r;
    r.x = (float)p[0];
    r.y = (float)p[1];
    r.z = (float)p[2];
    r.w = (float)p[3];
    return r;
}

__device__ __forceinline__ float2 scoria_load_f16x2(const _Float16* p) {
    float2 r;
    r.x = (float)p[0];
    r.y = (float)p[1];
    return r;
}

extern "C" __global__ void kv_cache_append_f16(const float* __restrict__ k_in,
                                               const float* __restrict__ v_in,
                                               _Float16* __restrict__ k_cache,
                                               _Float16* __restrict__ v_cache,
                                               const unsigned* __restrict__ pos_ptr,
                                               unsigned m,
                                               unsigned kv_dim) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = m * kv_dim;
    if (idx >= total) return;
    size_t dst_idx = (size_t)(*pos_ptr) * kv_dim + idx;
    k_cache[dst_idx] = (_Float16)k_in[idx];
    v_cache[dst_idx] = (_Float16)v_in[idx];
}

// Supports fast Wave32 paths for head_dim 128 and 64 on RDNA4, with a general
// fallback for arbitrary head_dim <= 256 (and Wave64 CDNA).
extern "C" __global__ void gqa_attn_f16(const float* __restrict__ q,
                                        const _Float16* __restrict__ k_cache,
                                        const _Float16* __restrict__ v_cache,
                                        float* __restrict__ out,
                                        const unsigned* __restrict__ pos_ptr,
                                        unsigned m,
                                        unsigned n_heads,
                                        unsigned n_kv_heads,
                                        unsigned head_dim) {
    unsigned q_head = blockIdx.x;
    unsigned tok = blockIdx.y;
    if (q_head >= n_heads || tok >= m || head_dim == 0u || head_dim > 256u) return;
    unsigned start_pos = *pos_ptr;

    unsigned tid = threadIdx.x;
    unsigned lane = tid & (SCORIA_WAVE_SIZE - 1u);
    unsigned wave_id = tid / SCORIA_WAVE_SIZE;
    unsigned n_waves = blockDim.x / SCORIA_WAVE_SIZE;

    unsigned kv_group = n_heads / n_kv_heads;
    unsigned kv_head = q_head / kv_group;
    unsigned kv_len = start_pos + tok + 1u;
    float scale = rsqrtf((float)head_dim);

    const float* q_base = q + ((size_t)tok * n_heads + q_head) * head_dim;
    float* out_base = out + ((size_t)tok * n_heads + q_head) * head_dim;

    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_o[4][256];

    float m_i = -1e38f;
    float l_i = 0.0f;

#ifdef SCORIA_RDNA4
    if (head_dim == 128u) {
        unsigned d = lane * 4u;
        float4 qv = *(const float4*)(q_base + d);
        float4 av = {0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned t = wave_id; t < kv_len; t += n_waves) {
            const _Float16* k_ptr = k_cache + ((size_t)t * n_kv_heads + kv_head) * 128u;
            const _Float16* v_ptr = v_cache + ((size_t)t * n_kv_heads + kv_head) * 128u;
            float4 kv = scoria_load_f16x4(k_ptr + d);
            float p = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            float score = scoria_wave_sum(p) * scale;
            float m_new = fmaxf(m_i, score);
            float alpha = (l_i == 0.0f) ? 0.0f : expf(m_i - m_new);
            float beta = expf(score - m_new);
            l_i = l_i * alpha + beta;
            float4 vv = scoria_load_f16x4(v_ptr + d);
            av.x = av.x * alpha + beta * vv.x;
            av.y = av.y * alpha + beta * vv.y;
            av.z = av.z * alpha + beta * vv.z;
            av.w = av.w * alpha + beta * vv.w;
            m_i = m_new;
        }
        if (lane == 0) {
            s_m[wave_id] = m_i;
            s_l[wave_id] = l_i;
        }
        *(float4*)(&s_o[wave_id][d]) = av;
        __syncthreads();
        if (wave_id == 0) {
            float m_g = -1e38f;
            for (unsigned w = 0; w < n_waves; ++w) {
                if (s_l[w] > 0.0f) m_g = fmaxf(m_g, s_m[w]);
            }
            float l_g = 0.0f;
            float4 out_v = {0.0f, 0.0f, 0.0f, 0.0f};
            for (unsigned w = 0; w < n_waves; ++w) {
                if (s_l[w] > 0.0f) {
                    float bw = expf(s_m[w] - m_g);
                    l_g += s_l[w] * bw;
                    float4 ov = *(const float4*)(&s_o[w][d]);
                    out_v.x += ov.x * bw;
                    out_v.y += ov.y * bw;
                    out_v.z += ov.z * bw;
                    out_v.w += ov.w * bw;
                }
            }
            float inv_l = 1.0f / l_g;
            out_v.x *= inv_l;
            out_v.y *= inv_l;
            out_v.z *= inv_l;
            out_v.w *= inv_l;
            *(float4*)(out_base + d) = out_v;
        }
        return;
    }

    if (head_dim == 64u) {
        unsigned d = lane * 2u;
        float2 qv = *(const float2*)(q_base + d);
        float2 av = {0.0f, 0.0f};
        for (unsigned t = wave_id; t < kv_len; t += n_waves) {
            const _Float16* k_ptr = k_cache + ((size_t)t * n_kv_heads + kv_head) * 64u;
            const _Float16* v_ptr = v_cache + ((size_t)t * n_kv_heads + kv_head) * 64u;
            float2 kv = scoria_load_f16x2(k_ptr + d);
            float p = qv.x * kv.x + qv.y * kv.y;
            float score = scoria_wave_sum(p) * scale;
            float m_new = fmaxf(m_i, score);
            float alpha = (l_i == 0.0f) ? 0.0f : expf(m_i - m_new);
            float beta = expf(score - m_new);
            l_i = l_i * alpha + beta;
            float2 vv = scoria_load_f16x2(v_ptr + d);
            av.x = av.x * alpha + beta * vv.x;
            av.y = av.y * alpha + beta * vv.y;
            m_i = m_new;
        }
        if (lane == 0) {
            s_m[wave_id] = m_i;
            s_l[wave_id] = l_i;
        }
        *(float2*)(&s_o[wave_id][d]) = av;
        __syncthreads();
        if (wave_id == 0) {
            float m_g = -1e38f;
            for (unsigned w = 0; w < n_waves; ++w) {
                if (s_l[w] > 0.0f) m_g = fmaxf(m_g, s_m[w]);
            }
            float l_g = 0.0f;
            float2 out_v = {0.0f, 0.0f};
            for (unsigned w = 0; w < n_waves; ++w) {
                if (s_l[w] > 0.0f) {
                    float bw = expf(s_m[w] - m_g);
                    l_g += s_l[w] * bw;
                    float2 ov = *(const float2*)(&s_o[w][d]);
                    out_v.x += ov.x * bw;
                    out_v.y += ov.y * bw;
                }
            }
            float inv_l = 1.0f / l_g;
            out_v.x *= inv_l;
            out_v.y *= inv_l;
            *(float2*)(out_base + d) = out_v;
        }
        return;
    }
#endif

    float q_r[8];
    float a_r[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
        q_r[e] = (d < head_dim) ? q_base[d] : 0.0f;
        a_r[e] = 0.0f;
    }
    for (unsigned t = wave_id; t < kv_len; t += n_waves) {
        const _Float16* k_ptr = k_cache + ((size_t)t * n_kv_heads + kv_head) * head_dim;
        const _Float16* v_ptr = v_cache + ((size_t)t * n_kv_heads + kv_head) * head_dim;
        float p = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
            if (d < head_dim) p += q_r[e] * (float)k_ptr[d];
        }
        float score = scoria_wave_sum(p) * scale;
        float m_new = fmaxf(m_i, score);
        float alpha = (l_i == 0.0f) ? 0.0f : expf(m_i - m_new);
        float beta = expf(score - m_new);
        l_i = l_i * alpha + beta;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
            if (d < head_dim) a_r[e] = a_r[e] * alpha + beta * (float)v_ptr[d];
        }
        m_i = m_new;
    }
    if (lane == 0) {
        s_m[wave_id] = m_i;
        s_l[wave_id] = l_i;
    }
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
        unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
        if (d < head_dim) s_o[wave_id][d] = a_r[e];
    }
    __syncthreads();

    if (wave_id == 0) {
        float m_g = -1e38f;
        for (unsigned w = 0; w < n_waves; ++w) {
            if (s_l[w] > 0.0f) m_g = fmaxf(m_g, s_m[w]);
        }
        float l_g = 0.0f;
        float acc[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
        for (unsigned w = 0; w < n_waves; ++w) {
            if (s_l[w] > 0.0f) {
                float bw = expf(s_m[w] - m_g);
                l_g += s_l[w] * bw;
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
                    if (d < head_dim) acc[e] += s_o[w][d] * bw;
                }
            }
        }
        float inv_l = 1.0f / l_g;
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            unsigned d = lane + (unsigned)e * SCORIA_WAVE_SIZE;
            if (d < head_dim) out_base[d] = acc[e] * inv_l;
        }
    }
}

// Decode-loop epilogue: feed the sampled token back into the input slot and advance
// the position counter, so a captured HIP graph can be replayed unchanged.
extern "C" __global__ void decode_advance_u32(const unsigned* __restrict__ argmax,
                                              unsigned* __restrict__ token_ids,
                                              unsigned* __restrict__ pos) {
    token_ids[0] = argmax[0];
    pos[0] = pos[0] + 1u;
}

extern "C" __global__ void silu_mul_f32(const float* __restrict__ gate,
                                        const float* __restrict__ up,
                                        float* __restrict__ out,
                                        unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((n & 3u) == 0u) {
        unsigned n4 = n >> 2;
        if (idx >= n4) return;
        float4 g = ((const float4*)gate)[idx];
        float4 u = ((const float4*)up)[idx];
        float4 o = {
            (g.x / (1.0f + expf(-g.x))) * u.x,
            (g.y / (1.0f + expf(-g.y))) * u.y,
            (g.z / (1.0f + expf(-g.z))) * u.z,
            (g.w / (1.0f + expf(-g.w))) * u.w,
        };
        ((float4*)out)[idx] = o;
        return;
    }
    if (idx >= n) return;
    float g = gate[idx];
    float silu = g / (1.0f + expf(-g));
    out[idx] = silu * up[idx];
}

extern "C" __global__ void add_inplace_f32(float* __restrict__ dst,
                                           const float* __restrict__ src,
                                           unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] += src[idx];
}

extern "C" __global__ void add_rmsnorm_f32(float* __restrict__ x,
                                           const float* __restrict__ residual,
                                           const float* __restrict__ w,
                                           float* __restrict__ normed_out,
                                           unsigned n_tokens,
                                           unsigned n_embd,
                                           float eps) {
    unsigned tok = blockIdx.x;
    if (tok >= n_tokens) return;
    unsigned tid = threadIdx.x;
    unsigned lane = tid & (SCORIA_WAVE_SIZE - 1u);
    unsigned wave_id = tid / SCORIA_WAVE_SIZE;
    unsigned n_waves = blockDim.x / SCORIA_WAVE_SIZE;

    float* x_row = x + (size_t)tok * n_embd;
    const float* r_row = residual + (size_t)tok * n_embd;
    float* out_row = normed_out + (size_t)tok * n_embd;

    float ss = 0.0f;
    if ((n_embd & 3u) == 0u) {
        unsigned n4 = n_embd >> 2;
        float4* x4 = (float4*)x_row;
        const float4* r4 = (const float4*)r_row;
        for (unsigned i = tid; i < n4; i += blockDim.x) {
            float4 xv = x4[i];
            float4 rv = r4[i];
            float4 v = {xv.x + rv.x, xv.y + rv.y, xv.z + rv.z, xv.w + rv.w};
            x4[i] = v;
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (unsigned i = tid; i < n_embd; i += blockDim.x) {
            float v = x_row[i] + r_row[i];
            x_row[i] = v;
            ss += v * v;
        }
    }
    float wsum = scoria_wave_sum(ss);
    __shared__ float s_sums[32];
    __shared__ float s_inv;
    if (lane == 0) {
        s_sums[wave_id] = wsum;
    }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (unsigned k = 0; k < n_waves; ++k) {
            total += s_sums[k];
        }
        s_inv = rsqrtf(total / (float)n_embd + eps);
    }
    __syncthreads();
    float inv = s_inv;
    if ((n_embd & 3u) == 0u) {
        unsigned n4 = n_embd >> 2;
        const float4* x4 = (const float4*)x_row;
        const float4* w4 = (const float4*)w;
        float4* out4 = (float4*)out_row;
        for (unsigned i = tid; i < n4; i += blockDim.x) {
            float4 xv = x4[i];
            float4 wv = w4[i];
            float4 ov = {xv.x * inv * wv.x, xv.y * inv * wv.y, xv.z * inv * wv.z, xv.w * inv * wv.w};
            out4[i] = ov;
        }
    } else {
        for (unsigned i = tid; i < n_embd; i += blockDim.x) {
            out_row[i] = x_row[i] * inv * w[i];
        }
    }
}

extern "C" __global__ void argmax_stage1_f32(const float* __restrict__ vals,
                                             float* __restrict__ out_vals,
                                             unsigned* __restrict__ out_idxs,
                                             unsigned n) {
    if (n == 0u) return;
    unsigned tid = threadIdx.x;
    unsigned lane = tid & (SCORIA_WAVE_SIZE - 1u);
    unsigned wave_id = tid / SCORIA_WAVE_SIZE;
    unsigned n_waves = blockDim.x / SCORIA_WAVE_SIZE;
    unsigned stride = gridDim.x * blockDim.x;

    float best_v = -__builtin_huge_valf();
    unsigned best_i = 0xFFFFFFFFu;
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        float v = vals[i];
        if (v > best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
    }

    for (int off = SCORIA_WAVE_SIZE >> 1; off > 0; off >>= 1) {
        float ov = __shfl_xor(best_v, off, SCORIA_WAVE_SIZE);
        unsigned oi = __shfl_xor(best_i, off, SCORIA_WAVE_SIZE);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    __shared__ float s_v[32];
    __shared__ unsigned s_i[32];
    if (lane == 0) {
        s_v[wave_id] = best_v;
        s_i[wave_id] = best_i;
    }
    __syncthreads();

    if (tid == 0) {
        float fv = s_v[0];
        unsigned fi = s_i[0];
        for (unsigned w = 1; w < n_waves; ++w) {
            float wv = s_v[w];
            unsigned wi = s_i[w];
            if (wv > fv || (wv == fv && wi < fi)) {
                fv = wv;
                fi = wi;
            }
        }
        out_vals[blockIdx.x] = fv;
        out_idxs[blockIdx.x] = fi;
    }
}

extern "C" __global__ void argmax_stage2_f32(const float* __restrict__ in_vals,
                                             const unsigned* __restrict__ in_idxs,
                                             unsigned* __restrict__ out_idx,
                                             unsigned count) {
    if (count == 0u) return;
    unsigned tid = threadIdx.x;
    unsigned lane = tid & (SCORIA_WAVE_SIZE - 1u);
    unsigned wave_id = tid / SCORIA_WAVE_SIZE;
    unsigned n_waves = blockDim.x / SCORIA_WAVE_SIZE;

    float best_v = -__builtin_huge_valf();
    unsigned best_i = 0xFFFFFFFFu;
    for (unsigned i = tid; i < count; i += blockDim.x) {
        float v = in_vals[i];
        unsigned idx = in_idxs[i];
        if (v > best_v || (v == best_v && idx < best_i)) {
            best_v = v;
            best_i = idx;
        }
    }

    for (int off = SCORIA_WAVE_SIZE >> 1; off > 0; off >>= 1) {
        float ov = __shfl_xor(best_v, off, SCORIA_WAVE_SIZE);
        unsigned oi = __shfl_xor(best_i, off, SCORIA_WAVE_SIZE);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    __shared__ float s_v[32];
    __shared__ unsigned s_i[32];
    if (lane == 0) {
        s_v[wave_id] = best_v;
        s_i[wave_id] = best_i;
    }
    __syncthreads();

    if (tid == 0) {
        float fv = s_v[0];
        unsigned fi = s_i[0];
        for (unsigned w = 1; w < n_waves; ++w) {
            float wv = s_v[w];
            unsigned wi = s_i[w];
            if (wv > fv || (wv == fv && wi < fi)) {
                fv = wv;
                fi = wi;
            }
        }
        *out_idx = fi;
    }
}

extern "C" __global__ void gemv_f16_f32(const unsigned char* __restrict__ weight,
                                        const float* __restrict__ x,
                                        const float* __restrict__ bias,
                                        float* __restrict__ y,
                                        unsigned n,
                                        unsigned k) {
    unsigned row = blockIdx.x;
    if (row >= n) return;
    unsigned tid = threadIdx.x;
    unsigned lane = tid & (SCORIA_WAVE_SIZE - 1u);
    unsigned wave_id = tid / SCORIA_WAVE_SIZE;
    unsigned n_waves = blockDim.x / SCORIA_WAVE_SIZE;

    const unsigned short* w_row = (const unsigned short*)(weight + (size_t)row * k * 2u);
    float acc = 0.0f;
    for (unsigned i = tid; i < k; i += blockDim.x) {
        acc += scoria_f16_bits_to_f32(w_row[i]) * x[i];
    }

    float wsum = scoria_wave_sum(acc);
    __shared__ float s_red[4];
    if (lane == 0) {
        s_red[wave_id] = wsum;
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (unsigned w = 0; w < n_waves; ++w) {
            sum += s_red[w];
        }
        y[row] = sum + (bias ? bias[row] : 0.0f);
    }
}

extern "C" __global__ void gemv_q5_0_f32(const unsigned char* __restrict__ weight,
                                         const float* __restrict__ x,
                                         const float* __restrict__ bias,
                                         float* __restrict__ y,
                                         unsigned n,
                                         unsigned k) {
    unsigned wave_id = threadIdx.x / SCORIA_WAVE_SIZE;
    unsigned lane = threadIdx.x & (SCORIA_WAVE_SIZE - 1u);
    unsigned row = blockIdx.x * (blockDim.x / SCORIA_WAVE_SIZE) + wave_id;
    if (row >= n) return;

    unsigned sub_wave = lane >> 5;
    unsigned sub_lane = lane & 31u;
    unsigned sub_waves = SCORIA_WAVE_SIZE >> 5;
    unsigned j = sub_lane & 15u;
    unsigned hi_half = sub_lane >> 4;
    unsigned nb = k / 32u;

    const unsigned char* w_row = weight + (size_t)row * nb * 22u;
    float acc = 0.0f;
    for (unsigned b = sub_wave; b < nb; b += sub_waves) {
        const unsigned char* blk = w_row + (size_t)b * 22u;
        float d = scoria_f16_bits_to_f32(*(const unsigned short*)blk);
        unsigned qh = (unsigned)(*(const unsigned short*)(blk + 2))
                    | ((unsigned)(*(const unsigned short*)(blk + 4)) << 16);
        unsigned char qb = blk[6u + j];
        unsigned nib = hi_half ? (qb >> 4) : (qb & 0x0Fu);
        unsigned xh = hi_half ? ((qh >> (j + 12u)) & 0x10u) : (((qh >> j) << 4) & 0x10u);
        int q = (int)(nib | xh) - 16;
        acc += d * (float)q * x[(size_t)b * 32u + sub_lane];
    }

    float sum = scoria_wave_sum(acc);
    if (lane == 0) {
        y[row] = sum + (bias ? bias[row] : 0.0f);
    }
}

extern "C" __global__ void gemv_q8_0_f32(const unsigned char* __restrict__ weight,
                                         const float* __restrict__ x,
                                         const float* __restrict__ bias,
                                         float* __restrict__ y,
                                         unsigned n,
                                         unsigned k) {
    unsigned wave_id = threadIdx.x / SCORIA_WAVE_SIZE;
    unsigned lane = threadIdx.x & (SCORIA_WAVE_SIZE - 1u);
    unsigned row = blockIdx.x * (blockDim.x / SCORIA_WAVE_SIZE) + wave_id;
    if (row >= n) return;

    unsigned sub_wave = lane >> 5;
    unsigned sub_lane = lane & 31u;
    unsigned sub_waves = SCORIA_WAVE_SIZE >> 5;
    unsigned nb = k / 32u;

    const unsigned char* w_row = weight + (size_t)row * nb * 34u;
    float acc = 0.0f;
    for (unsigned b = sub_wave; b < nb; b += sub_waves) {
        const unsigned char* blk = w_row + (size_t)b * 34u;
        float d = scoria_f16_bits_to_f32(*(const unsigned short*)blk);
        signed char q = (signed char)blk[2u + sub_lane];
        acc += d * (float)q * x[(size_t)b * 32u + sub_lane];
    }

    float sum = scoria_wave_sum(acc);
    if (lane == 0) {
        y[row] = sum + (bias ? bias[row] : 0.0f);
    }
}

__device__ __forceinline__ void scoria_q4k_unpack_scales(uint4 hdr,
                                                         unsigned chunk,
                                                         float* d1,
                                                         float* m1,
                                                         float* d2,
                                                         float* m2) {
    float d = scoria_f16_bits_to_f32((unsigned short)(hdr.x & 0xFFFFu));
    float dmin = scoria_f16_bits_to_f32((unsigned short)(hdr.x >> 16));
    unsigned s0 = hdr.y;
    unsigned s1 = hdr.z;
    unsigned s2 = hdr.w;
    unsigned sc1_u, min1_u, sc2_u, min2_u;
    if (chunk < 2u) {
        unsigned shift = chunk * 16u;
        sc1_u = (s0 >> shift) & 63u;
        min1_u = (s1 >> shift) & 63u;
        sc2_u = (s0 >> (shift + 8u)) & 63u;
        min2_u = (s1 >> (shift + 8u)) & 63u;
    } else {
        unsigned shift_s2 = (chunk - 2u) * 16u;
        unsigned shift_hi = shift_s2 + 6u;
        sc1_u = ((s2 >> shift_s2) & 0x0Fu) | (((s0 >> shift_hi) & 3u) << 4);
        min1_u = ((s2 >> (shift_s2 + 4u)) & 0x0Fu) | (((s1 >> shift_hi) & 3u) << 4);
        sc2_u = ((s2 >> (shift_s2 + 8u)) & 0x0Fu) | (((s0 >> (shift_hi + 8u)) & 3u) << 4);
        min2_u = ((s2 >> (shift_s2 + 12u)) & 0x0Fu) | (((s1 >> (shift_hi + 8u)) & 3u) << 4);
    }
    *d1 = d * (float)sc1_u;
    *m1 = dmin * (float)min1_u;
    *d2 = d * (float)sc2_u;
    *m2 = dmin * (float)min2_u;
}

__device__ __forceinline__ void scoria_q4k_dot_word(unsigned qw,
                                                    float4 x_lo,
                                                    float4 x_hi,
                                                    float* s_lo,
                                                    float* s_hi) {
    unsigned b0 = qw & 0xFFu;
    unsigned b1 = (qw >> 8) & 0xFFu;
    unsigned b2 = (qw >> 16) & 0xFFu;
    unsigned b3 = (qw >> 24) & 0xFFu;
    *s_lo += (float)(b0 & 0x0Fu) * x_lo.x
           + (float)(b1 & 0x0Fu) * x_lo.y
           + (float)(b2 & 0x0Fu) * x_lo.z
           + (float)(b3 & 0x0Fu) * x_lo.w;
    *s_hi += (float)(b0 >> 4) * x_hi.x
           + (float)(b1 >> 4) * x_hi.y
           + (float)(b2 >> 4) * x_hi.z
           + (float)(b3 >> 4) * x_hi.w;
}

// Q4_K decode GEMV.
//
// The workgroup is 128 threads = 4 waves = 16 groups of 8 lanes. Each group owns
// one 256-weight superblock: its 8 lanes split the 4 quant chunks x 2 nibble halves
// (32B of `qs` per lane pair) and the wave reduces the partial dots.
//
// A hard 16-group layout starves the block whenever `nb = k/256 < 16`
// (Qwen3-4B: nb=10 -> only 62.5% of groups had work, measured 342 GB/s vs the
// 7B model's 524 GB/s). `ROWS` rows per block are therefore chosen on the host so
// that `(ROWS/2) * nb >= 16`, i.e. every group gets at least one superblock per row
// pair, and rows per block grow as `k` shrinks.
template <unsigned ROWS>
__device__ __forceinline__ void gemv_q4_k_rows_impl(const unsigned char* __restrict__ weight,
                                                    const float* __restrict__ x,
                                                    const float* __restrict__ bias,
                                                    float* __restrict__ y,
                                                    unsigned n,
                                                    unsigned k) {
    unsigned row0 = blockIdx.x * ROWS;
    if (row0 >= n) return;

    unsigned tid = threadIdx.x;
    unsigned lane = tid & (SCORIA_WAVE_SIZE - 1u);
    unsigned wave_id = tid / SCORIA_WAVE_SIZE;
    unsigned n_waves = blockDim.x / SCORIA_WAVE_SIZE;
    unsigned grp = tid >> 3;
    unsigned lane_in_blk = tid & 7u;
    unsigned chunk = lane_in_blk >> 1;
    unsigned half_c = lane_in_blk & 1u;
    unsigned nb = k / 256u;
    size_t row_bytes = (size_t)nb * 144u;

    float acc[ROWS];
    #pragma unroll
    for (unsigned r = 0; r < ROWS; ++r) acc[r] = 0.0f;

    #pragma unroll
    for (unsigned rp = 0; rp < ROWS / 2u; ++rp) {
        unsigned r0 = row0 + rp * 2u;
        if (r0 < n) {
            bool has_r1 = (r0 + 1u) < n;
            const unsigned char* w0 = weight + (size_t)r0 * row_bytes;
            const unsigned char* w1 = w0 + row_bytes;

            for (unsigned b = grp; b < nb; b += 16u) {
                const unsigned char* blk0 = w0 + (size_t)b * 144u;
                uint4 hdr0 = *(const uint4*)blk0;
                uint4 q0 = *(const uint4*)(blk0 + 16u + (size_t)lane_in_blk * 16u);
                uint4 hdr1 = {0u, 0u, 0u, 0u};
                uint4 q1 = {0u, 0u, 0u, 0u};
                if (has_r1) {
                    const unsigned char* blk1 = w1 + (size_t)b * 144u;
                    hdr1 = *(const uint4*)blk1;
                    q1 = *(const uint4*)(blk1 + 16u + (size_t)lane_in_blk * 16u);
                }

                const float* xp = x + (size_t)b * 256u + chunk * 64u + half_c * 16u;
                float4 x_lo0 = *(const float4*)(xp + 0u);
                float4 x_lo1 = *(const float4*)(xp + 4u);
                float4 x_lo2 = *(const float4*)(xp + 8u);
                float4 x_lo3 = *(const float4*)(xp + 12u);
                float4 x_hi0 = *(const float4*)(xp + 32u);
                float4 x_hi1 = *(const float4*)(xp + 36u);
                float4 x_hi2 = *(const float4*)(xp + 40u);
                float4 x_hi3 = *(const float4*)(xp + 44u);

                float xs_lo = (x_lo0.x + x_lo0.y + x_lo0.z + x_lo0.w)
                            + (x_lo1.x + x_lo1.y + x_lo1.z + x_lo1.w)
                            + (x_lo2.x + x_lo2.y + x_lo2.z + x_lo2.w)
                            + (x_lo3.x + x_lo3.y + x_lo3.z + x_lo3.w);
                float xs_hi = (x_hi0.x + x_hi0.y + x_hi0.z + x_hi0.w)
                            + (x_hi1.x + x_hi1.y + x_hi1.z + x_hi1.w)
                            + (x_hi2.x + x_hi2.y + x_hi2.z + x_hi2.w)
                            + (x_hi3.x + x_hi3.y + x_hi3.z + x_hi3.w);

                float d1_0, m1_0, d2_0, m2_0;
                scoria_q4k_unpack_scales(hdr0, chunk, &d1_0, &m1_0, &d2_0, &m2_0);
                float s_lo0 = 0.0f, s_hi0 = 0.0f;
                scoria_q4k_dot_word(q0.x, x_lo0, x_hi0, &s_lo0, &s_hi0);
                scoria_q4k_dot_word(q0.y, x_lo1, x_hi1, &s_lo0, &s_hi0);
                scoria_q4k_dot_word(q0.z, x_lo2, x_hi2, &s_lo0, &s_hi0);
                scoria_q4k_dot_word(q0.w, x_lo3, x_hi3, &s_lo0, &s_hi0);
                acc[rp * 2u] += (d1_0 * s_lo0 - m1_0 * xs_lo) + (d2_0 * s_hi0 - m2_0 * xs_hi);

                if (has_r1) {
                    float d1_1, m1_1, d2_1, m2_1;
                    scoria_q4k_unpack_scales(hdr1, chunk, &d1_1, &m1_1, &d2_1, &m2_1);
                    float s_lo1 = 0.0f, s_hi1 = 0.0f;
                    scoria_q4k_dot_word(q1.x, x_lo0, x_hi0, &s_lo1, &s_hi1);
                    scoria_q4k_dot_word(q1.y, x_lo1, x_hi1, &s_lo1, &s_hi1);
                    scoria_q4k_dot_word(q1.z, x_lo2, x_hi2, &s_lo1, &s_hi1);
                    scoria_q4k_dot_word(q1.w, x_lo3, x_hi3, &s_lo1, &s_hi1);
                    acc[rp * 2u + 1u] += (d1_1 * s_lo1 - m1_1 * xs_lo) + (d2_1 * s_hi1 - m2_1 * xs_hi);
                }
            }
        }
    }

    __shared__ float s_red[ROWS][8];
    #pragma unroll
    for (unsigned r = 0; r < ROWS; ++r) {
        float s = scoria_wave_sum(acc[r]);
        if (lane == 0) s_red[r][wave_id] = s;
    }
    __syncthreads();
    if (tid == 0) {
        #pragma unroll
        for (unsigned r = 0; r < ROWS; ++r) {
            unsigned row = row0 + r;
            if (row < n) {
                float sum = 0.0f;
                for (unsigned w = 0; w < n_waves; ++w) sum += s_red[r][w];
                y[row] = sum + (bias ? bias[row] : 0.0f);
            }
        }
    }
}

extern "C" __global__ void gemv_q4_k_f32(const unsigned char* __restrict__ weight,
                                         const float* __restrict__ x,
                                         const float* __restrict__ bias,
                                         float* __restrict__ y,
                                         unsigned n,
                                         unsigned k) {
    gemv_q4_k_rows_impl<2u>(weight, x, bias, y, n, k);
}

extern "C" __global__ void gemv_q4_k_r4_f32(const unsigned char* __restrict__ weight,
                                            const float* __restrict__ x,
                                            const float* __restrict__ bias,
                                            float* __restrict__ y,
                                            unsigned n,
                                            unsigned k) {
    gemv_q4_k_rows_impl<4u>(weight, x, bias, y, n, k);
}

extern "C" __global__ void gemv_q4_k_r8_f32(const unsigned char* __restrict__ weight,
                                            const float* __restrict__ x,
                                            const float* __restrict__ bias,
                                            float* __restrict__ y,
                                            unsigned n,
                                            unsigned k) {
    gemv_q4_k_rows_impl<8u>(weight, x, bias, y, n, k);
}

// Fused Q4_K gate + up GEMV + SiLU*up for m=1 decode.
// One wavefront per output row computes gate & up simultaneously while reading `x` once,
// eliminating 2 kernel launches + the intermediate `up` DRAM roundtrip per layer.
extern "C" __global__ void gemv_q4_k_gate_up_silu_f32(const unsigned char* __restrict__ w_gate,
                                                      const unsigned char* __restrict__ w_up,
                                                      const float* __restrict__ x,
                                                      float* __restrict__ y_out,
                                                      unsigned n,
                                                      unsigned k) {
    unsigned row = blockIdx.x;
    if (row >= n) return;
    unsigned lane = threadIdx.x;
    if (lane >= SCORIA_WAVE_SIZE) return;

    unsigned grp = lane >> 3;
    unsigned lane_in_blk = lane & 7u;
    unsigned grp_stride = SCORIA_WAVE_SIZE >> 3;
    unsigned chunk = lane_in_blk >> 1;
    unsigned half_c = lane_in_blk & 1u;
    unsigned nb = k / 256u;
    size_t row_bytes = (size_t)nb * 144u;

    const unsigned char* wg = w_gate + (size_t)row * row_bytes;
    const unsigned char* wu = w_up + (size_t)row * row_bytes;
    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (unsigned b = grp; b < nb; b += grp_stride) {
        const unsigned char* bg = wg + (size_t)b * 144u;
        const unsigned char* bu = wu + (size_t)b * 144u;
        uint4 hdr_g = *(const uint4*)bg;
        uint4 q_g = *(const uint4*)(bg + 16u + (size_t)lane_in_blk * 16u);
        uint4 hdr_u = *(const uint4*)bu;
        uint4 q_u = *(const uint4*)(bu + 16u + (size_t)lane_in_blk * 16u);

        const float* xp = x + (size_t)b * 256u + chunk * 64u + half_c * 16u;
        float4 x_lo0 = *(const float4*)(xp + 0u);
        float4 x_lo1 = *(const float4*)(xp + 4u);
        float4 x_lo2 = *(const float4*)(xp + 8u);
        float4 x_lo3 = *(const float4*)(xp + 12u);
        float4 x_hi0 = *(const float4*)(xp + 32u);
        float4 x_hi1 = *(const float4*)(xp + 36u);
        float4 x_hi2 = *(const float4*)(xp + 40u);
        float4 x_hi3 = *(const float4*)(xp + 44u);

        float xs_lo = (x_lo0.x + x_lo0.y + x_lo0.z + x_lo0.w)
                    + (x_lo1.x + x_lo1.y + x_lo1.z + x_lo1.w)
                    + (x_lo2.x + x_lo2.y + x_lo2.z + x_lo2.w)
                    + (x_lo3.x + x_lo3.y + x_lo3.z + x_lo3.w);
        float xs_hi = (x_hi0.x + x_hi0.y + x_hi0.z + x_hi0.w)
                    + (x_hi1.x + x_hi1.y + x_hi1.z + x_hi1.w)
                    + (x_hi2.x + x_hi2.y + x_hi2.z + x_hi2.w)
                    + (x_hi3.x + x_hi3.y + x_hi3.z + x_hi3.w);

        float d1g, m1g, d2g, m2g;
        scoria_q4k_unpack_scales(hdr_g, chunk, &d1g, &m1g, &d2g, &m2g);
        float s_lo_g = 0.0f, s_hi_g = 0.0f;
        scoria_q4k_dot_word(q_g.x, x_lo0, x_hi0, &s_lo_g, &s_hi_g);
        scoria_q4k_dot_word(q_g.y, x_lo1, x_hi1, &s_lo_g, &s_hi_g);
        scoria_q4k_dot_word(q_g.z, x_lo2, x_hi2, &s_lo_g, &s_hi_g);
        scoria_q4k_dot_word(q_g.w, x_lo3, x_hi3, &s_lo_g, &s_hi_g);
        acc_g += (d1g * s_lo_g - m1g * xs_lo) + (d2g * s_hi_g - m2g * xs_hi);

        float d1u, m1u, d2u, m2u;
        scoria_q4k_unpack_scales(hdr_u, chunk, &d1u, &m1u, &d2u, &m2u);
        float s_lo_u = 0.0f, s_hi_u = 0.0f;
        scoria_q4k_dot_word(q_u.x, x_lo0, x_hi0, &s_lo_u, &s_hi_u);
        scoria_q4k_dot_word(q_u.y, x_lo1, x_hi1, &s_lo_u, &s_hi_u);
        scoria_q4k_dot_word(q_u.z, x_lo2, x_hi2, &s_lo_u, &s_hi_u);
        scoria_q4k_dot_word(q_u.w, x_lo3, x_hi3, &s_lo_u, &s_hi_u);
        acc_u += (d1u * s_lo_u - m1u * xs_lo) + (d2u * s_hi_u - m2u * xs_hi);
    }

    float sg = scoria_wave_sum(acc_g);
    float su = scoria_wave_sum(acc_u);
    if (lane == 0) {
        float silu = sg / (1.0f + expf(-sg));
        y_out[row] = silu * su;
    }
}

// 48-byte 16B-aligned Q8_1 block per 32 floats:
// [0..31] int8 qs[32], [32..35] float d, [36..39] float s (d * sum(qs)), [40..47] pad.
extern "C" __global__ void quantize_q8_1_f32(const float* __restrict__ x,
                                             unsigned char* __restrict__ out_q8,
                                             unsigned k) {
    unsigned blk_id = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (blk_id * 32u >= k) return;
    unsigned lane = threadIdx.x & 31u;

    unsigned idx = blk_id * 32u + lane;
    float v = (idx < k) ? x[idx] : 0.0f;
    float amax = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, off, 32));
    }
    float d = amax * (1.0f / 127.0f);
    float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = (int)rintf(v * inv_d);
    if (q > 127) q = 127;
    if (q < -127) q = -127;

    int q0 = q;
    int q1 = __shfl_xor(q, 1, 32);
    int q2 = __shfl_xor(q, 2, 32);
    int q3 = __shfl_xor(q, 3, 32);

    int sum_q = q;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum_q += __shfl_xor(sum_q, off, 32);
    }

    unsigned char* dst = out_q8 + (size_t)blk_id * 48u;
    if ((lane & 3u) == 0u) {
        int packed = (q0 & 0xFF)
                   | ((q1 & 0xFF) << 8)
                   | ((q2 & 0xFF) << 16)
                   | ((q3 & 0xFF) << 24);
        ((int*)dst)[lane >> 2] = packed;
    }
    if (lane == 0u) {
        float2 hdr = {d, d * (float)sum_q};
        *(float2*)(dst + 32u) = hdr;
    }
}

extern "C" __global__ void gemv_q4_k_q8_1_f32(const unsigned char* __restrict__ weight,
                                              const unsigned char* __restrict__ x_q8,
                                              const float* __restrict__ bias,
                                              float* __restrict__ y,
                                              unsigned n,
                                              unsigned k) {
    unsigned wave_id = threadIdx.x / SCORIA_WAVE_SIZE;
    unsigned lane = threadIdx.x & (SCORIA_WAVE_SIZE - 1u);
    unsigned waves_per_wg = blockDim.x / SCORIA_WAVE_SIZE;
    unsigned row0 = (blockIdx.x * waves_per_wg + wave_id) * 2u;
    if (row0 >= n) return;
    unsigned row1 = row0 + 1u;
    bool has_r1 = row1 < n;

    unsigned grp = lane >> 3;
    unsigned lane_in_blk = lane & 7u;
    unsigned grp_stride = SCORIA_WAVE_SIZE >> 3;
    unsigned chunk = lane_in_blk >> 1;
    unsigned half_c = lane_in_blk & 1u;
    unsigned nb = k / 256u;
    size_t row_bytes = (size_t)nb * 144u;

    const unsigned char* w0 = weight + (size_t)row0 * row_bytes;
    const unsigned char* w1 = has_r1 ? (w0 + row_bytes) : w0;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (unsigned b = grp; b < nb; b += grp_stride) {
        const unsigned char* blk0 = w0 + (size_t)b * 144u;
        const unsigned char* blk1 = w1 + (size_t)b * 144u;
        uint4 hdr0 = *(const uint4*)blk0;
        uint4 q0 = *(const uint4*)(blk0 + 16u + (size_t)lane_in_blk * 16u);
        uint4 hdr1 = *(const uint4*)blk1;
        uint4 q1 = *(const uint4*)(blk1 + 16u + (size_t)lane_in_blk * 16u);

        const unsigned char* x_lo_blk = x_q8 + ((size_t)b * 8u + chunk * 2u) * 48u;
        const unsigned char* x_hi_blk = x_lo_blk + 48u;
        int4 qx_lo = *(const int4*)(x_lo_blk + (size_t)half_c * 16u);
        int4 qx_hi = *(const int4*)(x_hi_blk + (size_t)half_c * 16u);
        float2 ds_lo = *(const float2*)(x_lo_blk + 32u);
        float2 ds_hi = *(const float2*)(x_hi_blk + 32u);

        float d1_0, m1_0, d2_0, m2_0;
        scoria_q4k_unpack_scales(hdr0, chunk, &d1_0, &m1_0, &d2_0, &m2_0);
        int dot_lo0 = 0;
        dot_lo0 = scoria_dot4_u8_i8(q0.x & 0x0F0F0F0Fu, qx_lo.x, dot_lo0);
        dot_lo0 = scoria_dot4_u8_i8(q0.y & 0x0F0F0F0Fu, qx_lo.y, dot_lo0);
        dot_lo0 = scoria_dot4_u8_i8(q0.z & 0x0F0F0F0Fu, qx_lo.z, dot_lo0);
        dot_lo0 = scoria_dot4_u8_i8(q0.w & 0x0F0F0F0Fu, qx_lo.w, dot_lo0);
        int dot_hi0 = 0;
        dot_hi0 = scoria_dot4_u8_i8((q0.x >> 4) & 0x0F0F0F0Fu, qx_hi.x, dot_hi0);
        dot_hi0 = scoria_dot4_u8_i8((q0.y >> 4) & 0x0F0F0F0Fu, qx_hi.y, dot_hi0);
        dot_hi0 = scoria_dot4_u8_i8((q0.z >> 4) & 0x0F0F0F0Fu, qx_hi.z, dot_hi0);
        dot_hi0 = scoria_dot4_u8_i8((q0.w >> 4) & 0x0F0F0F0Fu, qx_hi.w, dot_hi0);
        acc0 += d1_0 * ds_lo.x * (float)dot_lo0 + d2_0 * ds_hi.x * (float)dot_hi0;
        if (half_c == 0u) {
            acc0 -= (m1_0 * ds_lo.y + m2_0 * ds_hi.y);
        }

        float d1_1, m1_1, d2_1, m2_1;
        scoria_q4k_unpack_scales(hdr1, chunk, &d1_1, &m1_1, &d2_1, &m2_1);
        int dot_lo1 = 0;
        dot_lo1 = scoria_dot4_u8_i8(q1.x & 0x0F0F0F0Fu, qx_lo.x, dot_lo1);
        dot_lo1 = scoria_dot4_u8_i8(q1.y & 0x0F0F0F0Fu, qx_lo.y, dot_lo1);
        dot_lo1 = scoria_dot4_u8_i8(q1.z & 0x0F0F0F0Fu, qx_lo.z, dot_lo1);
        dot_lo1 = scoria_dot4_u8_i8(q1.w & 0x0F0F0F0Fu, qx_lo.w, dot_lo1);
        int dot_hi1 = 0;
        dot_hi1 = scoria_dot4_u8_i8((q1.x >> 4) & 0x0F0F0F0Fu, qx_hi.x, dot_hi1);
        dot_hi1 = scoria_dot4_u8_i8((q1.y >> 4) & 0x0F0F0F0Fu, qx_hi.y, dot_hi1);
        dot_hi1 = scoria_dot4_u8_i8((q1.z >> 4) & 0x0F0F0F0Fu, qx_hi.z, dot_hi1);
        dot_hi1 = scoria_dot4_u8_i8((q1.w >> 4) & 0x0F0F0F0Fu, qx_hi.w, dot_hi1);
        acc1 += d1_1 * ds_lo.x * (float)dot_lo1 + d2_1 * ds_hi.x * (float)dot_hi1;
        if (half_c == 0u) {
            acc1 -= (m1_1 * ds_lo.y + m2_1 * ds_hi.y);
        }
    }

    float s0 = scoria_wave_sum(acc0);
    float s1 = scoria_wave_sum(acc1);
    if (lane == 0u) {
        y[row0] = s0 + (bias ? bias[row0] : 0.0f);
        if (has_r1) {
            y[row1] = s1 + (bias ? bias[row1] : 0.0f);
        }
    }
}

extern "C" __global__ void gemv_q4_k_gate_up_silu_q8_1_f32(const unsigned char* __restrict__ w_gate,
                                                           const unsigned char* __restrict__ w_up,
                                                           const unsigned char* __restrict__ x_q8,
                                                           float* __restrict__ y_out,
                                                           unsigned n,
                                                           unsigned k) {
    unsigned wave_id = threadIdx.x / SCORIA_WAVE_SIZE;
    unsigned lane = threadIdx.x & (SCORIA_WAVE_SIZE - 1u);
    unsigned waves_per_wg = blockDim.x / SCORIA_WAVE_SIZE;
    unsigned row = blockIdx.x * waves_per_wg + wave_id;
    if (row >= n) return;

    unsigned grp = lane >> 3;
    unsigned lane_in_blk = lane & 7u;
    unsigned grp_stride = SCORIA_WAVE_SIZE >> 3;
    unsigned chunk = lane_in_blk >> 1;
    unsigned half_c = lane_in_blk & 1u;
    unsigned nb = k / 256u;
    size_t row_bytes = (size_t)nb * 144u;

    const unsigned char* wg = w_gate + (size_t)row * row_bytes;
    const unsigned char* wu = w_up + (size_t)row * row_bytes;
    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (unsigned b = grp; b < nb; b += grp_stride) {
        const unsigned char* bg = wg + (size_t)b * 144u;
        const unsigned char* bu = wu + (size_t)b * 144u;
        uint4 hdr_g = *(const uint4*)bg;
        uint4 q_g = *(const uint4*)(bg + 16u + (size_t)lane_in_blk * 16u);
        uint4 hdr_u = *(const uint4*)bu;
        uint4 q_u = *(const uint4*)(bu + 16u + (size_t)lane_in_blk * 16u);

        const unsigned char* x_lo_blk = x_q8 + ((size_t)b * 8u + chunk * 2u) * 48u;
        const unsigned char* x_hi_blk = x_lo_blk + 48u;
        int4 qx_lo = *(const int4*)(x_lo_blk + (size_t)half_c * 16u);
        int4 qx_hi = *(const int4*)(x_hi_blk + (size_t)half_c * 16u);
        float2 ds_lo = *(const float2*)(x_lo_blk + 32u);
        float2 ds_hi = *(const float2*)(x_hi_blk + 32u);

        float d1g, m1g, d2g, m2g;
        scoria_q4k_unpack_scales(hdr_g, chunk, &d1g, &m1g, &d2g, &m2g);
        int dot_lo_g = 0;
        dot_lo_g = scoria_dot4_u8_i8(q_g.x & 0x0F0F0F0Fu, qx_lo.x, dot_lo_g);
        dot_lo_g = scoria_dot4_u8_i8(q_g.y & 0x0F0F0F0Fu, qx_lo.y, dot_lo_g);
        dot_lo_g = scoria_dot4_u8_i8(q_g.z & 0x0F0F0F0Fu, qx_lo.z, dot_lo_g);
        dot_lo_g = scoria_dot4_u8_i8(q_g.w & 0x0F0F0F0Fu, qx_lo.w, dot_lo_g);
        int dot_hi_g = 0;
        dot_hi_g = scoria_dot4_u8_i8((q_g.x >> 4) & 0x0F0F0F0Fu, qx_hi.x, dot_hi_g);
        dot_hi_g = scoria_dot4_u8_i8((q_g.y >> 4) & 0x0F0F0F0Fu, qx_hi.y, dot_hi_g);
        dot_hi_g = scoria_dot4_u8_i8((q_g.z >> 4) & 0x0F0F0F0Fu, qx_hi.z, dot_hi_g);
        dot_hi_g = scoria_dot4_u8_i8((q_g.w >> 4) & 0x0F0F0F0Fu, qx_hi.w, dot_hi_g);
        acc_g += d1g * ds_lo.x * (float)dot_lo_g + d2g * ds_hi.x * (float)dot_hi_g;

        float d1u, m1u, d2u, m2u;
        scoria_q4k_unpack_scales(hdr_u, chunk, &d1u, &m1u, &d2u, &m2u);
        int dot_lo_u = 0;
        dot_lo_u = scoria_dot4_u8_i8(q_u.x & 0x0F0F0F0Fu, qx_lo.x, dot_lo_u);
        dot_lo_u = scoria_dot4_u8_i8(q_u.y & 0x0F0F0F0Fu, qx_lo.y, dot_lo_u);
        dot_lo_u = scoria_dot4_u8_i8(q_u.z & 0x0F0F0F0Fu, qx_lo.z, dot_lo_u);
        dot_lo_u = scoria_dot4_u8_i8(q_u.w & 0x0F0F0F0Fu, qx_lo.w, dot_lo_u);
        int dot_hi_u = 0;
        dot_hi_u = scoria_dot4_u8_i8((q_u.x >> 4) & 0x0F0F0F0Fu, qx_hi.x, dot_hi_u);
        dot_hi_u = scoria_dot4_u8_i8((q_u.y >> 4) & 0x0F0F0F0Fu, qx_hi.y, dot_hi_u);
        dot_hi_u = scoria_dot4_u8_i8((q_u.z >> 4) & 0x0F0F0F0Fu, qx_hi.z, dot_hi_u);
        dot_hi_u = scoria_dot4_u8_i8((q_u.w >> 4) & 0x0F0F0F0Fu, qx_hi.w, dot_hi_u);
        acc_u += d1u * ds_lo.x * (float)dot_lo_u + d2u * ds_hi.x * (float)dot_hi_u;

        if (half_c == 0u) {
            acc_g -= (m1g * ds_lo.y + m2g * ds_hi.y);
            acc_u -= (m1u * ds_lo.y + m2u * ds_hi.y);
        }
    }

    float sg = scoria_wave_sum(acc_g);
    float su = scoria_wave_sum(acc_u);
    if (lane == 0u) {
        float silu = sg / (1.0f + expf(-sg));
        y_out[row] = silu * su;
    }
}

extern "C" __global__ void gemv_q6_k_f32(const unsigned char* __restrict__ weight,
                                         const float* __restrict__ x,
                                         const float* __restrict__ bias,
                                         float* __restrict__ y,
                                         unsigned n,
                                         unsigned k) {
    unsigned row0 = blockIdx.x * 2u;
    if (row0 >= n) return;
    unsigned row1 = row0 + 1u;
    bool has_r1 = row1 < n;

    unsigned tid = threadIdx.x;
    unsigned lane = tid & (SCORIA_WAVE_SIZE - 1u);
    unsigned wave_id = tid / SCORIA_WAVE_SIZE;
    unsigned n_waves = blockDim.x / SCORIA_WAVE_SIZE;
    unsigned sub_wave = tid >> 5;
    unsigned sub_lane = tid & 31u;
    unsigned h = sub_lane >> 4;
    unsigned p = (sub_lane & 15u) * 2u;
    int is = (int)(p >> 4);
    unsigned nb = k / 256u;

    const unsigned char* w0 = weight + (size_t)row0 * nb * 210u;
    const unsigned char* w1 = has_r1 ? (w0 + (size_t)nb * 210u) : nullptr;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (unsigned b = sub_wave; b < nb; b += 4u) {
        const float* xp = x + (size_t)b * 256u + (size_t)h * 128u + p;
        float2 x0 = *(const float2*)(xp);
        float2 x1 = *(const float2*)(xp + 32);
        float2 x2 = *(const float2*)(xp + 64);
        float2 x3 = *(const float2*)(xp + 96);

        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (r == 0 || has_r1) {
                const unsigned char* blk = ((r == 0) ? w0 : w1) + (size_t)b * 210u;
                unsigned short ql0_16 = *(const unsigned short*)(blk + h * 64u + p);
                unsigned short ql1_16 = *(const unsigned short*)(blk + h * 64u + 32u + p);
                unsigned short qh_16 = *(const unsigned short*)(blk + 128u + h * 32u + p);
                float d = scoria_f16_bits_to_f32(*(const unsigned short*)(blk + 208u));
                const unsigned short* sc16 = (const unsigned short*)(blk + 192u + h * 8u);
                int sc_shift = is * 8;
                float sc0 = (float)(signed char)((sc16[0] >> sc_shift) & 0xFF);
                float sc2 = (float)(signed char)((sc16[1] >> sc_shift) & 0xFF);
                float sc4 = (float)(signed char)((sc16[2] >> sc_shift) & 0xFF);
                float sc6 = (float)(signed char)((sc16[3] >> sc_shift) & 0xFF);

                unsigned ql0a = ql0_16 & 0xFFu;
                unsigned ql1a = ql1_16 & 0xFFu;
                unsigned qha = qh_16 & 0xFFu;
                int q1a = (int)((ql0a & 0x0Fu) | ((qha & 0x03u) << 4)) - 32;
                int q2a = (int)((ql1a & 0x0Fu) | (((qha >> 2) & 0x03u) << 4)) - 32;
                int q3a = (int)((ql0a >> 4) | (((qha >> 4) & 0x03u) << 4)) - 32;
                int q4a = (int)((ql1a >> 4) | (((qha >> 6) & 0x03u) << 4)) - 32;

                unsigned ql0b = ql0_16 >> 8;
                unsigned ql1b = ql1_16 >> 8;
                unsigned qhb = qh_16 >> 8;
                int q1b = (int)((ql0b & 0x0Fu) | ((qhb & 0x03u) << 4)) - 32;
                int q2b = (int)((ql1b & 0x0Fu) | (((qhb >> 2) & 0x03u) << 4)) - 32;
                int q3b = (int)((ql0b >> 4) | (((qhb >> 4) & 0x03u) << 4)) - 32;
                int q4b = (int)((ql1b >> 4) | (((qhb >> 6) & 0x03u) << 4)) - 32;

                float delta = d * (sc0 * ((float)q1a * x0.x + (float)q1b * x0.y)
                                 + sc2 * ((float)q2a * x1.x + (float)q2b * x1.y)
                                 + sc4 * ((float)q3a * x2.x + (float)q3b * x2.y)
                                 + sc6 * ((float)q4a * x3.x + (float)q4b * x3.y));
                if (r == 0) {
                    acc0 += delta;
                } else {
                    acc1 += delta;
                }
            }
        }
    }

    float w0s = scoria_wave_sum(acc0);
    float w1s = scoria_wave_sum(acc1);
    __shared__ float s_red[2][32];
    if (lane == 0) {
        s_red[0][wave_id] = w0s;
        s_red[1][wave_id] = w1s;
    }
    __syncthreads();
    if (tid == 0) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (unsigned w = 0; w < n_waves; ++w) {
            sum0 += s_red[0][w];
            sum1 += s_red[1][w];
        }
        y[row0] = sum0 + (bias ? bias[row0] : 0.0f);
        if (has_r1) {
            y[row1] = sum1 + (bias ? bias[row1] : 0.0f);
        }
    }
}

#ifdef SCORIA_RDNA4
typedef float float8_t __attribute__((ext_vector_type(8)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));

extern "C" __global__ void wmma_probe_f16(const _Float16* __restrict__ A,
                                          const _Float16* __restrict__ B,
                                          float* __restrict__ out_c) {
    unsigned lane = threadIdx.x & 31u;
    unsigned mn = lane & 15u;
    unsigned k0 = (lane >> 4) * 8u;
    half8_t a_frag, b_frag;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        a_frag[i] = A[mn * 16u + k0 + (unsigned)i];
        b_frag[i] = B[mn * 16u + k0 + (unsigned)i];
    }
    float8_t c_frag = {0};
    c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, c_frag);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_c[lane * 8u + (unsigned)i] = c_frag[i];
    }
}

__device__ __forceinline__ uint2 scoria_pack_h4(_Float16 h0, _Float16 h1, _Float16 h2, _Float16 h3) {
    union {
        _Float16 h[4];
        uint2 u;
    } p;
    p.h[0] = h0;
    p.h[1] = h1;
    p.h[2] = h2;
    p.h[3] = h3;
    return p.u;
}

struct WmmaLoaderF16 {
    struct Regs {
        uint4 v;
        bool valid;
    };

    static __device__ __forceinline__ Regs fetch(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid) {
        Regs r;
        unsigned rw = tid >> 2;
        unsigned l0 = (tid & 3u) * 8u;
        unsigned gn = n0 + rw;
        r.valid = gn < n;
        if (r.valid) {
            r.v = *(const uint4*)(weight + ((size_t)gn * k + k0 + l0) * 2u);
        } else {
            r.v = make_uint4(0u, 0u, 0u, 0u);
        }
        return r;
    }

    static __device__ __forceinline__ void store(const Regs& r,
                                                 unsigned /*k0*/,
                                                 unsigned tid,
                                                 _Float16 s_w[64][36]) {
        unsigned rw = tid >> 2;
        unsigned l0 = (tid & 3u) * 8u;
        uint2* dst = (uint2*)(&s_w[rw][l0]);
        dst[0] = make_uint2(r.v.x, r.v.y);
        dst[1] = make_uint2(r.v.z, r.v.w);
    }

    static __device__ __forceinline__ void load(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid,
                                                _Float16 s_w[64][36]) {
        Regs r = fetch(weight, k, n, n0, k0, tid);
        store(r, k0, tid, s_w);
    }
};

struct WmmaLoaderQ8_0 {
    struct Regs {
        float d;
        unsigned q_lo;
        unsigned q_hi;
        bool valid;
    };

    static __device__ __forceinline__ Regs fetch(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid) {
        Regs r;
        unsigned rw = tid >> 2;
        unsigned l0 = (tid & 3u) * 8u;
        unsigned gn = n0 + rw;
        r.valid = gn < n;
        if (!r.valid) {
            r.d = 0.0f;
            r.q_lo = 0u;
            r.q_hi = 0u;
            return r;
        }
        unsigned nb = k / 32u;
        unsigned b = k0 >> 5;
        const unsigned char* blk = weight + ((size_t)gn * nb + b) * 34u;
        r.d = scoria_f16_bits_to_f32(*(const unsigned short*)blk);
        const unsigned short* qs16 = (const unsigned short*)(blk + 2u + l0);
        r.q_lo = (unsigned)qs16[0] | ((unsigned)qs16[1] << 16);
        r.q_hi = (unsigned)qs16[2] | ((unsigned)qs16[3] << 16);
        return r;
    }

    static __device__ __forceinline__ void store(const Regs& r,
                                                 unsigned /*k0*/,
                                                 unsigned tid,
                                                 _Float16 s_w[64][36]) {
        unsigned rw = tid >> 2;
        unsigned l0 = (tid & 3u) * 8u;
        uint2* dst = (uint2*)(&s_w[rw][l0]);
        if (!r.valid) {
            dst[0] = make_uint2(0u, 0u);
            dst[1] = make_uint2(0u, 0u);
            return;
        }
        float d = r.d;
        dst[0] = scoria_pack_h4(
            (_Float16)(d * (float)(signed char)(r.q_lo & 0xFFu)),
            (_Float16)(d * (float)(signed char)((r.q_lo >> 8) & 0xFFu)),
            (_Float16)(d * (float)(signed char)((r.q_lo >> 16) & 0xFFu)),
            (_Float16)(d * (float)(signed char)(r.q_lo >> 24)));
        dst[1] = scoria_pack_h4(
            (_Float16)(d * (float)(signed char)(r.q_hi & 0xFFu)),
            (_Float16)(d * (float)(signed char)((r.q_hi >> 8) & 0xFFu)),
            (_Float16)(d * (float)(signed char)((r.q_hi >> 16) & 0xFFu)),
            (_Float16)(d * (float)(signed char)(r.q_hi >> 24)));
    }

    static __device__ __forceinline__ void load(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid,
                                                _Float16 s_w[64][36]) {
        Regs r = fetch(weight, k, n, n0, k0, tid);
        store(r, k0, tid, s_w);
    }
};

struct WmmaLoaderQ5_0 {
    struct Regs {
        float d;
        unsigned qh;
        unsigned q4bytes;
        bool valid;
    };

    static __device__ __forceinline__ Regs fetch(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid) {
        Regs r;
        unsigned rw = tid >> 2;
        unsigned t4 = tid & 3u;
        unsigned j0 = t4 * 4u;
        unsigned gn = n0 + rw;
        r.valid = gn < n;
        if (!r.valid) {
            r.d = 0.0f;
            r.qh = 0u;
            r.q4bytes = 0u;
            return r;
        }
        unsigned nb = k / 32u;
        unsigned b = k0 >> 5;
        const unsigned char* blk = weight + ((size_t)gn * nb + b) * 22u;
        r.d = scoria_f16_bits_to_f32(*(const unsigned short*)blk);
        r.qh = (unsigned)(*(const unsigned short*)(blk + 2u))
             | ((unsigned)(*(const unsigned short*)(blk + 4u)) << 16);
        const unsigned short* qs16 = (const unsigned short*)(blk + 6u + j0);
        r.q4bytes = (unsigned)qs16[0] | ((unsigned)qs16[1] << 16);
        return r;
    }

    static __device__ __forceinline__ void store(const Regs& r,
                                                 unsigned /*k0*/,
                                                 unsigned tid,
                                                 _Float16 s_w[64][36]) {
        unsigned rw = tid >> 2;
        unsigned j0 = (tid & 3u) * 4u;
        if (!r.valid) {
            *(uint2*)(&s_w[rw][j0]) = make_uint2(0u, 0u);
            *(uint2*)(&s_w[rw][j0 + 16u]) = make_uint2(0u, 0u);
            return;
        }
        _Float16 h0[4], h1[4];
        #pragma unroll
        for (unsigned u = 0; u < 4u; ++u) {
            unsigned j = j0 + u;
            unsigned qb = (r.q4bytes >> (u * 8u)) & 0xFFu;
            unsigned xh_0 = ((r.qh >> j) << 4) & 0x10u;
            unsigned xh_1 = (r.qh >> (j + 12u)) & 0x10u;
            int q0 = (int)((qb & 0x0Fu) | xh_0) - 16;
            int q1 = (int)((qb >> 4) | xh_1) - 16;
            h0[u] = (_Float16)(r.d * (float)q0);
            h1[u] = (_Float16)(r.d * (float)q1);
        }
        *(uint2*)(&s_w[rw][j0]) = scoria_pack_h4(h0[0], h0[1], h0[2], h0[3]);
        *(uint2*)(&s_w[rw][j0 + 16u]) = scoria_pack_h4(h1[0], h1[1], h1[2], h1[3]);
    }

    static __device__ __forceinline__ void load(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid,
                                                _Float16 s_w[64][36]) {
        Regs r = fetch(weight, k, n, n0, k0, tid);
        store(r, k0, tid, s_w);
    }
};

struct WmmaLoaderQ4_K {
    struct Regs {
        uint4 hdr;
        uint2 q8;
        bool valid;
    };

    static __device__ __forceinline__ Regs fetch(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid) {
        Regs r;
        unsigned rw = tid >> 2;
        unsigned l0 = (tid & 3u) * 8u;
        unsigned gn = n0 + rw;
        r.valid = gn < n;
        if (!r.valid) {
            r.hdr = make_uint4(0u, 0u, 0u, 0u);
            r.q8 = make_uint2(0u, 0u);
            return r;
        }
        unsigned nb = k / 256u;
        unsigned b = k0 >> 8;
        unsigned chunk = (k0 >> 6) & 3u;
        const unsigned char* blk = weight + ((size_t)gn * nb + b) * 144u;
        r.hdr = *(const uint4*)blk;
        r.q8 = *(const uint2*)(blk + 16u + chunk * 32u + l0);
        return r;
    }

    static __device__ __forceinline__ void store_half(const Regs& r,
                                                      unsigned chunk,
                                                      unsigned hi,
                                                      unsigned tid,
                                                      _Float16 s_w[64][36]) {
        unsigned rw = tid >> 2;
        unsigned l0 = (tid & 3u) * 8u;
        uint2* dst = (uint2*)(&s_w[rw][l0]);
        if (!r.valid) {
            dst[0] = make_uint2(0u, 0u);
            dst[1] = make_uint2(0u, 0u);
            return;
        }
        float d1, m1, d2, m2;
        scoria_q4k_unpack_scales(r.hdr, chunk, &d1, &m1, &d2, &m2);
        float d_eff = hi ? d2 : d1;
        float m_eff = hi ? m2 : m1;

        _Float16 lo_h[4], hi_h[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned b_lo = (r.q8.x >> (i * 8)) & 0xFFu;
            unsigned b_hi = (r.q8.y >> (i * 8)) & 0xFFu;
            unsigned nib_lo = hi ? (b_lo >> 4) : (b_lo & 0x0Fu);
            unsigned nib_hi = hi ? (b_hi >> 4) : (b_hi & 0x0Fu);
            lo_h[i] = (_Float16)(d_eff * (float)nib_lo - m_eff);
            hi_h[i] = (_Float16)(d_eff * (float)nib_hi - m_eff);
        }
        dst[0] = scoria_pack_h4(lo_h[0], lo_h[1], lo_h[2], lo_h[3]);
        dst[1] = scoria_pack_h4(hi_h[0], hi_h[1], hi_h[2], hi_h[3]);
    }

    static __device__ __forceinline__ void store(const Regs& r,
                                                 unsigned k0,
                                                 unsigned tid,
                                                 _Float16 s_w[64][36]) {
        unsigned sub32 = (k0 >> 5) & 7u;
        store_half(r, sub32 >> 1, sub32 & 1u, tid, s_w);
    }

    static __device__ __forceinline__ void load(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid,
                                                _Float16 s_w[64][36]) {
        Regs r = fetch(weight, k, n, n0, k0, tid);
        store(r, k0, tid, s_w);
    }
};

struct WmmaLoaderQ6_K {
    struct Regs {
        float d_sc;
        unsigned ql0;
        unsigned ql1;
        unsigned qh0;
        unsigned qh1;
        bool valid;
    };

    static __device__ __forceinline__ Regs fetch(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid) {
        Regs r;
        unsigned rw = tid >> 2;
        unsigned l0 = (tid & 3u) * 8u;
        unsigned gn = n0 + rw;
        r.valid = gn < n;
        if (!r.valid) {
            r.d_sc = 0.0f;
            r.ql0 = r.ql1 = r.qh0 = r.qh1 = 0u;
            return r;
        }
        unsigned nb = k / 256u;
        unsigned b = k0 >> 8;
        unsigned sub32 = (k0 >> 5) & 7u;
        unsigned h = sub32 >> 2;
        unsigned q_part = sub32 & 3u;
        unsigned is = l0 >> 4;
        const unsigned char* blk = weight + ((size_t)gn * nb + b) * 210u;
        float d = scoria_f16_bits_to_f32(*(const unsigned short*)(blk + 208u));
        float sc = (float)(signed char)blk[192u + h * 8u + is + q_part * 2u];
        r.d_sc = d * sc;

        const unsigned short* ql_ptr =
            (const unsigned short*)(blk + h * 64u + ((q_part & 1u) ? 32u : 0u) + l0);
        const unsigned short* qh_ptr = (const unsigned short*)(blk + 128u + h * 32u + l0);
        r.ql0 = (unsigned)ql_ptr[0] | ((unsigned)ql_ptr[1] << 16);
        r.ql1 = (unsigned)ql_ptr[2] | ((unsigned)ql_ptr[3] << 16);
        r.qh0 = (unsigned)qh_ptr[0] | ((unsigned)qh_ptr[1] << 16);
        r.qh1 = (unsigned)qh_ptr[2] | ((unsigned)qh_ptr[3] << 16);
        return r;
    }

    static __device__ __forceinline__ void store(const Regs& r,
                                                 unsigned k0,
                                                 unsigned tid,
                                                 _Float16 s_w[64][36]) {
        unsigned rw = tid >> 2;
        unsigned l0 = (tid & 3u) * 8u;
        uint2* dst = (uint2*)(&s_w[rw][l0]);
        if (!r.valid) {
            dst[0] = make_uint2(0u, 0u);
            dst[1] = make_uint2(0u, 0u);
            return;
        }
        unsigned sub32 = (k0 >> 5) & 7u;
        unsigned q_part = sub32 & 3u;
        unsigned ql_shift = (q_part >= 2u) ? 4u : 0u;
        unsigned qh_shift = q_part * 2u;

        _Float16 h0[4], h1[4];
        #pragma unroll
        for (unsigned i = 0; i < 4u; ++i) {
            unsigned qla = (r.ql0 >> (i * 8u)) & 0xFFu;
            unsigned qha = (r.qh0 >> (i * 8u)) & 0xFFu;
            int qa = (int)(((qla >> ql_shift) & 0x0Fu) | (((qha >> qh_shift) & 0x03u) << 4)) - 32;
            h0[i] = (_Float16)(r.d_sc * (float)qa);

            unsigned qlb = (r.ql1 >> (i * 8u)) & 0xFFu;
            unsigned qhb = (r.qh1 >> (i * 8u)) & 0xFFu;
            int qb = (int)(((qlb >> ql_shift) & 0x0Fu) | (((qhb >> qh_shift) & 0x03u) << 4)) - 32;
            h1[i] = (_Float16)(r.d_sc * (float)qb);
        }
        dst[0] = scoria_pack_h4(h0[0], h0[1], h0[2], h0[3]);
        dst[1] = scoria_pack_h4(h1[0], h1[1], h1[2], h1[3]);
    }

    static __device__ __forceinline__ void load(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid,
                                                _Float16 s_w[64][36]) {
        Regs r = fetch(weight, k, n, n0, k0, tid);
        store(r, k0, tid, s_w);
    }
};

__device__ __forceinline__ void scoria_fetch_x64(const float* __restrict__ x,
                                                 unsigned m0,
                                                 unsigned m,
                                                 unsigned k,
                                                 unsigned k0,
                                                 unsigned rx,
                                                 unsigned cx,
                                                 float4* xv0,
                                                 float4* xv1) {
    unsigned gm0 = m0 + rx;
    unsigned gm1 = m0 + 32u + rx;
    *xv0 = (gm0 < m) ? *(const float4*)(x + (size_t)gm0 * k + k0 + cx)
                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    *xv1 = (gm1 < m) ? *(const float4*)(x + (size_t)gm1 * k + k0 + cx)
                     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ void scoria_store_x64(float4 xv0,
                                                 float4 xv1,
                                                 unsigned rx,
                                                 unsigned cx,
                                                 _Float16 s_x[64][36]) {
    *(uint2*)(&s_x[rx][cx]) = scoria_pack_h4(
        (_Float16)xv0.x, (_Float16)xv0.y, (_Float16)xv0.z, (_Float16)xv0.w);
    *(uint2*)(&s_x[32u + rx][cx]) = scoria_pack_h4(
        (_Float16)xv1.x, (_Float16)xv1.y, (_Float16)xv1.z, (_Float16)xv1.w);
}

__device__ __forceinline__ void scoria_wmma_step64x128(const _Float16 s_x[64][36],
                                                      const _Float16 s_w[128][36],
                                                      unsigned wm,
                                                      unsigned wn,
                                                      unsigned mn,
                                                      unsigned k_off,
                                                      float8_t acc[4]) {
    #pragma unroll
    for (unsigned kk = 0u; kk < 32u; kk += 16u) {
        half8_t a0, a1, b0, b1;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            a0[i] = s_x[(wm * 2u + 0u) * 16u + mn][kk + k_off + (unsigned)i];
            a1[i] = s_x[(wm * 2u + 1u) * 16u + mn][kk + k_off + (unsigned)i];
            b0[i] = s_w[(wn * 2u + 0u) * 16u + mn][kk + k_off + (unsigned)i];
            b1[i] = s_w[(wn * 2u + 1u) * 16u + mn][kk + k_off + (unsigned)i];
        }
        acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, b0, acc[0]);
        acc[1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, b1, acc[1]);
        acc[2] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, b0, acc[2]);
        acc[3] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, b1, acc[3]);
    }
}

// Double-buffered prefill GEMM: 64x128 macro-tile, 32-deep K step, four 16x16 WMMA
// accumulators per wave, split fetch/store loaders with packed 64-bit LDS stores and
// one barrier per K-step overlapping global loads with WMMA math.
template <typename WeightLoader>
__device__ __forceinline__ void gemm_wmma_wide_impl(const unsigned char* __restrict__ weight,
                                                     const float* __restrict__ x,
                                                     const float* __restrict__ bias,
                                                     float* __restrict__ y,
                                                     unsigned m,
                                                     unsigned n,
                                                     unsigned k) {
    if (m == 0u || n == 0u || k == 0u) return;
    unsigned tid = threadIdx.x;
    unsigned wave_id = tid >> 5;
    unsigned lane = tid & 31u;
    unsigned wm = wave_id >> 2;
    unsigned wn = wave_id & 3u;
    unsigned m0 = blockIdx.y * 64u;
    unsigned n0 = blockIdx.x * 128u;

    __shared__ _Float16 s_x[2][64][36];
    __shared__ _Float16 s_w[2][128][36];

    float8_t acc[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    unsigned rx = tid >> 3;
    unsigned cx = (tid & 7u) * 4u;
    unsigned mn = lane & 15u;
    unsigned k_off = (lane >> 4) * 8u;

    float4 xv0, xv1;
    typename WeightLoader::Regs rw0, rw1;
    scoria_fetch_x64(x, m0, m, k, 0u, rx, cx, &xv0, &xv1);
    rw0 = WeightLoader::fetch(weight, k, n, n0, 0u, tid);
    rw1 = WeightLoader::fetch(weight, k, n, n0 + 64u, 0u, tid);
    scoria_store_x64(xv0, xv1, rx, cx, s_x[0]);
    WeightLoader::store(rw0, 0u, tid, s_w[0]);
    WeightLoader::store(rw1, 0u, tid, s_w[0] + 64);
    __syncthreads();

    unsigned buf = 0u;
    for (unsigned k0 = 32u; k0 < k; k0 += 32u) {
        scoria_fetch_x64(x, m0, m, k, k0, rx, cx, &xv0, &xv1);
        rw0 = WeightLoader::fetch(weight, k, n, n0, k0, tid);
        rw1 = WeightLoader::fetch(weight, k, n, n0 + 64u, k0, tid);

        scoria_wmma_step64x128(s_x[buf], s_w[buf], wm, wn, mn, k_off, acc);

        unsigned next_buf = buf ^ 1u;
        scoria_store_x64(xv0, xv1, rx, cx, s_x[next_buf]);
        WeightLoader::store(rw0, k0, tid, s_w[next_buf]);
        WeightLoader::store(rw1, k0, tid, s_w[next_buf] + 64);
        __syncthreads();
        buf = next_buf;
    }

    scoria_wmma_step64x128(s_x[buf], s_w[buf], wm, wn, mn, k_off, acc);

    #pragma unroll
    for (unsigned im = 0; im < 2u; ++im) {
        #pragma unroll
        for (unsigned in = 0; in < 2u; ++in) {
            float8_t c = acc[im * 2u + in];
            unsigned out_n = n0 + (wn * 2u + in) * 16u + mn;
            if (out_n < n) {
                float b_val = bias ? bias[out_n] : 0.0f;
                unsigned m_base = m0 + (wm * 2u + im) * 16u + (lane >> 4) * 8u;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    unsigned out_m = m_base + (unsigned)i;
                    if (out_m < m) {
                        y[(size_t)out_m * n + out_n] = c[i] + b_val;
                    }
                }
            }
        }
    }
}

// Specialized 2-half Q4_K double-buffered GEMM: fetches uint4 hdr + uint2 q8 ONCE per 64
// K-elements and keeps them resident across both lo (k0) and hi (k0+32) halves.
__device__ __forceinline__ void gemm_wmma_q4_k_dbuf_impl(const unsigned char* __restrict__ weight,
                                                         const float* __restrict__ x,
                                                         const float* __restrict__ bias,
                                                         float* __restrict__ y,
                                                         unsigned m,
                                                         unsigned n,
                                                         unsigned k) {
    if (m == 0u || n == 0u || k == 0u) return;
    unsigned tid = threadIdx.x;
    unsigned wave_id = tid >> 5;
    unsigned lane = tid & 31u;
    unsigned wm = wave_id >> 2;
    unsigned wn = wave_id & 3u;
    unsigned m0 = blockIdx.y * 64u;
    unsigned n0 = blockIdx.x * 128u;

    __shared__ _Float16 s_x[2][64][36];
    __shared__ _Float16 s_w[2][128][36];

    float8_t acc[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    unsigned rx = tid >> 3;
    unsigned cx = (tid & 7u) * 4u;
    unsigned mn = lane & 15u;
    unsigned k_off = (lane >> 4) * 8u;

    float4 xv0, xv1;
    WmmaLoaderQ4_K::Regs rw0 = WmmaLoaderQ4_K::fetch(weight, k, n, n0, 0u, tid);
    WmmaLoaderQ4_K::Regs rw1 = WmmaLoaderQ4_K::fetch(weight, k, n, n0 + 64u, 0u, tid);
    scoria_fetch_x64(x, m0, m, k, 0u, rx, cx, &xv0, &xv1);
    scoria_store_x64(xv0, xv1, rx, cx, s_x[0]);
    WmmaLoaderQ4_K::store_half(rw0, 0u, 0u, tid, s_w[0]);
    WmmaLoaderQ4_K::store_half(rw1, 0u, 0u, tid, s_w[0] + 64);
    __syncthreads();

    unsigned buf = 0u;
    for (unsigned k0 = 0u; k0 < k; k0 += 64u) {
        unsigned chunk = (k0 >> 6) & 3u;

        // Half 0 compute on `buf` overlapped with X(k0 + 32) load + W_hi store into `buf ^ 1`.
        scoria_fetch_x64(x, m0, m, k, k0 + 32u, rx, cx, &xv0, &xv1);
        scoria_wmma_step64x128(s_x[buf], s_w[buf], wm, wn, mn, k_off, acc);
        unsigned b_hi = buf ^ 1u;
        scoria_store_x64(xv0, xv1, rx, cx, s_x[b_hi]);
        WmmaLoaderQ4_K::store_half(rw0, chunk, 1u, tid, s_w[b_hi]);
        WmmaLoaderQ4_K::store_half(rw1, chunk, 1u, tid, s_w[b_hi] + 64);
        __syncthreads();
        buf = b_hi;

        // Half 1 compute on `buf` overlapped with next 64-K X & W fetch if any remain.
        if (k0 + 64u < k) {
            unsigned nk = k0 + 64u;
            unsigned nchunk = (nk >> 6) & 3u;
            scoria_fetch_x64(x, m0, m, k, nk, rx, cx, &xv0, &xv1);
            rw0 = WmmaLoaderQ4_K::fetch(weight, k, n, n0, nk, tid);
            rw1 = WmmaLoaderQ4_K::fetch(weight, k, n, n0 + 64u, nk, tid);
            scoria_wmma_step64x128(s_x[buf], s_w[buf], wm, wn, mn, k_off, acc);
            unsigned b_next = buf ^ 1u;
            scoria_store_x64(xv0, xv1, rx, cx, s_x[b_next]);
            WmmaLoaderQ4_K::store_half(rw0, nchunk, 0u, tid, s_w[b_next]);
            WmmaLoaderQ4_K::store_half(rw1, nchunk, 0u, tid, s_w[b_next] + 64);
            __syncthreads();
            buf = b_next;
        } else {
            scoria_wmma_step64x128(s_x[buf], s_w[buf], wm, wn, mn, k_off, acc);
        }
    }

    #pragma unroll
    for (unsigned im = 0; im < 2u; ++im) {
        #pragma unroll
        for (unsigned in = 0; in < 2u; ++in) {
            float8_t c = acc[im * 2u + in];
            unsigned out_n = n0 + (wn * 2u + in) * 16u + mn;
            if (out_n < n) {
                float b_val = bias ? bias[out_n] : 0.0f;
                unsigned m_base = m0 + (wm * 2u + im) * 16u + (lane >> 4) * 8u;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    unsigned out_m = m_base + (unsigned)i;
                    if (out_m < m) {
                        y[(size_t)out_m * n + out_n] = c[i] + b_val;
                    }
                }
            }
        }
    }
}

// Narrow variant: 32x64 macro-tile (one 16x16 WMMA per wave). Keeps enough blocks to
// fill the CUs when the output matrix is small (e.g. Qwen3-4B kv projections at
// n=1024, where the wide tile would launch only 16 blocks for 64 CUs).
template <typename WeightLoader>
__device__ __forceinline__ void gemm_wmma_narrow_impl(const unsigned char* __restrict__ weight,
                                                      const float* __restrict__ x,
                                                      const float* __restrict__ bias,
                                                      float* __restrict__ y,
                                                      unsigned m,
                                                      unsigned n,
                                                      unsigned k) {
    if (m == 0u || n == 0u || k == 0u) return;
    unsigned tid = threadIdx.x;
    unsigned wave_id = tid >> 5;
    unsigned lane = tid & 31u;
    unsigned wm = wave_id >> 2;
    unsigned wn = wave_id & 3u;
    unsigned m0 = blockIdx.y * 32u;
    unsigned n0 = blockIdx.x * 64u;

    __shared__ _Float16 s_x[32][36];
    __shared__ _Float16 s_w[64][36];

    float8_t c_frag = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    unsigned rx = tid >> 3;
    unsigned cx = (tid & 7u) * 4u;
    unsigned gm = m0 + rx;

    unsigned mn = lane & 15u;
    unsigned k_off = (lane >> 4) * 8u;

    for (unsigned k0 = 0; k0 < k; k0 += 32u) {
        if (gm < m) {
            float4 xv = *(const float4*)(x + (size_t)gm * k + k0 + cx);
            s_x[rx][cx + 0u] = (_Float16)xv.x;
            s_x[rx][cx + 1u] = (_Float16)xv.y;
            s_x[rx][cx + 2u] = (_Float16)xv.z;
            s_x[rx][cx + 3u] = (_Float16)xv.w;
        } else {
            s_x[rx][cx + 0u] = (_Float16)0.0f;
            s_x[rx][cx + 1u] = (_Float16)0.0f;
            s_x[rx][cx + 2u] = (_Float16)0.0f;
            s_x[rx][cx + 3u] = (_Float16)0.0f;
        }

        WeightLoader::load(weight, k, n, n0, k0, tid, s_w);
        __syncthreads();

        #pragma unroll
        for (unsigned kk = 0u; kk < 32u; kk += 16u) {
            half8_t a_frag, b_frag;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                a_frag[i] = s_x[wm * 16u + mn][kk + k_off + (unsigned)i];
                b_frag[i] = s_w[wn * 16u + mn][kk + k_off + (unsigned)i];
            }
            c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    unsigned out_n = n0 + wn * 16u + (lane & 15u);
    if (out_n < n) {
        float b_val = bias ? bias[out_n] : 0.0f;
        unsigned m_base = m0 + wm * 16u + (lane >> 4) * 8u;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned out_m = m_base + (unsigned)i;
            if (out_m < m) {
                y[(size_t)out_m * n + out_n] = c_frag[i] + b_val;
            }
        }
    }
}

extern "C" __global__ void gemm_wmma_narrow_f16(const unsigned char* __restrict__ weight,
                                                const float* __restrict__ x,
                                                const float* __restrict__ bias,
                                                float* __restrict__ y,
                                                unsigned m,
                                                unsigned n,
                                                unsigned k) {
    gemm_wmma_narrow_impl<WmmaLoaderF16>(weight, x, bias, y, m, n, k);
}

extern "C" __global__ void gemm_wmma_narrow_q5_0(const unsigned char* __restrict__ weight,
                                                 const float* __restrict__ x,
                                                 const float* __restrict__ bias,
                                                 float* __restrict__ y,
                                                 unsigned m,
                                                 unsigned n,
                                                 unsigned k) {
    gemm_wmma_narrow_impl<WmmaLoaderQ5_0>(weight, x, bias, y, m, n, k);
}

extern "C" __global__ void gemm_wmma_narrow_q8_0(const unsigned char* __restrict__ weight,
                                                 const float* __restrict__ x,
                                                 const float* __restrict__ bias,
                                                 float* __restrict__ y,
                                                 unsigned m,
                                                 unsigned n,
                                                 unsigned k) {
    gemm_wmma_narrow_impl<WmmaLoaderQ8_0>(weight, x, bias, y, m, n, k);
}

extern "C" __global__ void gemm_wmma_narrow_q4_k(const unsigned char* __restrict__ weight,
                                                 const float* __restrict__ x,
                                                 const float* __restrict__ bias,
                                                 float* __restrict__ y,
                                                 unsigned m,
                                                 unsigned n,
                                                 unsigned k) {
    gemm_wmma_narrow_impl<WmmaLoaderQ4_K>(weight, x, bias, y, m, n, k);
}

extern "C" __global__ void gemm_wmma_narrow_q6_k(const unsigned char* __restrict__ weight,
                                                 const float* __restrict__ x,
                                                 const float* __restrict__ bias,
                                                 float* __restrict__ y,
                                                 unsigned m,
                                                 unsigned n,
                                                 unsigned k) {
    gemm_wmma_narrow_impl<WmmaLoaderQ6_K>(weight, x, bias, y, m, n, k);
}

extern "C" __global__ void gemm_wmma_f16(const unsigned char* __restrict__ weight,
                                         const float* __restrict__ x,
                                         const float* __restrict__ bias,
                                         float* __restrict__ y,
                                         unsigned m,
                                         unsigned n,
                                         unsigned k) {
    gemm_wmma_wide_impl<WmmaLoaderF16>(weight, x, bias, y, m, n, k);
}

extern "C" __global__ void gemm_wmma_q5_0(const unsigned char* __restrict__ weight,
                                          const float* __restrict__ x,
                                          const float* __restrict__ bias,
                                          float* __restrict__ y,
                                          unsigned m,
                                          unsigned n,
                                          unsigned k) {
    gemm_wmma_wide_impl<WmmaLoaderQ5_0>(weight, x, bias, y, m, n, k);
}

extern "C" __global__ void gemm_wmma_q8_0(const unsigned char* __restrict__ weight,
                                          const float* __restrict__ x,
                                          const float* __restrict__ bias,
                                          float* __restrict__ y,
                                          unsigned m,
                                          unsigned n,
                                          unsigned k) {
    gemm_wmma_wide_impl<WmmaLoaderQ8_0>(weight, x, bias, y, m, n, k);
}

extern "C" __global__ void gemm_wmma_q4_k(const unsigned char* __restrict__ weight,
                                          const float* __restrict__ x,
                                          const float* __restrict__ bias,
                                          float* __restrict__ y,
                                          unsigned m,
                                          unsigned n,
                                          unsigned k) {
    gemm_wmma_q4_k_dbuf_impl(weight, x, bias, y, m, n, k);
}

extern "C" __global__ void gemm_wmma_q6_k(const unsigned char* __restrict__ weight,
                                          const float* __restrict__ x,
                                          const float* __restrict__ bias,
                                          float* __restrict__ y,
                                          unsigned m,
                                          unsigned n,
                                          unsigned k) {
    gemm_wmma_wide_impl<WmmaLoaderQ6_K>(weight, x, bias, y, m, n, k);
}
#endif
