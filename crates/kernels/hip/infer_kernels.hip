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
    __shared__ float s_sums[8];
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

    float inv_freq = powf(freq_base, -2.0f * (float)i / (float)head_dim);
    float theta = (float)(start_pos + tok) * inv_freq;
    float c = cosf(theta);
    float s = sinf(theta);

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
    __shared__ float s_sums[8];
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
    unsigned tid = threadIdx.x;
    unsigned lane = tid & (SCORIA_WAVE_SIZE - 1u);
    unsigned wave_id = tid / SCORIA_WAVE_SIZE;
    unsigned n_waves = blockDim.x / SCORIA_WAVE_SIZE;
    unsigned stride = gridDim.x * blockDim.x;

    float best_v = -1e38f;
    unsigned best_i = 0u;
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
        float v = vals[i];
        if (v > best_v) {
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

    __shared__ float s_v[8];
    __shared__ unsigned s_i[8];
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
    unsigned tid = threadIdx.x;
    unsigned lane = tid & (SCORIA_WAVE_SIZE - 1u);
    unsigned wave_id = tid / SCORIA_WAVE_SIZE;
    unsigned n_waves = blockDim.x / SCORIA_WAVE_SIZE;

    float best_v = (tid < count) ? in_vals[tid] : -1e38f;
    unsigned best_i = (tid < count) ? in_idxs[tid] : 0u;

    for (int off = SCORIA_WAVE_SIZE >> 1; off > 0; off >>= 1) {
        float ov = __shfl_xor(best_v, off, SCORIA_WAVE_SIZE);
        unsigned oi = __shfl_xor(best_i, off, SCORIA_WAVE_SIZE);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    __shared__ float s_v[8];
    __shared__ unsigned s_i[8];
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
    __shared__ float s_red[2][4];
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

struct WmmaLoaderF16 {
    static __device__ __forceinline__ void load(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid,
                                                _Float16 s_w[64][36]) {
        unsigned rw = tid >> 2;
        unsigned t4 = tid & 3u;
        unsigned l0 = t4 * 8u;
        unsigned gn = n0 + rw;
        if (gn >= n) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) s_w[rw][l0 + (unsigned)i] = (_Float16)0.0f;
            return;
        }
        const unsigned char* ptr = weight + ((size_t)gn * k + k0 + l0) * 2u;
        uint4 v = *(const uint4*)ptr;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned word = (i == 0) ? v.x : ((i == 1) ? v.y : ((i == 2) ? v.z : v.w));
            union {
                unsigned short u;
                _Float16 h;
            } lo_c, hi_c;
            lo_c.u = (unsigned short)(word & 0xFFFFu);
            hi_c.u = (unsigned short)(word >> 16);
            s_w[rw][l0 + (unsigned)i * 2u] = lo_c.h;
            s_w[rw][l0 + (unsigned)i * 2u + 1u] = hi_c.h;
        }
    }
};

struct WmmaLoaderQ8_0 {
    static __device__ __forceinline__ void load(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid,
                                                _Float16 s_w[64][36]) {
        unsigned rw = tid >> 2;
        unsigned t4 = tid & 3u;
        unsigned l0 = t4 * 8u;
        unsigned gn = n0 + rw;
        if (gn >= n) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) s_w[rw][l0 + (unsigned)i] = (_Float16)0.0f;
            return;
        }
        unsigned nb = k / 32u;
        unsigned b = k0 >> 5;
        const unsigned char* blk = weight + ((size_t)gn * nb + b) * 34u;
        float d = scoria_f16_bits_to_f32(*(const unsigned short*)blk);
        const signed char* qs = (const signed char*)(blk + 2u + l0);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_w[rw][l0 + (unsigned)i] = (_Float16)(d * (float)qs[i]);
        }
    }
};

struct WmmaLoaderQ5_0 {
    static __device__ __forceinline__ void load(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid,
                                                _Float16 s_w[64][36]) {
        unsigned rw = tid >> 2;
        unsigned t4 = tid & 3u;
        unsigned j0 = t4 * 4u;
        unsigned gn = n0 + rw;
        if (gn >= n) {
            #pragma unroll
            for (unsigned u = 0; u < 4u; ++u) {
                s_w[rw][j0 + u] = (_Float16)0.0f;
                s_w[rw][j0 + 16u + u] = (_Float16)0.0f;
            }
            return;
        }
        unsigned nb = k / 32u;
        unsigned b = k0 >> 5;
        const unsigned char* blk = weight + ((size_t)gn * nb + b) * 22u;
        float d = scoria_f16_bits_to_f32(*(const unsigned short*)blk);
        unsigned qh = (unsigned)(*(const unsigned short*)(blk + 2u))
                    | ((unsigned)(*(const unsigned short*)(blk + 4u)) << 16);
        #pragma unroll
        for (unsigned u = 0; u < 4u; ++u) {
            unsigned j = j0 + u;
            unsigned char qb = blk[6u + j];
            unsigned xh_0 = ((qh >> j) << 4) & 0x10u;
            unsigned xh_1 = (qh >> (j + 12u)) & 0x10u;
            int q0 = (int)((qb & 0x0Fu) | xh_0) - 16;
            int q1 = (int)((qb >> 4) | xh_1) - 16;
            s_w[rw][j] = (_Float16)(d * (float)q0);
            s_w[rw][j + 16u] = (_Float16)(d * (float)q1);
        }
    }
};

struct WmmaLoaderQ4_K {
    static __device__ __forceinline__ void load(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid,
                                                _Float16 s_w[64][36]) {
        unsigned rw = tid >> 2;
        unsigned t4 = tid & 3u;
        unsigned l0 = t4 * 8u;
        unsigned gn = n0 + rw;
        if (gn >= n) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) s_w[rw][l0 + (unsigned)i] = (_Float16)0.0f;
            return;
        }
        unsigned nb = k / 256u;
        unsigned b = k0 >> 8;
        unsigned sub32 = (k0 >> 5) & 7u;
        unsigned chunk = sub32 >> 1;
        unsigned hi = sub32 & 1u;
        const unsigned char* blk = weight + ((size_t)gn * nb + b) * 144u;
        uint4 hdr = *(const uint4*)blk;
        float d1, m1, d2, m2;
        scoria_q4k_unpack_scales(hdr, chunk, &d1, &m1, &d2, &m2);
        float d_eff = hi ? d2 : d1;
        float m_eff = hi ? m2 : m1;

        uint2 q8 = *(const uint2*)(blk + 16u + chunk * 32u + l0);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            unsigned b_lo = (q8.x >> (i * 8)) & 0xFFu;
            unsigned b_hi = (q8.y >> (i * 8)) & 0xFFu;
            unsigned nib_lo = hi ? (b_lo >> 4) : (b_lo & 0x0Fu);
            unsigned nib_hi = hi ? (b_hi >> 4) : (b_hi & 0x0Fu);
            s_w[rw][l0 + (unsigned)i] = (_Float16)(d_eff * (float)nib_lo - m_eff);
            s_w[rw][l0 + 4u + (unsigned)i] = (_Float16)(d_eff * (float)nib_hi - m_eff);
        }
    }
};

struct WmmaLoaderQ6_K {
    static __device__ __forceinline__ void load(const unsigned char* __restrict__ weight,
                                                unsigned k,
                                                unsigned n,
                                                unsigned n0,
                                                unsigned k0,
                                                unsigned tid,
                                                _Float16 s_w[64][36]) {
        unsigned rw = tid >> 2;
        unsigned t4 = tid & 3u;
        unsigned l0 = t4 * 8u;
        unsigned gn = n0 + rw;
        if (gn >= n) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) s_w[rw][l0 + (unsigned)i] = (_Float16)0.0f;
            return;
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
        float d_sc = d * sc;

        unsigned ql_off = h * 64u + ((q_part & 1u) ? 32u : 0u) + l0;
        unsigned qh_off = 128u + h * 32u + l0;
        unsigned ql_shift = (q_part >= 2u) ? 4u : 0u;
        unsigned qh_shift = q_part * 2u;
        #pragma unroll
        for (unsigned pair = 0; pair < 4u; ++pair) {
            unsigned short ql16 = *(const unsigned short*)(blk + ql_off + pair * 2u);
            unsigned short qh16 = *(const unsigned short*)(blk + qh_off + pair * 2u);
            unsigned qla = ql16 & 0xFFu;
            unsigned qha = qh16 & 0xFFu;
            int qa = (int)(((qla >> ql_shift) & 0x0Fu) | (((qha >> qh_shift) & 0x03u) << 4)) - 32;
            unsigned qlb = ql16 >> 8;
            unsigned qhb = qh16 >> 8;
            int qb = (int)(((qlb >> ql_shift) & 0x0Fu) | (((qhb >> qh_shift) & 0x03u) << 4)) - 32;
            s_w[rw][l0 + pair * 2u] = (_Float16)(d_sc * (float)qa);
            s_w[rw][l0 + pair * 2u + 1u] = (_Float16)(d_sc * (float)qb);
        }
    }
};

// Prefill GEMM: 64x128 macro-tile, 32-deep K step, four 16x16 WMMA accumulators per
// wave. Eight waves cover the tile as 2 (m) x 4 (n) wave-tile pairs of 2x2 fragments,
// which doubles arithmetic intensity to ~43 flop/byte over the previous 32x64 tile
// (21 flop/byte) and halves the LDS traffic per WMMA.
template <typename WeightLoader>
__device__ __forceinline__ void gemm_wmma_wide_impl(const unsigned char* __restrict__ weight,
                                                     const float* __restrict__ x,
                                                     const float* __restrict__ bias,
                                                     float* __restrict__ y,
                                                     unsigned m,
                                                     unsigned n,
                                                     unsigned k) {
    unsigned tid = threadIdx.x;
    unsigned wave_id = tid >> 5;
    unsigned lane = tid & 31u;
    unsigned wm = wave_id >> 2;
    unsigned wn = wave_id & 3u;
    unsigned m0 = blockIdx.y * 64u;
    unsigned n0 = blockIdx.x * 128u;

    __shared__ _Float16 s_x[64][36];
    __shared__ _Float16 s_w[128][36];

    float8_t acc[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = (float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    unsigned rx = tid >> 3;
    unsigned cx = (tid & 7u) * 4u;

    unsigned mn = lane & 15u;
    unsigned k_off = (lane >> 4) * 8u;

    for (unsigned k0 = 0; k0 < k; k0 += 32u) {
        #pragma unroll
        for (unsigned half = 0; half < 2u; ++half) {
            unsigned gm = m0 + half * 32u + rx;
            _Float16* dst = &s_x[half * 32u + rx][cx];
            if (gm < m) {
                float4 xv = *(const float4*)(x + (size_t)gm * k + k0 + cx);
                dst[0] = (_Float16)xv.x;
                dst[1] = (_Float16)xv.y;
                dst[2] = (_Float16)xv.z;
                dst[3] = (_Float16)xv.w;
            } else {
                dst[0] = (_Float16)0.0f;
                dst[1] = (_Float16)0.0f;
                dst[2] = (_Float16)0.0f;
                dst[3] = (_Float16)0.0f;
            }
        }

        WeightLoader::load(weight, k, n, n0, k0, tid, s_w);
        WeightLoader::load(weight, k, n, n0 + 64u, k0, tid, s_w + 64);
        __syncthreads();

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
        __syncthreads();
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
    gemm_wmma_wide_impl<WmmaLoaderQ4_K>(weight, x, bias, y, m, n, k);
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
