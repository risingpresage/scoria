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
    for (unsigned i = tid; i < n_embd; i += blockDim.x) {
        float v = x_row[i];
        ss += v * v;
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
    for (unsigned i = tid; i < n_embd; i += blockDim.x) {
        out_row[i] = x_row[i] * inv * w[i];
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
                                         unsigned m,
                                         unsigned start_pos,
                                         unsigned n_heads,
                                         unsigned n_kv_heads,
                                         unsigned head_dim,
                                         float freq_base) {
    unsigned tok = blockIdx.x;
    unsigned h = blockIdx.y;
    if (tok >= m || h >= (n_heads + n_kv_heads)) return;

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
                                               unsigned m,
                                               unsigned start_pos,
                                               unsigned kv_dim) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = m * kv_dim;
    if (idx >= total) return;
    size_t dst_idx = (size_t)start_pos * kv_dim + idx;
    k_cache[dst_idx] = k_in[idx];
    v_cache[dst_idx] = v_in[idx];
}

extern "C" __global__ void gqa_attn_f32(const float* __restrict__ q,
                                        const float* __restrict__ k_cache,
                                        const float* __restrict__ v_cache,
                                        float* __restrict__ out,
                                        unsigned m,
                                        unsigned start_pos,
                                        unsigned n_heads,
                                        unsigned n_kv_heads,
                                        unsigned head_dim) {
    unsigned q_head = blockIdx.x;
    unsigned tok = blockIdx.y;
    if (q_head >= n_heads || tok >= m) return;

    unsigned d = threadIdx.x;
    unsigned lane = d & (SCORIA_WAVE_SIZE - 1u);
    unsigned wave_id = d / SCORIA_WAVE_SIZE;
    unsigned n_waves = blockDim.x / SCORIA_WAVE_SIZE;

    unsigned kv_group = n_heads / n_kv_heads;
    unsigned kv_head = q_head / kv_group;
    unsigned kv_len = start_pos + tok + 1u;
    float scale = rsqrtf((float)head_dim);

    float q_d = (d < head_dim) ? q[((size_t)tok * n_heads + q_head) * head_dim + d] : 0.0f;

    __shared__ float s_dot[8];
    __shared__ float s_score;

    float m_i = -1e38f;
    float l_i = 0.0f;
    float acc_v = 0.0f;

    for (unsigned t = 0; t < kv_len; ++t) {
        const float* k_ptr = k_cache + ((size_t)t * n_kv_heads + kv_head) * head_dim;
        float partial = (d < head_dim) ? (q_d * k_ptr[d]) : 0.0f;
        float wsum = scoria_wave_sum(partial);
        if (lane == 0) {
            s_dot[wave_id] = wsum;
        }
        __syncthreads();
        if (d == 0) {
            float total = 0.0f;
            for (unsigned w = 0; w < n_waves; ++w) {
                total += s_dot[w];
            }
            s_score = total * scale;
        }
        __syncthreads();

        float score = s_score;
        float m_new = fmaxf(m_i, score);
        float alpha = (l_i == 0.0f) ? 0.0f : expf(m_i - m_new);
        float beta = expf(score - m_new);
        l_i = l_i * alpha + beta;

        const float* v_ptr = v_cache + ((size_t)t * n_kv_heads + kv_head) * head_dim;
        float v_d = (d < head_dim) ? v_ptr[d] : 0.0f;
        acc_v = acc_v * alpha + beta * v_d;
        m_i = m_new;
    }

    if (d < head_dim) {
        out[((size_t)tok * n_heads + q_head) * head_dim + d] = acc_v / l_i;
    }
}

extern "C" __global__ void silu_mul_f32(const float* __restrict__ gate,
                                        const float* __restrict__ up,
                                        float* __restrict__ out,
                                        unsigned n) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
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
