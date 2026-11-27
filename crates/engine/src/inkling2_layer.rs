//! Layer 0 of an `inkling2_loop` checkpoint, end to end on the host.
//!
//! This is the first half of a forward pass: token embedding -> attention RMSNorm ->
//! the six projections -> the verified `post_qkv` (conv / clip / RMSNorm / gate) ->
//! relative bias -> the verified decode attention -> output gate -> `W_o` -> the post
//! convolution -> the residual. The MoE FFN, the loop gates and the block attention
//! residuals are not here yet, so this produces one *layer's* attention contribution, not
//! a logit.
//!
//! Everything runs on dequantised f32 weights from the GGUF, through the same helpers the
//! GPU kernels are verified against, so a transposed projection or a misread dimension
//! shows up as garbage activations immediately.

use scoria_gguf::{GgufError, GgufFile, ModelConfig};

use crate::inkling2::Inkling2Params;
use crate::inkling2_attn::{
    decode_head, post_qkv, relative_bias, scaled_l2_clip, short_conv1d_4tap,
};

pub struct LayerZero {
    n_embd: usize,
    n_heads: usize,
    head_dim: usize,
    kv_heads: usize,
    kv_dim: usize,
    qk_clip: f32,
    eps: f32,
    rel_extent: usize,
    d_rel: usize,
    rel_clip: f32,
    /// Raw embedding table bytes: the whole table as f32 would be 1.85 GB, so rows are
    /// dequantised on demand (one row is 9 Q5_K blocks).
    token_embd: Vec<u8>,
    token_embd_type: scoria_gguf::GgmlType,
    token_embd_row_bytes: usize,
    attn_norm: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_gate: Vec<f32>,
    w_rel: Vec<f32>,
    w_sink: Vec<f32>,
    w_o: Vec<f32>,
    scale_q: Vec<f32>,
    scale_rel: Vec<f32>,
    proj_rel: Vec<f32>,
    conv_k: Vec<f32>,
    conv_v: Vec<f32>,
    post_conv: Vec<f32>,
    caches: Caches,
}

#[derive(Default, Clone)]
pub struct Caches {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub post: Vec<f32>,
    /// `[kv_len][kv_heads][head_dim]` for K and V.
    pub k_seq: Vec<f32>,
    pub v_seq: Vec<f32>,
    pub len: usize,
}

pub struct LayerZeroOutput {
    pub hidden_norm: f32,
    pub q_norm: f32,
    pub k_norm: f32,
    pub v_norm: f32,
    pub gate_mean: f32,
    pub sink: Vec<f32>,
    pub bias_at_zero: Vec<f32>,
    pub attn_norm: f32,
    pub post_conv_norm: f32,
    pub residual_norm: f32,
    pub head0_q: Vec<f32>,
    pub head0_attn: Vec<f32>,
}

fn dequant_tensor(file: &GgufFile, name: &str) -> Result<Vec<f32>, GgufError> {
    let info = file
        .tensor(name)
        .ok_or_else(|| GgufError::MissingMetadata(format!("tensor `{name}`")))?;
    let n = info
        .n_elements()
        .ok_or_else(|| GgufError::Malformed(format!("tensor `{name}` shape overflows")))?;
    let bytes = file.tensor_bytes(info)?;
    let mut out = vec![0f32; n];
    scoria_gguf::dequant::dequant_row(info.ggml_type, bytes, &mut out)?;
    Ok(out)
}

fn norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn rmsnorm(x: &mut [f32], w: &[f32], eps: f32) {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for (v, w) in x.iter_mut().zip(w) {
        *v = *v * inv * w;
    }
}

/// `y[o] = sum_i x[i] * w[o][i]`, with GGUF's `[in, out]` layout.
fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), in_dim);
    assert_eq!(w.len(), out_dim * in_dim);
    let mut y = vec![0f32; out_dim];
    for (o, slot) in y.iter_mut().enumerate() {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        *slot = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
    y
}

impl LayerZero {
    pub fn load(file: &GgufFile, cfg: &ModelConfig) -> Result<Self, GgufError> {
        let params = Inkling2Params::from_gguf(file, cfg);
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim;
        let kv_heads = cfg.n_kv_heads;
        Ok(Self {
            n_embd: cfg.n_embd,
            n_heads,
            head_dim,
            kv_heads,
            kv_dim: kv_heads * head_dim,
            qk_clip: params.qk_clip,
            eps: cfg.rms_eps,
            rel_extent: params.relative_extent,
            d_rel: params.relative_length,
            rel_clip: params.relative_clip,
            token_embd: file
                .tensor_bytes(
                    file.tensor("token_embd.weight")
                        .ok_or_else(|| GgufError::MissingMetadata("token_embd.weight".into()))?,
                )?
                .to_vec(),
            token_embd_type: file
                .tensor("token_embd.weight")
                .map(|t| t.ggml_type)
                .unwrap_or(scoria_gguf::GgmlType::F32),
            token_embd_row_bytes: {
                let t = file
                    .tensor("token_embd.weight")
                    .ok_or_else(|| GgufError::MissingMetadata("token_embd.weight".into()))?;
                t.ggml_type
                    .tensor_byte_len(cfg.n_embd)
                    .map_err(|_| GgufError::Malformed("token_embd row size".into()))?
            },
            attn_norm: dequant_tensor(file, "blk.0.attn_norm.weight")?,
            w_q: dequant_tensor(file, "blk.0.attn_q.weight")?,
            w_k: dequant_tensor(file, "blk.0.attn_k.weight")?,
            w_v: dequant_tensor(file, "blk.0.attn_v.weight")?,
            w_gate: dequant_tensor(file, "blk.0.attn_gate.weight")?,
            w_rel: dequant_tensor(file, "blk.0.attn_rel.weight")?,
            w_sink: dequant_tensor(file, "blk.0.attn_sink_proj.weight")?,
            w_o: dequant_tensor(file, "blk.0.attn_output.weight")?,
            scale_q: dequant_tensor(file, "blk.0.attn_q_clip.weight")?,
            scale_rel: dequant_tensor(file, "blk.0.attn_rel_clip.weight")?,
            proj_rel: dequant_tensor(file, "blk.0.attn_rel_bias.weight")?,
            conv_k: dequant_tensor(file, "blk.0.attn_k_conv.weight")?,
            conv_v: dequant_tensor(file, "blk.0.attn_v_conv.weight")?,
            post_conv: dequant_tensor(file, "blk.0.attn_post_conv.weight")?,
            caches: Caches {
                k: vec![0.0; 3 * kv_heads * head_dim],
                v: vec![0.0; 3 * kv_heads * head_dim],
                post: vec![0.0; 3 * cfg.n_embd],
                k_seq: Vec::new(),
                v_seq: Vec::new(),
                len: 0,
            },
        })
    }

    pub fn reset(&mut self) {
        self.caches = Caches {
            k: vec![0.0; 3 * self.kv_dim],
            v: vec![0.0; 3 * self.kv_dim],
            post: vec![0.0; 3 * self.n_embd],
            k_seq: Vec::new(),
            v_seq: Vec::new(),
            len: 0,
        };
    }

    /// One token through layer 0's attention. Returns diagnostics, not a logit: the FFN
    /// and the rest of the stack are not in this function.
    pub fn attention_step(&mut self, token: u32) -> LayerZeroOutput {
        // Dequantise just this token's row.
        let start = token as usize * self.token_embd_row_bytes;
        let mut accumulated = vec![0f32; self.n_embd];
        scoria_gguf::dequant::dequant_row(
            self.token_embd_type,
            &self.token_embd[start..start + self.token_embd_row_bytes],
            &mut accumulated,
        )
        .expect("embedding row dequantises");
        let row = accumulated.as_slice();
        let hidden = row.to_vec();

        let mut normed = hidden.clone();
        rmsnorm(&mut normed, &self.attn_norm, self.eps);

        // Raw projections: [n_heads*head_dim] for q and the gate, [kv_dim] for k and v.
        let mut q = matvec(
            &self.w_q,
            &normed,
            self.n_heads * self.head_dim,
            self.n_embd,
        );
        let mut k = matvec(&self.w_k, &normed, self.kv_dim, self.n_embd);
        let mut v = matvec(&self.w_v, &normed, self.kv_dim, self.n_embd);
        let mut gate = matvec(
            &self.w_gate,
            &normed,
            self.n_heads * self.head_dim,
            self.n_embd,
        );
        let mut rel_states = matvec(&self.w_rel, &normed, self.n_heads * self.d_rel, self.n_embd);
        let sink = matvec(&self.w_sink, &normed, self.n_heads, self.n_embd);

        // Relative state clipping uses the same per-head scaled clip as Q.
        for h in 0..self.n_heads {
            scaled_l2_clip(
                &mut rel_states[h * self.d_rel..(h + 1) * self.d_rel],
                &self.scale_rel[h * self.d_rel..(h + 1) * self.d_rel],
                self.rel_clip,
            );
        }

        post_qkv(
            &mut q,
            &mut k,
            &mut v,
            &mut gate,
            &self.scale_q,
            &self.conv_k,
            &self.conv_v,
            &mut self.caches.k,
            &mut self.caches.v,
            self.n_heads,
            self.head_dim,
            self.kv_dim,
            self.qk_clip,
            self.eps,
        );

        self.caches.k_seq.extend_from_slice(&k);
        self.caches.v_seq.extend_from_slice(&v);
        let pos = self.caches.len;
        self.caches.len += 1;

        // Relative bias row for this query against every cached position.
        let pos_k: Vec<usize> = (0..=pos).collect();
        let mut bias = vec![0f32; self.n_heads * (pos + 1)];
        relative_bias(
            &rel_states,
            self.n_heads,
            self.d_rel,
            &self.proj_rel,
            self.rel_extent,
            pos,
            &pos_k,
            &mut bias,
        );

        // Attend per head, narrow the cache to this head's width, then gate and project.
        let group = self.n_heads / self.kv_heads;
        let mut attn = vec![0f32; self.n_heads * self.head_dim];
        for h in 0..self.n_heads {
            let kvh = h / group;
            let mut kh = vec![0f32; (pos + 1) * self.head_dim];
            let mut vh = vec![0f32; (pos + 1) * self.head_dim];
            for t in 0..=pos {
                kh[t * self.head_dim..(t + 1) * self.head_dim].copy_from_slice(
                    &self.caches.k_seq[(t * self.kv_heads + kvh) * self.head_dim..]
                        [..self.head_dim],
                );
                vh[t * self.head_dim..(t + 1) * self.head_dim].copy_from_slice(
                    &self.caches.v_seq[(t * self.kv_heads + kvh) * self.head_dim..]
                        [..self.head_dim],
                );
            }
            let bias_row = &bias[h * (pos + 1)..(h + 1) * (pos + 1)];
            let out = decode_head(
                &q[h * self.head_dim..(h + 1) * self.head_dim],
                &kh,
                &vh,
                bias_row,
                sink[h],
                pos,
                Some(1024),
            );
            attn[h * self.head_dim..(h + 1) * self.head_dim].copy_from_slice(&out);
        }

        let gated: Vec<f32> = attn.iter().zip(&gate).map(|(a, g)| a * g).collect();
        let projected = matvec(&self.w_o, &gated, self.n_embd, self.n_heads * self.head_dim);

        // Post convolution, then the residual add.
        let mut post_out = projected;
        short_conv1d_4tap(
            &mut post_out,
            &self.post_conv,
            &mut self.caches.post,
            self.n_embd,
        );
        let residual: Vec<f32> = row.iter().zip(&post_out).map(|(a, b)| a + b).collect();

        LayerZeroOutput {
            hidden_norm: norm(&hidden),
            q_norm: norm(&q),
            k_norm: norm(&k),
            v_norm: norm(&v),
            gate_mean: gate.iter().sum::<f32>() / gate.len() as f32,
            sink,
            bias_at_zero: (0..self.n_heads)
                .map(|h| bias[h * (pos + 1) + pos])
                .collect(),
            attn_norm: norm(&attn),
            post_conv_norm: norm(&post_out),
            residual_norm: norm(&residual),
            head0_q: q[..self.head_dim].to_vec(),
            head0_attn: attn[..self.head_dim].to_vec(),
        }
    }
}
