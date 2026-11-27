//! The whole `inkling2_loop` stack, host-side, one token position at a time.
//!
//! What this implements, transcribed from `modeling_inkling2_loop.py`:
//!
//! ```text
//! hidden = embed_norm(embed[token])              # weighted RMSNorm on the embedding row
//! block_states = []
//! for layer in 0..6:   hidden = layer(hidden, block_states)
//! pre_hidden, pre_blocks = hidden, block_states
//! for pass in 1..3:                              # the loop: layers 6..18, three times
//!     prev_hidden, prev_blocks = hidden, block_states[num_pre_blocks + 1..]
//!     hidden, block_states = pre_hidden, pre_blocks
//!     for i, layer in 6..18:
//!         messages = message_attention(prev_blocks, prev_hidden, i)   # 4 learned queries
//!         hidden = layer(hidden, block_states, messages)
//! for layer in 18..24: hidden = layer(hidden, block_states)
//! hidden = final_input(block_states, hidden)     # 1 learned query
//! logits = lm_head(norm(hidden))[..200058]; logits = tanh(logits / 30) * 30
//! ```
//!
//! Per layer, in the reference's order:
//!
//! ```text
//! block_residual = hidden
//! q/k/v inputs   = attention_input(block_states, hidden)   # 3 learned queries (full) or 1 (sliding)
//!                  then the message gates on later loop passes
//! q/k/v          = input norms (three for full layers, one shared for sliding)
//! q              = clip(scale * Wq x) [* log-scale]; k/v = conv -> clip / RMSNorm
//! attn           = per-head attention with relative bias and a sink, windowed on sliding layers
//! out            = Wo(attn * sigmoid(gate))
//! out            = post_attention_conv(out)
//! if layer % 6 == 0: block_states.push(block_residual); residual = out
//! else:              residual = block_residual + out
//! ffn_in         = mlp_input(block_states, residual), then its message gate
//! hidden         = residual + post_mlp_conv(moe(rmsnorm(ffn_in)))
//! ```
//!
//! Two things this file deliberately does *not* do: masks over a sequence (it runs one
//! position at a time, which is exact for causal attention) and any speed work. It exists
//! to produce a logit vector from real weights that can be checked against the reference.

use scoria_gguf::{GgufError, GgufFile, ModelConfig, TensorInfo};

use crate::inkling2::{Inkling2Params, Visit, schedule};
use crate::inkling2_attn::{
    decode_head, l2_clip, relative_bias, scaled_l2_clip, short_conv1d_4tap,
    short_conv1d_4tap_dynamic, unweighted_rmsnorm,
};
use crate::inkling2_head::softcap;
use crate::inkling2_moe::{Moe, RouterScore};

fn dequant_tensor(file: &GgufFile, name: &str) -> Result<Vec<f32>, GgufError> {
    let info = file
        .tensor(name)
        .ok_or_else(|| GgufError::MissingMetadata(format!("tensor `{name}`")))?;
    let n = info
        .n_elements()
        .ok_or_else(|| GgufError::Malformed(format!("tensor `{name}` shape overflows")))?;
    let mut out = vec![0f32; n];
    scoria_gguf::dequant::dequant_row(info.ggml_type, file.tensor_bytes(info)?, &mut out)?;
    Ok(out)
}

fn norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Weighted RMSNorm: `x * w / sqrt(mean(x^2) + eps)`.
fn rmsnorm(x: &mut [f32], w: &[f32], eps: f32) {
    assert_eq!(x.len(), w.len());
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for (v, w) in x.iter_mut().zip(w) {
        *v = *v * inv * w;
    }
}

/// `y[o] = sum_i x[i] * w[o][i]` with GGUF's `[in, out]` row layout.
fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), in_dim, "matvec input");
    assert_eq!(w.len(), out_dim * in_dim, "matvec weight");
    let mut y = vec![0f32; out_dim];
    for (o, slot) in y.iter_mut().enumerate() {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        *slot = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
    y
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One learned query (or several) attending over the block snapshots plus the current
/// hidden state, non-causally, scaled by `hidden_size^-0.5`.
fn block_attention(
    query: &[f32],
    queries: usize,
    blocks: &[Vec<f32>],
    partial: &[f32],
    eps: f32,
) -> Vec<Vec<f32>> {
    let n = partial.len();
    assert_eq!(query.len(), queries * n);
    let mut stacked: Vec<Vec<f32>> = blocks.to_vec();
    stacked.push(partial.to_vec());
    for v in stacked.iter_mut() {
        unweighted_rmsnorm(v, eps);
    }
    let scale = (n as f32).sqrt().recip();
    let mut out = vec![vec![0f32; n]; queries];
    for (qi, slot) in out.iter_mut().enumerate() {
        let q = &query[qi * n..(qi + 1) * n];
        let mut weights: Vec<f32> = stacked
            .iter()
            .map(|v| q.iter().zip(v).map(|(a, b)| a * b).sum::<f32>() * scale)
            .collect();
        softmax_in_place(&mut weights);
        for (w, v) in weights.iter().zip(&stacked) {
            for (o, vi) in slot.iter_mut().zip(v) {
                *o += w * vi;
            }
        }
    }
    out
}

fn softmax_in_place(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `Inkling2LoopMessageGate`: `(1 - tanh(forget(z))) * hidden + tanh(message(z)) * message`,
/// where `z = down([norm(hidden); norm(message)])`.
struct MessageGate {
    down: Vec<f32>,
    forget: Vec<f32>,
    message: Vec<f32>,
    rank: usize,
}

impl MessageGate {
    fn load(file: &GgufFile, layer: usize, which: &str) -> Result<Self, GgufError> {
        let prefix = format!("blk.{layer}.loop_{which}_ctrl");
        Ok(Self {
            down: dequant_tensor(file, &format!("{prefix}_down.weight"))?,
            forget: dequant_tensor(file, &format!("{prefix}_forget.weight"))?,
            message: dequant_tensor(file, &format!("{prefix}_gain.weight"))?,
            rank: file
                .tensor(&format!("{prefix}_down.weight"))
                .ok_or_else(|| GgufError::MissingMetadata(format!("{prefix}_down.weight")))?
                .dims[1] as usize,
        })
    }

    fn apply(&self, hidden: &[f32], message: &[f32], n_embd: usize, eps: f32) -> Vec<f32> {
        assert_eq!(hidden.len(), n_embd);
        assert_eq!(message.len(), n_embd);
        let mut h = hidden.to_vec();
        let mut m = message.to_vec();
        unweighted_rmsnorm(&mut h, eps);
        unweighted_rmsnorm(&mut m, eps);
        let mut cat = h;
        cat.extend_from_slice(&m);
        let z = matvec(&self.down, &cat, self.rank, 2 * n_embd);
        let forget = matvec(&self.forget, &z, n_embd, self.rank);
        let gain = matvec(&self.message, &z, n_embd, self.rank);
        (0..n_embd)
            .map(|i| (1.0 - forget[i].tanh()) * hidden[i] + gain[i].tanh() * message[i])
            .collect()
    }
}

/// Per-visit state: the four convolution histories and the KV cache. The reference gives
/// every loop pass its own cache layer, which `schedule` already models as own visits.
#[derive(Clone)]
struct VisitCache {
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    post_conv: Vec<f32>,
    ffn_conv: Vec<f32>,
    k_seq: Vec<f32>,
    v_seq: Vec<f32>,
    len: usize,
}

impl VisitCache {
    fn new(n_embd: usize, kv_dim: usize) -> Self {
        Self {
            k_conv: vec![0.0; 3 * kv_dim],
            v_conv: vec![0.0; 3 * kv_dim],
            post_conv: vec![0.0; 3 * n_embd],
            ffn_conv: vec![0.0; 3 * n_embd],
            k_seq: Vec::new(),
            v_seq: Vec::new(),
            len: 0,
        }
    }
}

pub struct Layer {
    pub idx: usize,
    pub sliding: bool,
    pub looped: bool,
    pub boundary: bool,
    /// Rows in `attn_res_query`: three (q, k, v) on full layers, one on sliding layers.
    pub queries: usize,
    rel_extent: usize,
    n_heads: usize,
    head_dim: usize,
    kv_heads: usize,
    kv_dim: usize,
    d_rel: usize,
    qk_clip: f32,
    rel_clip: f32,
    /// Sliding layers normalise q/k/v with one shared norm; full layers with three.
    input_norm: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    v_norm: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_gate: Vec<f32>,
    w_o: Vec<f32>,
    w_rel: Vec<f32>,
    w_sink: Vec<f32>,
    q_clip: Vec<f32>,
    rel_scale: Vec<f32>,
    rel_proj: Vec<f32>,
    conv_k: Vec<f32>,
    conv_v: Vec<f32>,
    /// Full layers convolve K/V with taps that depend on the token; sliding layers don't.
    conv_k_proj: Option<Vec<f32>>,
    conv_v_proj: Option<Vec<f32>>,
    post_conv: Vec<f32>,
    attn_res_query: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_post_conv: Vec<f32>,
    ffn_res_query: Vec<f32>,
    /// The four message gates and their shared query, on loop layers.
    gates: Option<[MessageGate; 4]>,
    msg_query: Vec<f32>,
}

impl Layer {
    fn load(
        file: &GgufFile,
        cfg: &ModelConfig,
        params: &Inkling2Params,
        idx: usize,
    ) -> Result<Self, GgufError> {
        let n_heads = cfg.n_heads;
        let head_dim = cfg.head_dim;
        let kv_heads = cfg.n_kv_heads;
        let kv_dim = kv_heads * head_dim;
        let sliding = cfg
            .sliding_window_pattern
            .as_ref()
            .and_then(|p| p.get(idx).copied())
            .is_none_or(|v| v != 0);
        let looped = matches!(
            (cfg.loop_count, cfg.loop_start, cfg.loop_end),
            (Some(p), Some(s), Some(e)) if p > 1 && idx >= s && idx < e
        );
        let t = |name: &str| format!("blk.{idx}.{name}");
        let has = |name: &str| file.has_tensor(&t(name));

        let queries = if sliding { 1 } else { 3 };
        let attn_res_query = if idx > 0 {
            dequant_tensor(file, &t("attn_res_query.weight"))?
        } else {
            Vec::new()
        };
        // A sliding layer's three message-gated streams all go through one norm; a full
        // layer has a separate norm per stream.
        let (input_norm, q_norm, k_norm, v_norm) = if sliding {
            let n = dequant_tensor(file, &t("attn_norm.weight"))?;
            (n, Vec::new(), Vec::new(), Vec::new())
        } else {
            (
                Vec::new(),
                dequant_tensor(file, &t("attn_q_inp_norm.weight"))?,
                dequant_tensor(file, &t("attn_k_inp_norm.weight"))?,
                dequant_tensor(file, &t("attn_v_inp_norm.weight"))?,
            )
        };
        let gates = if looped {
            Some([
                MessageGate::load(file, idx, "q")?,
                MessageGate::load(file, idx, "k")?,
                MessageGate::load(file, idx, "v")?,
                MessageGate::load(file, idx, "ffn")?,
            ])
        } else {
            None
        };
        Ok(Self {
            idx,
            sliding,
            looped,
            boundary: idx.is_multiple_of(params.attn_res_block.max(1)),
            queries,
            rel_extent: if sliding {
                cfg.sliding_window.unwrap_or(params.relative_extent)
            } else {
                params.relative_extent
            },
            n_heads,
            head_dim,
            kv_heads,
            kv_dim,
            d_rel: params.relative_length,
            qk_clip: params.qk_clip,
            rel_clip: params.relative_clip,
            input_norm,
            q_norm,
            k_norm,
            v_norm,
            w_q: dequant_tensor(file, &t("attn_q.weight"))?,
            w_k: dequant_tensor(file, &t("attn_k.weight"))?,
            w_v: dequant_tensor(file, &t("attn_v.weight"))?,
            w_gate: dequant_tensor(file, &t("attn_gate.weight"))?,
            w_o: dequant_tensor(file, &t("attn_output.weight"))?,
            w_rel: dequant_tensor(file, &t("attn_rel.weight"))?,
            w_sink: dequant_tensor(file, &t("attn_sink_proj.weight"))?,
            q_clip: dequant_tensor(file, &t("attn_q_clip.weight"))?,
            rel_scale: dequant_tensor(file, &t("attn_rel_clip.weight"))?,
            rel_proj: dequant_tensor(file, &t("attn_rel_bias.weight"))?,
            conv_k: dequant_tensor(file, &t("attn_k_conv.weight"))?,
            conv_v: dequant_tensor(file, &t("attn_v_conv.weight"))?,
            conv_k_proj: if has("attn_k_conv_proj.weight") {
                Some(dequant_tensor(file, &t("attn_k_conv_proj.weight"))?)
            } else {
                None
            },
            conv_v_proj: if has("attn_v_conv_proj.weight") {
                Some(dequant_tensor(file, &t("attn_v_conv_proj.weight"))?)
            } else {
                None
            },
            post_conv: dequant_tensor(file, &t("attn_post_conv.weight"))?,
            attn_res_query,
            ffn_norm: dequant_tensor(file, &t("ffn_norm.weight"))?,
            ffn_post_conv: dequant_tensor(file, &t("ffn_post_conv.weight"))?,
            ffn_res_query: dequant_tensor(file, &t("ffn_res_query.weight"))?,
            gates,
            msg_query: if looped {
                dequant_tensor(file, &t("loop_msg_query.weight"))?
            } else {
                Vec::new()
            },
        })
    }
}

/// The output head: `output_res_query` over the final block states, `output_norm`, the
/// unembedding and `final_logit_softcapping`.
pub struct Head {
    n_embd: usize,
    eps: f32,
    cap: f32,
    vocab: usize,
    query: Vec<f32>,
    norm: Vec<f32>,
    info: TensorInfo,
    row_bytes: usize,
}

impl Head {
    fn load(
        file: &GgufFile,
        cfg: &ModelConfig,
        params: &Inkling2Params,
    ) -> Result<Self, GgufError> {
        let info = file
            .tensor("output.weight")
            .ok_or_else(|| GgufError::MissingMetadata("output.weight".into()))?
            .clone();
        let rows = info.n_elements().unwrap_or(0) / cfg.n_embd.max(1);
        let vocab_unpadded = file
            .meta_u32("inkling2_loop.vocab_size_unpadded")
            .map(|v| v as usize)
            .unwrap_or(rows);
        Ok(Self {
            n_embd: cfg.n_embd,
            eps: cfg.rms_eps,
            cap: params.final_logit_softcap,
            vocab: vocab_unpadded.min(rows),
            query: dequant_tensor(file, "output_res_query.weight")?,
            norm: dequant_tensor(file, "output_norm.weight")?,
            row_bytes: info
                .ggml_type
                .tensor_byte_len(cfg.n_embd)
                .map_err(|_| GgufError::Malformed("output row size".into()))?,
            info,
        })
    }

    /// `logits[v] = softcap(normed . W[v])` for the unpadded vocabulary.
    pub fn logits(&self, file: &GgufFile, hidden: &[f32]) -> Vec<f32> {
        let bytes = file
            .tensor_bytes(&self.info)
            .expect("output tensor is in the mmap");
        let mut out = Vec::with_capacity(self.vocab);
        let mut row = vec![0f32; self.n_embd];
        for v in 0..self.vocab {
            let start = v * self.row_bytes;
            scoria_gguf::dequant::dequant_row(
                self.info.ggml_type,
                &bytes[start..start + self.row_bytes],
                &mut row,
            )
            .expect("unembedding row dequantises");
            let dot = row.iter().zip(hidden).map(|(a, b)| a * b).sum::<f32>();
            out.push(softcap(dot, self.cap));
        }
        out
    }
}

pub struct Model {
    file: GgufFile,
    pub cfg: ModelConfig,
    params: Inkling2Params,
    layers: Vec<Layer>,
    moes: Vec<Moe>,
    caches: Vec<VisitCache>,
    visits: Vec<Visit>,
    head: Head,
    embd: TensorInfo,
    embd_row_bytes: usize,
    embd_norm: Vec<f32>,
    block_states: Vec<Vec<f32>>,
    /// Token position of the next forward, i.e. how much history the caches hold.
    pub pos: usize,
    traces: Vec<VisitTrace>,
    /// Stop after this many visits and run the head there anyway. Only for depth sweeps:
    /// a truncated model is not a model, but where perplexity stops improving localises a
    /// broken layer.
    pub max_visits: usize,
    pub last_report: Option<StepReport>,
}

/// Diagnostics for the last layer visited, so a wrong-looking logit can be traced back to
/// the layer that produced it.
/// Attention diagnostics for one visit, head 0: the peak softmax weight, how far back it
/// looks, and the mass that is not the sink. A model that ignores its history shows up
/// here as a tiny peak and a peak at distance 0.
#[derive(Debug, Clone, Copy, Default)]
pub struct VisitTrace {
    pub layer: usize,
    pub peak: f32,
    pub peak_distance: usize,
    pub sink: f32,
    pub entropy: f32,
}

#[derive(Debug, Clone, Default)]
pub struct StepReport {
    pub hidden_norm: f32,
    pub block_snapshots: usize,
    /// `(layer, |out|)` for every visit, in order.
    pub per_visit: Vec<(usize, f32)>,
    pub attention: Vec<VisitTrace>,
}

impl Model {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, GgufError> {
        Self::open_with_router(path, RouterScore::Sigmoid)
    }

    /// The checkpoint does not name the router activation. Sigmoid is the default because
    /// `exp_probs_b.bias` is a noaux-tc feature, but the choice is testable: with sigmoid
    /// all 128 scores sit in [0, 1] and the top-8 weights come out nearly uniform, while
    /// softmax differentiates them.
    pub fn open_with_router(
        path: impl AsRef<std::path::Path>,
        score: RouterScore,
    ) -> Result<Self, GgufError> {
        let file = GgufFile::open(path)?;
        let cfg = ModelConfig::from_gguf(&file)?;
        Self::from_file_with_router(file, cfg, score)
    }

    pub fn from_file(file: GgufFile, cfg: ModelConfig) -> Result<Self, GgufError> {
        Self::from_file_with_router(file, cfg, RouterScore::Sigmoid)
    }

    pub fn from_file_with_router(
        file: GgufFile,
        cfg: ModelConfig,
        score: RouterScore,
    ) -> Result<Self, GgufError> {
        let params = Inkling2Params::from_gguf(&file, &cfg);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut moes = Vec::with_capacity(cfg.n_layers);
        for idx in 0..cfg.n_layers {
            layers.push(Layer::load(&file, &cfg, &params, idx)?);
            moes.push(Moe::load_layer(&file, &cfg, score, idx)?);
        }
        let visits = schedule(&cfg);
        let caches = (0..visits.len())
            .map(|_| VisitCache::new(cfg.n_embd, cfg.n_kv_heads * cfg.head_dim))
            .collect();
        let embd = file
            .tensor("token_embd.weight")
            .ok_or_else(|| GgufError::MissingMetadata("token_embd.weight".into()))?
            .clone();
        let row_bytes = embd
            .ggml_type
            .tensor_byte_len(cfg.n_embd)
            .map_err(|_| GgufError::Malformed("token_embd row size".into()))?;
        let head = Head::load(&file, &cfg, &params)?;
        Ok(Self {
            embd_norm: dequant_tensor(&file, "token_embd_norm.weight")?,
            head,
            embd_row_bytes: row_bytes,
            embd,
            block_states: Vec::new(),
            caches,
            visits,
            layers,
            moes,
            pos: 0,
            traces: Vec::new(),
            max_visits: usize::MAX,
            params,
            cfg,
            file,
            last_report: None,
        })
    }

    pub fn reset(&mut self) {
        for c in self.caches.iter_mut() {
            *c = VisitCache::new(self.cfg.n_embd, self.cfg.n_kv_heads * self.cfg.head_dim);
        }
        self.block_states.clear();
        self.pos = 0;
        self.last_report = None;
    }

    pub fn visits(&self) -> &[Visit] {
        &self.visits
    }

    /// Embedding + weighted RMSNorm, the reference's `Inkling2LoopNormedEmbedding`.
    fn embed(&self, token: u32) -> Vec<f32> {
        let start = token as usize * self.embd_row_bytes;
        let bytes = self.file.tensor_bytes(&self.embd).expect("embd in mmap");
        let mut h = vec![0f32; self.cfg.n_embd];
        scoria_gguf::dequant::dequant_row(
            self.embd.ggml_type,
            &bytes[start..start + self.embd_row_bytes],
            &mut h,
        )
        .expect("embedding row dequantises");
        rmsnorm(&mut h, &self.embd_norm, self.cfg.rms_eps);
        h
    }

    /// One token position through all visits, returning the softcapped logits.
    pub fn forward(&mut self, token: u32) -> Vec<f32> {
        self.pos += 1;
        let mut h = self.embed(token);
        self.block_states.clear();
        self.traces.clear();
        let mut report = StepReport {
            hidden_norm: norm(&h),
            ..Default::default()
        };

        let looped = matches!(
            (self.cfg.loop_count, self.cfg.loop_start, self.cfg.loop_end),
            (Some(p), Some(s), Some(e)) if p > 1 && e > s
        );
        let (passes, start, end) = if looped {
            (
                self.cfg.loop_count.unwrap(),
                self.cfg.loop_start.unwrap(),
                self.cfg.loop_end.unwrap(),
            )
        } else {
            (1, 0, self.cfg.n_layers)
        };

        let budget = self.max_visits;
        let visit_of = |layer: usize, pass: usize| -> usize {
            if layer < start {
                layer
            } else if layer < end {
                start + pass * (end - start) + (layer - start)
            } else {
                start + passes * (end - start) + (layer - end)
            }
        };

        let mut done = 0usize;
        'stack: {
            for layer in 0..start {
                if done >= budget {
                    break 'stack;
                }
                h = self.layer_forward(layer, &h, None, visit_of(layer, 0));
                report.per_visit.push((layer, norm(&h)));
                done += 1;
            }

            let pre_hidden = h.clone();
            let pre_blocks = self.block_states.clone();
            let num_pre_blocks = pre_blocks.len();

            // Pass 0 is a plain run over the looped block. Passes 1.. get messages from
            // the pass before, and restart from the pre-loop state.
            for layer in start..end {
                if done >= budget {
                    break 'stack;
                }
                h = self.layer_forward(layer, &h, None, visit_of(layer, 0));
                report.per_visit.push((layer, norm(&h)));
                done += 1;
            }

            for pass in 1..passes {
                // The reference restarts from the pre-loop state and reads the previous pass's
                // *last* block snapshot plus its final hidden state as the message context.
                let prev_hidden = h.clone();
                let prev_blocks: Vec<Vec<f32>> = self
                    .block_states
                    .iter()
                    .skip((num_pre_blocks + 1).min(self.block_states.len()))
                    .cloned()
                    .collect();
                h = pre_hidden.clone();
                self.block_states = pre_blocks.clone();
                for layer in start..end {
                    if done >= budget {
                        break 'stack;
                    }
                    let i = layer - start;
                    let msg = self.messages(layer, &prev_blocks, &prev_hidden, i);
                    h = self.layer_forward(layer, &h, Some(&msg), visit_of(layer, pass));
                    report.per_visit.push((layer, norm(&h)));
                    done += 1;
                }
            }

            for layer in end..self.cfg.n_layers {
                if done >= budget {
                    break 'stack;
                }
                h = self.layer_forward(layer, &h, None, visit_of(layer, 0));
                report.per_visit.push((layer, norm(&h)));
                done += 1;
            }
        }

        // `final_input`: one learned query over the block snapshots plus the last hidden.
        let final_hidden = block_attention(
            &self.head.query,
            1,
            &self.block_states,
            &h,
            self.cfg.rms_eps,
        )
        .remove(0);
        let mut normed = final_hidden.clone();
        rmsnorm(&mut normed, &self.head.norm, self.head.eps);
        report.hidden_norm = norm(&normed);
        report.block_snapshots = self.block_states.len();
        report.attention = self.traces.clone();
        self.last_report = Some(report);
        self.head.logits(&self.file, &normed)
    }

    /// The four message queries for one loop layer: `[q, k, v, mlp]`, each a single
    /// non-causal attention over the previous pass's block snapshot(s) and hidden state.
    fn messages(
        &self,
        layer: usize,
        prev_blocks: &[Vec<f32>],
        prev_hidden: &[f32],
        _loop_idx: usize,
    ) -> Vec<Vec<f32>> {
        block_attention(
            &self.layers[layer].msg_query,
            4,
            prev_blocks,
            prev_hidden,
            self.cfg.rms_eps,
        )
    }

    fn layer_forward(
        &mut self,
        layer: usize,
        h: &[f32],
        msg: Option<&[Vec<f32>]>,
        visit: usize,
    ) -> Vec<f32> {
        let cfg = &self.cfg;
        let eps = cfg.rms_eps;
        let n_embd = cfg.n_embd;
        let lw = &self.layers[layer];
        let cache = &mut self.caches[visit];
        let moe = &mut self.moes[layer];
        let n_heads = lw.n_heads;
        let head_dim = lw.head_dim;
        let kv_dim = lw.kv_dim;
        let group = n_heads / lw.kv_heads;

        let block_residual = h.to_vec();

        // `attention_input` yields the q/k/v streams; layer 0 has none and passes hidden
        // through unchanged.
        let mut streams: Vec<Vec<f32>> = if lw.idx == 0 {
            vec![h.to_vec(); lw.queries]
        } else {
            block_attention(&lw.attn_res_query, lw.queries, &self.block_states, h, eps)
        };
        let (mut q_in, mut k_in, mut v_in) = if lw.queries == 3 {
            (streams.remove(0), streams.remove(0), streams.remove(0))
        } else {
            let s = streams.remove(0);
            (s.clone(), s.clone(), s)
        };
        if let (Some(msg), Some(gates)) = (msg, lw.gates.as_ref()) {
            q_in = gates[0].apply(&q_in, &msg[0], n_embd, eps);
            k_in = gates[1].apply(&k_in, &msg[1], n_embd, eps);
            v_in = gates[2].apply(&v_in, &msg[2], n_embd, eps);
        }

        // Sliding layers share one input norm; full layers normalise each stream.
        let (qn, kn, vn) = if lw.sliding {
            let qn = {
                let mut t = q_in.clone();
                rmsnorm(&mut t, &lw.input_norm, eps);
                t
            };
            let kn = {
                let mut t = k_in.clone();
                rmsnorm(&mut t, &lw.input_norm, eps);
                t
            };
            let vn = {
                let mut t = v_in.clone();
                rmsnorm(&mut t, &lw.input_norm, eps);
                t
            };
            (qn, kn, vn)
        } else {
            let mut qn = q_in.clone();
            let mut kn = k_in.clone();
            let mut vn = v_in.clone();
            rmsnorm(&mut qn, &lw.q_norm, eps);
            rmsnorm(&mut kn, &lw.k_norm, eps);
            rmsnorm(&mut vn, &lw.v_norm, eps);
            (qn, kn, vn)
        };

        // Log-scaled positions (full layers only): tau = 1 + alpha * ln(max((pos+1)/floor, 1)).
        let tau = if lw.sliding {
            1.0
        } else {
            let x = (self.pos as f32) / self.params.log_scale_floor.max(1) as f32;
            1.0 + self.params.log_scale_alpha * x.max(1.0).ln()
        };

        let mut q = matvec(&lw.w_q, &qn, n_heads * head_dim, n_embd);
        for hh in 0..n_heads {
            scaled_l2_clip(
                &mut q[hh * head_dim..(hh + 1) * head_dim],
                &lw.q_clip[hh * head_dim..(hh + 1) * head_dim],
                lw.qk_clip,
            );
        }
        if tau != 1.0 {
            for v in q.iter_mut() {
                *v *= tau;
            }
        }

        let mut k = matvec(&lw.w_k, &kn, kv_dim, n_embd);
        let mut v = matvec(&lw.w_v, &vn, kv_dim, n_embd);
        match (&lw.conv_k_proj, &lw.conv_v_proj) {
            (Some(pk), Some(pv)) => {
                let dyn_k = matvec(pk, &kn, lw.kv_heads * 4, n_embd);
                let dyn_v = matvec(pv, &vn, lw.kv_heads * 4, n_embd);
                short_conv1d_4tap_dynamic(
                    &mut k,
                    &lw.conv_k,
                    &dyn_k,
                    &mut cache.k_conv,
                    kv_dim,
                    head_dim,
                );
                short_conv1d_4tap_dynamic(
                    &mut v,
                    &lw.conv_v,
                    &dyn_v,
                    &mut cache.v_conv,
                    kv_dim,
                    head_dim,
                );
            }
            _ => {
                short_conv1d_4tap(&mut k, &lw.conv_k, &mut cache.k_conv, kv_dim);
                short_conv1d_4tap(&mut v, &lw.conv_v, &mut cache.v_conv, kv_dim);
            }
        }
        for hh in 0..lw.kv_heads {
            l2_clip(&mut k[hh * head_dim..(hh + 1) * head_dim], lw.qk_clip);
            unweighted_rmsnorm(&mut v[hh * head_dim..(hh + 1) * head_dim], eps);
        }

        let mut gate = matvec(&lw.w_gate, &qn, n_heads * head_dim, n_embd);
        for g in gate.iter_mut() {
            *g = sigmoid(*g);
        }
        let mut rel = matvec(&lw.w_rel, &qn, n_heads * lw.d_rel, n_embd);
        for hh in 0..n_heads {
            scaled_l2_clip(
                &mut rel[hh * lw.d_rel..(hh + 1) * lw.d_rel],
                &lw.rel_scale[hh * lw.d_rel..(hh + 1) * lw.d_rel],
                lw.rel_clip,
            );
        }
        if tau != 1.0 {
            for v in rel.iter_mut() {
                *v *= tau;
            }
        }
        let sink = matvec(&lw.w_sink, &qn, n_heads, n_embd);

        cache.k_seq.extend_from_slice(&k);
        cache.v_seq.extend_from_slice(&v);
        let kv_pos = cache.len;
        cache.len += 1;

        let pos_k: Vec<usize> = (0..=kv_pos).collect();
        let mut bias = vec![0f32; n_heads * (kv_pos + 1)];
        relative_bias(
            &rel,
            n_heads,
            lw.d_rel,
            &lw.rel_proj,
            lw.rel_extent,
            kv_pos,
            &pos_k,
            &mut bias,
        );

        let window = if lw.sliding {
            Some(cfg.sliding_window.unwrap_or(lw.rel_extent))
        } else {
            None
        };
        let mut attn = vec![0f32; n_heads * head_dim];
        for hh in 0..n_heads {
            let kvh = hh / group;
            let mut kh = Vec::with_capacity((kv_pos + 1) * head_dim);
            let mut vh = Vec::with_capacity((kv_pos + 1) * head_dim);
            for t in 0..=kv_pos {
                let base = (t * lw.kv_heads + kvh) * head_dim;
                kh.extend_from_slice(&cache.k_seq[base..base + head_dim]);
                vh.extend_from_slice(&cache.v_seq[base..base + head_dim]);
            }
            let out = decode_head(
                &q[hh * head_dim..(hh + 1) * head_dim],
                &kh,
                &vh,
                &bias[hh * (kv_pos + 1)..(hh + 1) * (kv_pos + 1)],
                sink[hh],
                kv_pos,
                window,
            );
            attn[hh * head_dim..(hh + 1) * head_dim].copy_from_slice(&out);
        }

        // Diagnostics: head 0's attention distribution including the sink.
        {
            let mut logits = Vec::with_capacity(kv_pos + 2);
            let scale = (head_dim as f32).sqrt().recip();
            for (t, bias_t) in bias.iter().enumerate().take(kv_pos + 1) {
                let visible = window.is_none_or(|w| kv_pos - t < w);
                if !visible {
                    logits.push(f32::NEG_INFINITY);
                    continue;
                }
                let base = t * lw.kv_heads * head_dim;
                let dot: f32 = (0..head_dim).map(|d| q[d] * cache.k_seq[base + d]).sum();
                logits.push(dot * scale + bias_t);
            }
            let mut with_sink = logits.clone();
            with_sink.push(sink[0]);
            let max = with_sink.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = with_sink.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|v| v / sum).collect();
            let (peak_idx, peak) = probs[..=kv_pos].iter().enumerate().fold(
                (0usize, f32::NEG_INFINITY),
                |(bi, bv), (i, v)| {
                    if *v > bv { (i, *v) } else { (bi, bv) }
                },
            );
            let entropy: f32 = probs
                .iter()
                .filter(|p| **p > 0.0)
                .map(|p| -p * p.ln())
                .sum();
            self.traces.push(VisitTrace {
                layer,
                peak,
                peak_distance: kv_pos - peak_idx,
                sink: probs[kv_pos + 1],
                entropy,
            });
        }

        let gated: Vec<f32> = attn.iter().zip(&gate).map(|(a, g)| a * g).collect();
        let mut out = matvec(&lw.w_o, &gated, n_embd, n_heads * head_dim);
        short_conv1d_4tap(&mut out, &lw.post_conv, &mut cache.post_conv, n_embd);

        let residual = if lw.boundary {
            self.block_states.push(block_residual);
            out
        } else {
            h.iter().zip(&out).map(|(a, b)| a + b).collect()
        };

        let mut ffn_in =
            block_attention(&lw.ffn_res_query, 1, &self.block_states, &residual, eps).remove(0);
        if let (Some(msg), Some(gates)) = (msg, lw.gates.as_ref()) {
            ffn_in = gates[3].apply(&ffn_in, &msg[3], n_embd, eps);
        }
        let mut moe_in = ffn_in;
        rmsnorm(&mut moe_in, &lw.ffn_norm, eps);
        let mut y = moe.forward(&self.file, &moe_in);
        short_conv1d_4tap(&mut y, &lw.ffn_post_conv, &mut cache.ffn_conv, n_embd);

        residual.iter().zip(&y).map(|(a, b)| a + b).collect()
    }

    /// The top-`k` `(token, logit)` pairs, highest first.
    pub fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        idx.into_iter().take(k).map(|i| (i, logits[i])).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_attention_with_three_queries_returns_three_convex_combinations() {
        let eps = 1e-6;
        let blocks = vec![vec![1.0f32, 0.0], vec![0.0, 2.0]];
        let partial = vec![1.0f32, 1.0];
        // Three queries: one favouring each block's direction and one neutral.
        let query = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let out = block_attention(&query, 3, &blocks, &partial, eps);
        assert_eq!(out.len(), 3);
        for row in &out {
            assert_eq!(row.len(), 2);
            assert!(row.iter().all(|v| v.is_finite()));
        }
        // Query 0 projects onto the first axis, query 1 onto the second: each must
        // emphasise its own direction.
        assert!(out[0][0].abs() > out[0][1].abs(), "{:?}", out[0]);
        assert!(out[1][1].abs() > out[1][0].abs(), "{:?}", out[1]);
        // The neutral query averages, so it lies between the two extremes.
        assert_ne!(out[0], out[2]);
    }

    #[test]
    fn message_gate_is_a_convex_mix_when_the_gates_saturate() {
        // A gate whose forget and message projections are zero gives tanh(0) = 0 for both,
        // so the output is exactly the hidden state: (1-0)*hidden + 0*message.
        let n = 2;
        let gate = MessageGate {
            down: vec![0.0; 2 * 2],
            forget: vec![0.0; n],
            message: vec![0.0; n],
            rank: 1,
        };
        let hidden = vec![3.0f32, -4.0];
        let message = vec![100.0f32, 100.0];
        let out = gate.apply(&hidden, &message, n, 1e-6);
        assert_eq!(out, hidden);
        // `down` must be non-zero for the gates to see anything: a zero down projection
        // makes z = 0, so both tanh terms are zero and the output is the hidden state no
        // matter how large the gate weights are (which is what the first case relies on).
        let gate = MessageGate {
            down: vec![1.0; 2 * n],
            forget: vec![1e9; n],
            message: vec![1e9; n],
            rank: 1,
        };
        let out = gate.apply(&hidden, &message, n, 1e-6);
        for (o, m) in out.iter().zip(&message) {
            assert!((*o - m).abs() < 1e-4, "{out:?} vs {message:?}");
        }
    }

    #[test]
    fn top_k_breaks_ties_by_index_and_is_ordered() {
        let logits = vec![1.0f32, 3.0, 3.0, -1.0];
        let top = Model::top_k(&logits, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0], (1, 3.0));
        assert_eq!(top[1], (2, 3.0), "ties resolve to the lower id");
        assert_eq!(top[2], (0, 1.0));
    }
}
