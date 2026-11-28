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

use std::ffi::c_void;

use scoria_gguf::{GgmlType, GgufError, GgufFile, ModelConfig, TensorInfo};
use scoria_hip_rt::{DeviceBuf, HipStream};

use crate::EngineContext;

use crate::inkling2::{Inkling2Params, Visit, schedule};
use crate::inkling2_attn::{
    decode_head_in_cache, l2_clip, relative_bias, scaled_l2_clip, short_conv1d_4tap,
    short_conv1d_4tap_dynamic, unweighted_rmsnorm,
};
use crate::inkling2_head::softcap;
use crate::inkling2_moe::{Moe, RouterScore, swiglu_softcapped as swiglu_softcapped_public};

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
/// One visit's KV history.
///
/// A sliding-window visit only ever attends to the last `window` positions, so keeping more
/// is pure waste: at 48 visits x 2 tensors x 1024 values x 4 bytes it is 393 KB per token,
/// i.e. 78 GB at 200k positions. Windowed visits therefore store a **ring** of `window`
/// slots: a position is written once and overwritten when it falls out of the window, so
/// memory is bounded by the window rather than by the sequence.
///
/// Full-history visits have no such bound -- dense attention in those 8 visits reads every
/// earlier position every token, so eviction there is not a policy question, it is a
/// correctness question. They keep everything.
///
/// The ring keeps absolute position numbers (`count`), so relative distances — and hence
/// the relative-bias bank and the mask — are computed exactly as before. That is what makes
/// the ring verifiable against the unbounded path instead of merely plausible.
struct KvStore {
    /// `usize::MAX` when the visit keeps full history.
    window: usize,
    dim: usize,
    /// Allocated slots; grows towards `window` and never shrinks.
    cap: usize,
    /// Absolute number of positions pushed.
    count: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl KvStore {
    fn new(window: Option<usize>, dim: usize) -> Self {
        Self {
            window: window.unwrap_or(usize::MAX),
            dim,
            cap: 0,
            count: 0,
            k: Vec::new(),
            v: Vec::new(),
        }
    }

    fn unbounded(&self) -> bool {
        self.window == usize::MAX
    }

    /// Absolute index of the oldest retained position.
    fn first(&self) -> usize {
        if self.unbounded() || self.count <= self.window {
            0
        } else {
            self.count - self.window
        }
    }

    /// Slot holding absolute position `t`.
    fn slot(&self, t: usize) -> usize {
        if self.unbounded() { t } else { t % self.window }
    }

    fn push(&mut self, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.dim);
        assert_eq!(v.len(), self.dim);
        if self.unbounded() {
            self.k.extend_from_slice(k);
            self.v.extend_from_slice(v);
        } else {
            if self.cap == 0 {
                self.cap = self.window.clamp(1, 16);
                self.k.resize(self.cap * self.dim, 0.0);
                self.v.resize(self.cap * self.dim, 0.0);
            } else if self.count == self.cap && self.cap < self.window {
                // While `count <= cap < window` every slot index equals the absolute
                // position, so growing is a resize and nothing has to be repacked.
                let next = (self.cap * 2).min(self.window);
                self.cap = next;
                self.k.resize(self.cap * self.dim, 0.0);
                self.v.resize(self.cap * self.dim, 0.0);
            }
            let slot = self.slot(self.count);
            self.k[slot * self.dim..(slot + 1) * self.dim].copy_from_slice(k);
            self.v[slot * self.dim..(slot + 1) * self.dim].copy_from_slice(v);
        }
        self.count += 1;
    }

    /// Bytes of KV **in use**. Deliberately `len` and not `capacity`: a `Vec` doubles
    /// itself, so capacity overstates the history by up to 2x and would make the per-token
    /// arithmetic disagree with the layout (`2 tensors x kv_dim x 4`).
    fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * 4
    }
}

struct VisitCache {
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    post_conv: Vec<f32>,
    ffn_conv: Vec<f32>,
    kv: KvStore,
}

impl VisitCache {
    fn new(n_embd: usize, kv_dim: usize, window: Option<usize>) -> Self {
        Self {
            k_conv: vec![0.0; 3 * kv_dim],
            v_conv: vec![0.0; 3 * kv_dim],
            post_conv: vec![0.0; 3 * n_embd],
            ffn_conv: vec![0.0; 3 * n_embd],
            kv: KvStore::new(window, kv_dim),
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
    ///
    /// 200058 rows of 2304 values is 322 MB of quantised weights read per token, which cost
    /// about a second single-threaded. The rows are independent, so they are split across
    /// threads; each worker dequantises into its own row buffer and writes a disjoint slice
    /// of the output.
    pub fn logits(&self, file: &GgufFile, hidden: &[f32]) -> Vec<f32> {
        let bytes = file
            .tensor_bytes(&self.info)
            .expect("output tensor is in the mmap");
        let n_embd = self.n_embd;
        let row_bytes = self.row_bytes;
        let cap = self.cap;
        let ggml_type = self.info.ggml_type;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(self.vocab.div_ceil(1))
            .max(1);
        let rows_per = self.vocab.div_ceil(threads).max(1);
        let mut out = vec![0f32; self.vocab];
        std::thread::scope(|scope| {
            for (chunk, slot) in out.chunks_mut(rows_per).enumerate() {
                let base = chunk * rows_per;
                scope.spawn(move || {
                    let mut row = vec![0f32; n_embd];
                    for (i, value) in slot.iter_mut().enumerate() {
                        let start = (base + i) * row_bytes;
                        scoria_gguf::dequant::dequant_row(
                            ggml_type,
                            &bytes[start..start + row_bytes],
                            &mut row,
                        )
                        .expect("unembedding row dequantises");
                        let dot = row.iter().zip(hidden).map(|(a, b)| a * b).sum::<f32>();
                        *value = softcap(dot, cap);
                    }
                });
            }
        });
        out
    }
}

// ---------------------------------------------------------------------------------------
// GPU projections
// ---------------------------------------------------------------------------------------
//
// The six attention projections and `W_o` are the arithmetic half of a visit: 2.6 GB of
// f32 weights read per token. They are uploaded once and run through `linear_f32`, which is
// already verified against the host (`harness/ledger` gate). Everything else -- the
// per-head attention, the relative bias, the convolutions, the MoE -- stays on the host
// for now, so this is a drop-in replacement for `matvec` with an explicit oracle: the same
// token must produce the same logits whether the projections run here or on the card.
/// Raw device pointers to one layer's projection weights. Copying them out of the `Gpu`
/// first keeps the borrow checker out of the way: the pointers outlive the `&mut Gpu` call
/// that consumes them.
/// Uploads a tensor only when the device can decode its encoding.
/// Uploads a tensor when the device can decode its encoding, recording which kernel does.
fn upload_if_supported(
    file: &GgufFile,
    info: &TensorInfo,
    ctx: &EngineContext,
    expert_rows: usize,
    expert_k: usize,
    outer: Vec<f32>,
) -> Option<GpuTensor> {
    let handle = match info.ggml_type {
        GgmlType::NVFP4 => ctx.infer.linear_nvfp4,
        GgmlType::IQ4NL => ctx.infer.linear_iq4nl,
        GgmlType::IQ3XXS => ctx.infer.linear_iq3_xxs,
        _ => return None,
    };
    let bytes = file.tensor_bytes(info).ok()?;
    let slab = info
        .ggml_type
        .tensor_byte_len(expert_k)
        .ok()?
        .checked_mul(expert_rows)?;
    let mut buf = DeviceBuf::alloc(bytes.len()).ok()?;
    buf.copy_from_host(bytes).ok()?;
    Some(GpuTensor {
        buf,
        handle,
        slab,
        rows: expert_rows,
        k: expert_k,
        outer,
    })
}

/// A `GpuTensor`'s launch parameters, copied out so the borrow of `gpu` can end.
#[derive(Clone)]
struct ProjSnapshot {
    ptr: *mut c_void,
    handle: scoria_hip_rt::KernelHandle,
    slab: usize,
    rows: usize,
    outer: Vec<f32>,
}

fn snapshot(t: &GpuTensor) -> ProjSnapshot {
    ProjSnapshot {
        ptr: t.buf.as_ptr(),
        handle: t.handle,
        slab: t.slab,
        rows: t.rows,
        outer: t.outer.clone(),
    }
}

/// The per-position inputs to the six projections.
struct AttnInputs {
    block_residual: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    vn: Vec<f32>,
    tau: f32,
}

struct GpuPtrs {
    q: *mut c_void,
    k: *mut c_void,
    v: *mut c_void,
    gate: *mut c_void,
    rel: *mut c_void,
    sink: *mut c_void,
    o: *mut c_void,
}

/// Upper bound on the routed experts batched into one synchronisation.
const TOP_K_FOR_BATCH: usize = 8;
/// Upper bound on the positions batched into one prefill chunk.
const CHUNK_MAX: usize = 64;

struct GpuLayer {
    q: DeviceBuf,
    k: DeviceBuf,
    v: DeviceBuf,
    gate: DeviceBuf,
    o: DeviceBuf,
    rel: DeviceBuf,
    sink: DeviceBuf,
}

/// One layer's expert tensors, when the checkpoint stores them in a format the device can
/// decode. NVFP4 only, for now: IQ2XS/IQ3XXS/IQ3S/IQ2S/Q3K/IQ4NL have no device
/// dequantiser yet, so those layers keep running on the host.
/// One expert tensor resident on the card, with the kernel that decodes *its* format.
struct GpuTensor {
    buf: DeviceBuf,
    handle: scoria_hip_rt::KernelHandle,
    slab: usize,
    rows: usize,
    k: usize,
    outer: Vec<f32>,
}

struct GpuExperts {
    gate_up: Option<GpuTensor>,
    down: Option<GpuTensor>,
}

impl GpuExperts {
    fn complete(&self) -> bool {
        self.gate_up.is_some() && self.down.is_some()
    }
}

struct Gpu {
    ctx: EngineContext,
    stream: HipStream,
    layers: Vec<GpuLayer>,
    experts: Vec<Option<GpuExperts>>,
    x: DeviceBuf,
    y: DeviceBuf,
    wave_size: u32,
    /// Bytes resident in VRAM, for the log line.
    pub resident: usize,
}

fn upload(ctx: &EngineContext, data: &[f32]) -> DeviceBuf {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    let mut buf = DeviceBuf::alloc(bytes.len()).expect("VRAM allocation for a projection");
    buf.copy_from_host(bytes).expect("H2D upload");
    let _ = ctx;
    buf
}

impl Gpu {
    fn new(ctx: EngineContext, layers: &[Layer], max_in: usize, max_out: usize) -> Self {
        let wave_size = ctx.wave_size;
        let stream = HipStream::new().expect("HIP stream");
        let mut resident = 0;
        let mut out = Vec::with_capacity(layers.len());
        for lw in layers {
            macro_rules! up {
                ($w:expr) => {{
                    let b = upload(&ctx, $w);
                    resident += b.bytes();
                    b
                }};
            }
            out.push(GpuLayer {
                q: up!(&lw.w_q),
                k: up!(&lw.w_k),
                v: up!(&lw.w_v),
                gate: up!(&lw.w_gate),
                o: up!(&lw.w_o),
                rel: up!(&lw.w_rel),
                sink: up!(&lw.w_sink),
            });
        }
        Self {
            experts: Vec::new(),
            // `W_o` reads `n_heads * head_dim` floats, which is wider than the model's
            // hidden size, so the input buffer is sized for the widest operand.
            // A batch carries `top_k` input vectors, so the input buffer is that much
            // wider than one projection vector.
            x: DeviceBuf::alloc(max_in.max(max_out) * 4 * TOP_K_FOR_BATCH * CHUNK_MAX)
                .expect("VRAM input buffer"),
            // One batch is `top_k` expert outputs, so the output buffer is wider than a
            // single projection's.
            y: DeviceBuf::alloc(max_out * 4 * TOP_K_FOR_BATCH * CHUNK_MAX)
                .expect("VRAM output buffer"),
            ctx,
            stream,
            layers: out,
            wave_size,
            resident,
        }
    }

    /// The six projections for a whole chunk: one upload per distinct input (`qn` feeds q,
    /// gate, rel and sink), one launch per projection with `m = chunk`, and a single
    /// synchronisation.
    #[allow(clippy::too_many_arguments)]
    fn run_projections_chunk(
        &mut self,
        bufs: GpuPtrs,
        qn: &[f32],
        kn: &[f32],
        vn: &[f32],
        m: usize,
        n_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        d_rel: usize,
    ) -> [Vec<f32>; 6] {
        let n_embd = qn.len() / m;
        let q_dim = n_heads * head_dim;
        let dims = [q_dim, kv_dim, kv_dim, n_heads * d_rel, q_dim, n_heads];
        // Input regions in the upload buffer: qn, then kn, then vn. All three inputs are
        // `n_embd` wide -- `kv_dim` is the *output* width of the k/v projections, and using
        // it as the input stride was the bug that made only k and v wrong.
        let x_off = [0usize, m * n_embd, 2 * m * n_embd, 0, 0, 0];
        let in_dim = [n_embd, n_embd, n_embd, n_embd, n_embd, n_embd];
        let weights = [bufs.q, bufs.k, bufs.v, bufs.rel, bufs.gate, bufs.sink];

        let mut flat = Vec::with_capacity(3 * m * n_embd);
        flat.extend_from_slice(qn);
        flat.extend_from_slice(kn);
        flat.extend_from_slice(vn);
        let bytes =
            unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4) };
        self.x.copy_from_host(bytes).expect("H2D chunk projections");

        let mut offsets = [0usize; 6];
        let mut total = 0usize;
        for (i, d) in dims.iter().enumerate() {
            offsets[i] = total;
            total += d * m;
        }
        let waves_per_wg = 256u32 / self.wave_size.max(1);
        for i in 0..6 {
            let mut arg_w = weights[i];
            let mut arg_x =
                unsafe { (self.x.as_ptr() as *mut u8).add(x_off[i] * 4) as *mut c_void };
            let mut arg_bias: *mut c_void = std::ptr::null_mut();
            let mut arg_y =
                unsafe { (self.y.as_ptr() as *mut u8).add(offsets[i] * 4) as *mut c_void };
            let mut arg_m = m as u32;
            let mut arg_n = dims[i] as u32;
            let mut arg_k = in_dim[i] as u32;
            let mut args: [*mut c_void; 7] = [
                &mut arg_w as *mut *mut c_void as *mut c_void,
                &mut arg_x as *mut *mut c_void as *mut c_void,
                &mut arg_bias as *mut *mut c_void as *mut c_void,
                &mut arg_y as *mut *mut c_void as *mut c_void,
                &mut arg_m as *mut u32 as *mut c_void,
                &mut arg_n as *mut u32 as *mut c_void,
                &mut arg_k as *mut u32 as *mut c_void,
            ];
            unsafe {
                self.ctx
                    .infer_module
                    .launch_handle(
                        self.ctx.infer.linear_f32,
                        [
                            ((dims[i] as u32).div_ceil(waves_per_wg)).max(1),
                            m as u32,
                            1,
                        ],
                        [256, 1, 1],
                        0,
                        &self.stream,
                        &mut args,
                    )
                    .expect("chunked projection launch");
            }
        }
        self.stream.synchronize().expect("projection sync");
        let mut all = vec![0f32; total];
        let out_bytes =
            unsafe { std::slice::from_raw_parts_mut(all.as_mut_ptr() as *mut u8, total * 4) };
        self.y
            .copy_to_host(out_bytes)
            .expect("D2H chunk projections");
        let mut out: [Vec<f32>; 6] = Default::default();
        for i in 0..6 {
            out[i] = all[offsets[i]..offsets[i] + dims[i] * m].to_vec();
        }
        out
    }

    /// One launch per group, one synchronisation: `groups` is
    /// `(expert, offset, count)` and `x_flat`/`y_flat` are indexed at
    /// `offset * k` / `offset * n_out`. This is how a batched prefill runs the experts: the
    /// positions that chose an expert become `count` rows of one GEMV, so the expert's
    /// weights are read once instead of once per position.
    #[allow(clippy::too_many_arguments)]
    fn run_groups(
        &mut self,
        handle: scoria_hip_rt::KernelHandle,
        base: *mut c_void,
        slab: usize,
        groups: &[(usize, usize, usize)],
        x_flat: &[f32],
        k: usize,
        n_out: usize,
    ) -> Vec<f32> {
        let bytes =
            unsafe { std::slice::from_raw_parts(x_flat.as_ptr() as *const u8, x_flat.len() * 4) };
        self.x.copy_from_host(bytes).expect("H2D chunk inputs");
        let mut n = n_out as u32;
        let mut kk = k as u32;
        let waves_per_wg = 256u32 / self.wave_size.max(1);
        let grid_x = n.div_ceil(waves_per_wg);
        for (expert, offset, count) in groups {
            if *count == 0 {
                continue;
            }
            let mut arg_w = unsafe { (base as *mut u8).add(expert * slab) as *mut c_void };
            let mut arg_x =
                unsafe { (self.x.as_ptr() as *mut u8).add(offset * k * 4) as *mut c_void };
            let mut arg_bias: *mut c_void = std::ptr::null_mut();
            let mut arg_y =
                unsafe { (self.y.as_ptr() as *mut u8).add(offset * n_out * 4) as *mut c_void };
            let mut arg_m = *count as u32;
            let mut args: [*mut c_void; 7] = [
                &mut arg_w as *mut *mut c_void as *mut c_void,
                &mut arg_x as *mut *mut c_void as *mut c_void,
                &mut arg_bias as *mut *mut c_void as *mut c_void,
                &mut arg_y as *mut *mut c_void as *mut c_void,
                &mut arg_m as *mut u32 as *mut c_void,
                &mut n as *mut u32 as *mut c_void,
                &mut kk as *mut u32 as *mut c_void,
            ];
            unsafe {
                self.ctx
                    .infer_module
                    .launch_handle(
                        handle,
                        [grid_x, *count as u32, 1],
                        [256, 1, 1],
                        0,
                        &self.stream,
                        &mut args,
                    )
                    .expect("grouped GEMV launch");
            }
        }
        self.stream.synchronize().expect("stream sync");
        let total: usize = groups.iter().map(|g| g.2 * n_out).sum();
        let mut out = vec![0f32; total];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, total * 4) };
        self.y.copy_to_host(bytes).expect("D2H groups");
        out
    }

    /// Keep every expert tensor the device can decode resident, one allocation at a time,
    /// and give up on a tensor (not on the whole run) if VRAM runs out. This is the
    /// residency half of a tiered store: the hot set is whatever fits, the cold set is the
    /// GGUF mmap, and residency is decided per tensor because half an expert tensor is
    /// never useful.
    fn upload_experts(&mut self, file: &GgufFile, moes: &[Moe], residency: bool) -> usize {
        if !residency || std::env::var_os("SCORIA_NO_EXPERT_UPLOAD").is_some() {
            self.experts = (0..moes.len()).map(|_| None).collect();
            return 0;
        }
        let mut resident = 0;
        self.experts = Vec::with_capacity(moes.len());
        for moe in moes {
            let gu = moe.gate_up_tensor();
            let dn = moe.down_tensor();
            let gu_buf = upload_if_supported(
                file,
                &gu.info,
                &self.ctx,
                gu.out_dim,
                gu.in_dim,
                gu.outer.clone(),
            );
            let dn_buf = upload_if_supported(
                file,
                &dn.info,
                &self.ctx,
                dn.out_dim,
                dn.in_dim,
                dn.outer.clone(),
            );
            resident += gu_buf.as_ref().map(|t| t.buf.bytes()).unwrap_or(0);
            resident += dn_buf.as_ref().map(|t| t.buf.bytes()).unwrap_or(0);
            self.experts.push(Some(GpuExperts {
                gate_up: gu_buf,
                down: dn_buf,
            }));
        }
        resident
    }

    /// `y = W x` for one token through `linear_f32`.
    fn run(&mut self, weight: *mut c_void, x: &[f32], n_out: usize) -> Vec<f32> {
        self.run_kernel(self.ctx.infer.linear_f32, weight, x, n_out)
    }

    /// Several independent single-token GEMVs — one weight matrix and one input vector per
    /// slot — as **one** host-to-device copy, one launch per slot, and **one**
    /// synchronisation. The routed experts are independent, so this is where the launch
    /// round-trip stops dominating: sixteen synchronisations per visit became two.
    fn run_batch(
        &mut self,
        handle: scoria_hip_rt::KernelHandle,
        weights: &[*mut c_void],
        inputs: &[&[f32]],
        n_out: usize,
    ) -> Vec<f32> {
        assert_eq!(weights.len(), inputs.len(), "one weight per input");
        let k = inputs[0].len();
        assert!(
            inputs.iter().all(|i| i.len() == k),
            "inputs must share a width"
        );
        let mut flat = Vec::with_capacity(inputs.len() * k);
        for input in inputs {
            flat.extend_from_slice(input);
        }
        let bytes =
            unsafe { std::slice::from_raw_parts(flat.as_ptr() as *const u8, flat.len() * 4) };
        self.x.copy_from_host(bytes).expect("H2D batch inputs");
        let mut one = 1u32;
        let mut n = n_out as u32;
        let mut kk = k as u32;
        let waves_per_wg = 256u32 / self.wave_size.max(1);
        let grid_x = n.div_ceil(waves_per_wg);
        for (slot, weight) in weights.iter().enumerate() {
            let mut arg_w = *weight;
            let mut arg_x =
                unsafe { (self.x.as_ptr() as *mut u8).add(slot * k * 4) as *mut c_void };
            let mut arg_bias: *mut c_void = std::ptr::null_mut();
            let mut arg_y =
                unsafe { (self.y.as_ptr() as *mut u8).add(slot * n_out * 4) as *mut c_void };
            let mut args: [*mut c_void; 7] = [
                &mut arg_w as *mut *mut c_void as *mut c_void,
                &mut arg_x as *mut *mut c_void as *mut c_void,
                &mut arg_bias as *mut *mut c_void as *mut c_void,
                &mut arg_y as *mut *mut c_void as *mut c_void,
                &mut one as *mut u32 as *mut c_void,
                &mut n as *mut u32 as *mut c_void,
                &mut kk as *mut u32 as *mut c_void,
            ];
            unsafe {
                self.ctx
                    .infer_module
                    .launch_handle(
                        handle,
                        [grid_x, 1, 1],
                        [256, 1, 1],
                        0,
                        &self.stream,
                        &mut args,
                    )
                    .expect("batched GEMV launch");
            }
        }
        self.stream.synchronize().expect("stream sync");
        let total = weights.len() * n_out;
        let mut out = vec![0f32; total];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, total * 4) };
        self.y.copy_to_host(bytes).expect("D2H batch");
        out
    }

    fn run_kernel(
        &mut self,
        handle: scoria_hip_rt::KernelHandle,
        weight: *mut c_void,
        x: &[f32],
        n_out: usize,
    ) -> Vec<f32> {
        let x_bytes = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        self.x.copy_from_host(x_bytes).expect("H2D x");
        let mut m = 1u32;
        let mut n = n_out as u32;
        let mut k = x.len() as u32;
        let waves_per_wg = 256u32 / self.wave_size.max(1);
        let grid_x = n.div_ceil(waves_per_wg);
        // `hipModuleLaunchKernel` copies each argument *from the address* in this array,
        // including pointer arguments: the entry must be `&ptr`, not `ptr`. Passing the
        // pointer value makes HIP read the kernel argument out of device memory on the host,
        // which faults inside libamdhip64 (or, once the queue runs, raises
        // HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION from the kernel).
        let mut arg_w = weight;
        let mut arg_x = self.x.as_ptr();
        let mut arg_bias: *mut c_void = std::ptr::null_mut();
        let mut arg_y = self.y.as_ptr();
        let mut args: [*mut c_void; 7] = [
            &mut arg_w as *mut *mut c_void as *mut c_void,
            &mut arg_x as *mut *mut c_void as *mut c_void,
            &mut arg_bias as *mut *mut c_void as *mut c_void,
            &mut arg_y as *mut *mut c_void as *mut c_void,
            &mut m as *mut u32 as *mut c_void,
            &mut n as *mut u32 as *mut c_void,
            &mut k as *mut u32 as *mut c_void,
        ];
        unsafe {
            self.ctx
                .infer_module
                .launch_handle(
                    handle,
                    [grid_x, 1, 1],
                    [256, 1, 1],
                    0,
                    &self.stream,
                    &mut args,
                )
                .expect("linear_f32 launch");
        }
        self.stream.synchronize().expect("stream sync");
        let mut out = vec![0f32; n_out];
        let out_bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n_out * 4) };
        self.y.copy_to_host(out_bytes).expect("D2H y");
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
    /// Overrides the attention window (and the ring capacity) for tests; `None` uses the
    /// checkpoint's `sliding_window`.
    kv_window_override: Option<usize>,
    /// When false, windowed visits keep their whole history and rely on the mask alone.
    /// That is the oracle the ring is verified against.
    kv_ring: bool,
    /// Present when the projections run on the card; `None` keeps the verified host path.
    gpu: Option<Gpu>,
    /// Whether expert tensors may be uploaded to VRAM. Measured to be strongly
    /// path-dependent: residency is 2.5x faster for decode and 20-50x slower for the chunked
    /// prefill, so it is a policy, not a constant.
    expert_residency: bool,
    traces: Vec<VisitTrace>,
    /// Time spent in the MoE half of each visit, and in everything else.
    pub profile: Profile,
    /// Stop after this many visits and run the head there anyway. Only for depth sweeps:
    /// a truncated model is not a model, but where perplexity stops improving localises a
    /// broken layer.
    pub max_visits: usize,
    pub last_report: Option<StepReport>,
}

/// Diagnostics for the last layer visited, so a wrong-looking logit can be traced back to
/// the layer that produced it.
/// Where a token's time goes, in seconds. Two buckets are enough to decide what to
/// optimise next.
#[derive(Debug, Clone, Copy, Default)]
pub struct Profile {
    pub attention: f64,
    pub moe: f64,
    /// Of `attention`, the part spent in GPU projections (if any).
    pub gpu_proj: f64,
    /// Of `moe`, the part spent in GPU expert GEMVs (if any).
    pub gpu_experts: f64,
    pub visits: usize,
}

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
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let caches = visits
            .iter()
            .map(|v| {
                let sliding = layers[v.layer].sliding;
                let window = sliding.then(|| cfg.sliding_window.unwrap_or(1024));
                VisitCache::new(cfg.n_embd, kv_dim, window)
            })
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
            kv_window_override: None,
            kv_ring: true,
            gpu: None,
            expert_residency: true,
            traces: Vec::new(),
            profile: Profile::default(),
            max_visits: usize::MAX,
            params,
            cfg,
            file,
            last_report: None,
        })
    }

    /// Move the projection weights to VRAM and run them on the card.
    ///
    /// 2.6 GB of f32 weights, uploaded once. Everything else stays on the host, so this is
    /// exactly the same arithmetic as `matvec` and the logits are the oracle.
    pub fn enable_gpu(&mut self) -> Result<usize, String> {
        if self.gpu.is_some() {
            return Ok(0);
        }
        let ctx = EngineContext::init(0).map_err(|e| format!("HIP init: {e:?}"))?;
        let max_in = self.cfg.n_embd;
        let max_out = (self.cfg.n_heads * self.cfg.head_dim).max(self.cfg.n_embd);
        let mut gpu = Gpu::new(ctx, &self.layers, max_in, max_out);
        let resident =
            gpu.resident + gpu.upload_experts(&self.file, &self.moes, self.expert_residency);
        self.gpu = Some(gpu);
        Ok(resident)
    }

    /// Rebuild the caches with a different attention window and/or with the ring eviction
    /// switched off. `ring = false` keeps every position and masks afterwards, which is the
    /// behaviour the ring has to reproduce exactly.
    pub fn configure_kv(&mut self, window: Option<usize>, ring: bool) {
        self.kv_window_override = window;
        self.kv_ring = ring;
        self.reset();
    }

    /// The routing decision of the last forward for `layer`, for locality analysis.
    pub fn route_of(&self, layer: usize) -> Option<&crate::inkling2_moe::Route> {
        self.moes[layer].last_route.as_ref()
    }

    /// Time `dequant_row` against the dot product for one expert matrix, so the question
    /// "would caching dequantised experts help?" is answered by arithmetic instead of hope.
    /// Returns `(dequant_seconds, dot_seconds)` for `reps` sweeps of the slab.
    /// `down = true` benchmarks the `ffn_down_exps` matrix, otherwise gate/up.
    pub fn bench_dequant(
        &self,
        layer: usize,
        down: bool,
        expert: usize,
        reps: usize,
    ) -> (f64, f64) {
        let tensor = if down {
            self.moes[layer].down_tensor()
        } else {
            self.moes[layer].gate_up_tensor()
        };
        let e = expert;
        let out_dim = tensor.out_dim;
        let in_dim = tensor.in_dim;
        let bytes = self
            .file
            .tensor_bytes(&tensor.info)
            .expect("expert tensor in mmap");
        let row_bytes = tensor
            .info
            .ggml_type
            .tensor_byte_len(in_dim)
            .expect("row size");
        let base = e * tensor.slab_bytes;
        let mut row = vec![0f32; in_dim];
        let x: Vec<f32> = (0..in_dim).map(|i| (i % 17) as f32 * 0.01).collect();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            for o in 0..out_dim {
                let start = base + o * row_bytes;
                scoria_gguf::dequant::dequant_row(
                    tensor.info.ggml_type,
                    &bytes[start..start + row_bytes],
                    &mut row,
                )
                .expect("row dequantises");
            }
        }
        let dequant = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            for o in 0..out_dim {
                let start = base + o * row_bytes;
                scoria_gguf::dequant::dequant_row(
                    tensor.info.ggml_type,
                    &bytes[start..start + row_bytes],
                    &mut row,
                )
                .expect("row dequantises");
                std::hint::black_box(row.iter().zip(&x).map(|(a, b)| a * b).sum::<f32>());
            }
        }
        let both = t1.elapsed().as_secs_f64();
        (dequant, (both - dequant).max(0.0))
    }

    /// The attention window in force for this model.
    fn mask_window(&self) -> Option<usize> {
        self.kv_window_override.or(self.cfg.sliding_window)
    }

    /// Bytes the KV history currently occupies, split by whether the visit is bounded.
    /// `(windowed, full)` in bytes.
    pub fn kv_bytes(&self) -> (usize, usize) {
        let mut windowed = 0;
        let mut full = 0;
        for (cache, v) in self.caches.iter().zip(&self.visits) {
            if self.layers[v.layer].sliding {
                windowed += cache.kv.bytes();
            } else {
                full += cache.kv.bytes();
            }
        }
        (windowed, full)
    }

    /// Keep expert tensors off the card. Decode wants them resident; the chunked prefill is
    /// 20-50x slower with them resident on this machine, so the two paths disagree and the
    /// caller decides.
    pub fn set_expert_residency(&mut self, on: bool) {
        self.expert_residency = on;
    }

    pub fn gpu_active(&self) -> bool {
        self.gpu.is_some()
    }

    /// Worker count for the MoE expert loop of every layer.
    pub fn set_threads(&mut self, threads: usize) {
        for moe in self.moes.iter_mut() {
            moe.set_threads(threads);
        }
    }

    pub fn reset(&mut self) {
        let kv_dim = self.cfg.n_kv_heads * self.cfg.head_dim;
        let ring = self.kv_ring;
        // The ring capacity is the *effective* window, so overriding it for a test shrinks
        // the ring too -- otherwise the store would never evict and the test would compare
        // two unbounded stores.
        let window = self.mask_window().unwrap_or(1024);
        for (c, v) in self.caches.iter_mut().zip(&self.visits) {
            let sliding = self.layers[v.layer].sliding;
            let capacity = sliding.then_some(window);
            *c = VisitCache::new(self.cfg.n_embd, kv_dim, if ring { capacity } else { None });
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
        self.profile = Profile::default();
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

    /// The attention half of a visit: everything up to and including the normalised MoE
    /// input. Returns `(residual, moe_in)`.
    ///
    /// Split out so that a chunk of positions can run the same code per position and then
    /// share one batched MoE. The single-position path calls the two halves back to back,
    /// so batching cannot drift from it.
    /// The projection inputs for one position: the attention residual, the q/k/v streams after
    /// the message gates, their input norms, and the log-scale factor.
    ///
    /// Split out of `attention_half` so a chunk can compute every position's inputs and then
    /// run the six projection GEMVs once with `m = chunk`.
    fn attn_inputs(
        &self,
        layer: usize,
        h: &[f32],
        blocks: &[Vec<f32>],
        msg: Option<&[Vec<f32>]>,
    ) -> AttnInputs {
        let cfg = &self.cfg;
        let eps = cfg.rms_eps;
        let n_embd = cfg.n_embd;
        let lw = &self.layers[layer];

        let block_residual = h.to_vec();

        // `attention_input` yields the q/k/v streams; layer 0 has none and passes hidden
        // through unchanged.
        let mut streams: Vec<Vec<f32>> = if lw.idx == 0 {
            vec![h.to_vec(); lw.queries]
        } else {
            block_attention(&lw.attn_res_query, lw.queries, blocks, h, eps)
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
        AttnInputs {
            block_residual,
            qn,
            kn,
            vn,
            tau,
        }
    }

    /// The six projections for one position, on the card when the weights are resident.
    fn project_position(&mut self, layer: usize, ins: &AttnInputs) -> [Vec<f32>; 6] {
        let qn: &[f32] = &ins.qn;
        let kn: &[f32] = &ins.kn;
        let vn: &[f32] = &ins.vn;
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd;
        let lw = &self.layers[layer];
        let n_heads = lw.n_heads;
        let head_dim = lw.head_dim;
        let kv_dim = lw.kv_dim;
        // The projection weights may be resident in VRAM, in which case `linear_f32` runs
        // each one as a single-token GEMV. Otherwise the six host matvecs run concurrently:
        // one spawn round per visit instead of one per matvec.
        let gpu_ptrs = self.gpu.as_ref().map(|g| {
            let gl = &g.layers[layer];
            GpuPtrs {
                q: gl.q.as_ptr(),
                k: gl.k.as_ptr(),
                v: gl.v.as_ptr(),
                gate: gl.gate.as_ptr(),
                rel: gl.rel.as_ptr(),
                sink: gl.sink.as_ptr(),
                o: gl.o.as_ptr(),
            }
        });
        let (q, k, v, rel, gate, sink) = match gpu_ptrs {
            Some(ptrs) => {
                let t = std::time::Instant::now();
                let g = self.gpu.as_mut().expect("gpu active");
                let q = g.run(ptrs.q, qn, n_heads * head_dim);
                let k = g.run(ptrs.k, kn, kv_dim);
                let v = g.run(ptrs.v, vn, kv_dim);
                let gate = g.run(ptrs.gate, qn, n_heads * head_dim);
                let rel = g.run(ptrs.rel, qn, n_heads * lw.d_rel);
                let sink = g.run(ptrs.sink, qn, n_heads);
                self.profile.gpu_proj += t.elapsed().as_secs_f64();
                (q, k, v, rel, gate, sink)
            }
            None => std::thread::scope(|scope| {
                let hq = scope.spawn(|| matvec(&lw.w_q, qn, n_heads * head_dim, n_embd));
                let hk = scope.spawn(|| matvec(&lw.w_k, kn, kv_dim, n_embd));
                let hv = scope.spawn(|| matvec(&lw.w_v, vn, kv_dim, n_embd));
                let hg = scope.spawn(|| matvec(&lw.w_gate, qn, n_heads * head_dim, n_embd));
                let hr = scope.spawn(|| matvec(&lw.w_rel, qn, n_heads * lw.d_rel, n_embd));
                let hs = scope.spawn(|| matvec(&lw.w_sink, qn, n_heads, n_embd));
                (
                    hq.join().expect("q projection"),
                    hk.join().expect("k projection"),
                    hv.join().expect("v projection"),
                    hr.join().expect("rel projection"),
                    hg.join().expect("gate projection"),
                    hs.join().expect("sink projection"),
                )
            }),
        };
        [q, k, v, rel, gate, sink]
    }

    /// Everything after the projections for one position: the per-head clip and convolution,
    /// the relative bias, attention, the output gate, `W_o`, the post convolution, the block
    /// boundary, and the normalised MoE input.
    #[allow(clippy::too_many_arguments)]
    fn attn_finish(
        &mut self,
        layer: usize,
        h: &[f32],
        ins: &AttnInputs,
        proj: [Vec<f32>; 6],
        blocks: &mut Vec<Vec<f32>>,
        visit: usize,
        msg: Option<&[Vec<f32>]>,
    ) -> (Vec<f32>, Vec<f32>) {
        let cfg = &self.cfg;
        let eps = cfg.rms_eps;
        let n_embd = cfg.n_embd;
        let lw = &self.layers[layer];
        let mask_window = self.mask_window();
        let cache = &mut self.caches[visit];
        let n_heads = lw.n_heads;
        let head_dim = lw.head_dim;
        let kv_dim = lw.kv_dim;
        let group = n_heads / lw.kv_heads;
        let block_residual = ins.block_residual.clone();
        let tau = ins.tau;
        let kn: &[f32] = &ins.kn;
        let vn: &[f32] = &ins.vn;
        let [mut q, mut k, mut v, mut rel, mut gate, sink] = proj;
        let gpu_ptrs = self.gpu.as_ref().map(|g| {
            let gl = &g.layers[layer];
            GpuPtrs {
                q: gl.q.as_ptr(),
                k: gl.k.as_ptr(),
                v: gl.v.as_ptr(),
                gate: gl.gate.as_ptr(),
                rel: gl.rel.as_ptr(),
                sink: gl.sink.as_ptr(),
                o: gl.o.as_ptr(),
            }
        });
        for hh in 0..n_heads {
            scaled_l2_clip(
                &mut q[hh * head_dim..(hh + 1) * head_dim],
                &lw.q_clip[hh * head_dim..(hh + 1) * head_dim],
                lw.qk_clip,
            );
        }
        if tau != 1.0 {
            for value in q.iter_mut() {
                *value *= tau;
            }
        }

        match (&lw.conv_k_proj, &lw.conv_v_proj) {
            (Some(pk), Some(pv)) => {
                let dyn_k = matvec(pk, kn, lw.kv_heads * 4, n_embd);
                let dyn_v = matvec(pv, vn, lw.kv_heads * 4, n_embd);
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

        for g in gate.iter_mut() {
            *g = sigmoid(*g);
        }
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

        let kv_pos = cache.kv.count;
        cache.kv.push(&k, &v);

        // The visible set is the intersection of "still retained" (the ring's window) and
        // "inside the attention window". Trimming here instead of masking inside
        // `decode_head` keeps the two identical: the ring can only ever drop positions the
        // mask would have hidden anyway.
        let window = if lw.sliding {
            Some(mask_window.unwrap_or(lw.rel_extent))
        } else {
            None
        };
        let first_visible = match window {
            Some(w) => cache.kv.first().max((kv_pos + 1).saturating_sub(w)),
            None => cache.kv.first(),
        };
        let visible = kv_pos + 1 - first_visible;
        assert_eq!(visible, cache.kv.count - first_visible);

        let pos_k: Vec<usize> = (first_visible..=kv_pos).collect();
        let mut bias = vec![0f32; n_heads * visible];
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

        // Read each head straight out of the interleaved cache. The per-head copy this
        // replaces was 25 M floats per full-history visit at 4096 positions, i.e. the copy
        // loop rather than the dots dominated attention at long context.
        let slots: Vec<usize> = (first_visible..=kv_pos).map(|t| cache.kv.slot(t)).collect();
        let mut attn = vec![0f32; n_heads * head_dim];
        for hh in 0..n_heads {
            let kvh = hh / group;
            let out = decode_head_in_cache(
                &q[hh * head_dim..(hh + 1) * head_dim],
                &cache.kv.k,
                &cache.kv.v,
                &slots,
                lw.kv_heads,
                kvh,
                &bias[hh * visible..(hh + 1) * visible],
                sink[hh],
            );
            attn[hh * head_dim..(hh + 1) * head_dim].copy_from_slice(&out);
        }

        // Diagnostics: head 0's attention distribution including the sink.
        {
            let mut logits = Vec::with_capacity(visible + 1);
            let scale = (head_dim as f32).sqrt().recip();
            // Head 0 only: `bias` is `heads * visible` long, so the row must be bounded
            // explicitly or the diagnostic walks into the next head's row.
            for (i, bias_t) in bias.iter().enumerate().take(visible) {
                let t = first_visible + i;
                let slot = cache.kv.slot(t);
                let base = slot * lw.kv_dim;
                let dot: f32 = (0..head_dim).map(|d| q[d] * cache.kv.k[base + d]).sum();
                logits.push(dot * scale + bias_t);
            }
            let mut with_sink = logits.clone();
            with_sink.push(sink[0]);
            let max = with_sink.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = with_sink.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|v| v / sum).collect();
            let (peak_idx, peak) = probs[..visible].iter().enumerate().fold(
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
                peak_distance: (visible - 1) - peak_idx,
                sink: probs[visible],
                entropy,
            });
        }

        let gated: Vec<f32> = attn.iter().zip(&gate).map(|(a, g)| a * g).collect();
        let mut out = match gpu_ptrs {
            Some(ptrs) => {
                let t = std::time::Instant::now();
                let g = self.gpu.as_mut().expect("gpu active");
                let out = g.run(ptrs.o, &gated, n_embd);
                self.profile.gpu_proj += t.elapsed().as_secs_f64();
                out
            }
            None => matvec(&lw.w_o, &gated, n_embd, n_heads * head_dim),
        };
        short_conv1d_4tap(&mut out, &lw.post_conv, &mut cache.post_conv, n_embd);

        let residual = if lw.boundary {
            blocks.push(block_residual);
            out
        } else {
            h.iter().zip(&out).map(|(a, b)| a + b).collect()
        };

        let mut ffn_in = block_attention(&lw.ffn_res_query, 1, blocks, &residual, eps).remove(0);
        if let (Some(msg), Some(gates)) = (msg, lw.gates.as_ref()) {
            ffn_in = gates[3].apply(&ffn_in, &msg[3], n_embd, eps);
        }
        let mut moe_in = ffn_in;
        rmsnorm(&mut moe_in, &lw.ffn_norm, eps);
        (residual, moe_in)
    }

    /// The six projections for a whole chunk, split back into one set per position.
    fn project_chunk(&mut self, layer: usize, ins: &[AttnInputs]) -> Vec<[Vec<f32>; 6]> {
        let m = ins.len();
        let lw = &self.layers[layer];
        let (n_heads, head_dim, kv_dim, d_rel) = (lw.n_heads, lw.head_dim, lw.kv_dim, lw.d_rel);
        let bufs = self.gpu.as_ref().map(|g| {
            let gl = &g.layers[layer];
            GpuPtrs {
                q: gl.q.as_ptr(),
                k: gl.k.as_ptr(),
                v: gl.v.as_ptr(),
                gate: gl.gate.as_ptr(),
                rel: gl.rel.as_ptr(),
                sink: gl.sink.as_ptr(),
                o: gl.o.as_ptr(),
            }
        });
        let Some(bufs) = bufs.filter(|_| m > 1) else {
            return ins
                .iter()
                .map(|i| self.project_position(layer, i))
                .collect();
        };

        let mut qn = Vec::with_capacity(m * self.cfg.n_embd);
        let mut kn = Vec::new();
        let mut vn = Vec::new();
        for i in ins {
            qn.extend_from_slice(&i.qn);
            kn.extend_from_slice(&i.kn);
            vn.extend_from_slice(&i.vn);
        }
        let t = std::time::Instant::now();
        let flat = self
            .gpu
            .as_mut()
            .expect("gpu")
            .run_projections_chunk(bufs, &qn, &kn, &vn, m, n_heads, head_dim, kv_dim, d_rel);
        self.profile.gpu_proj += t.elapsed().as_secs_f64();

        let dims = [
            n_heads * head_dim,
            kv_dim,
            kv_dim,
            n_heads * d_rel,
            n_heads * head_dim,
            n_heads,
        ];
        (0..m)
            .map(|p| {
                let mut per: [Vec<f32>; 6] = Default::default();
                for j in 0..6 {
                    let d = dims[j];
                    per[j] = flat[j][p * d..(p + 1) * d].to_vec();
                }
                per
            })
            .collect()
    }

    /// Compare the batched projection launch against the verified per-position path. Used by
    /// `--proj-selftest`: the batched path produced wrong logits once, and this is the
    /// smallest possible reproduction of that failure.
    pub fn projection_selftest(&mut self, m: usize) -> Vec<(usize, [f32; 6])> {
        let mut worst = Vec::new();
        let n_embd = self.cfg.n_embd;
        for layer in [0usize, 5, 12, 23] {
            let ins: Vec<AttnInputs> = (0..m)
                .map(|p| {
                    let mk = |salt: usize| -> Vec<f32> {
                        (0..n_embd)
                            .map(|i| (((i + salt) * 37 % 211) as f32 - 105.0) * 1e-3)
                            .collect()
                    };
                    AttnInputs {
                        block_residual: vec![0.0; n_embd],
                        qn: mk(p),
                        kn: mk(p + 1),
                        vn: mk(p + 2),
                        tau: 1.0,
                    }
                })
                .collect();
            let batched = self.project_chunk(layer, &ins);
            let mut layer_worst = [0f32; 6];
            for (p, i) in ins.iter().enumerate() {
                let single = self.project_position(layer, i);
                for j in 0..6 {
                    for (a, b) in batched[p][j].iter().zip(&single[j]) {
                        layer_worst[j] = layer_worst[j].max((a - b).abs());
                    }
                }
            }
            worst.push((layer, layer_worst));
        }
        worst
    }

    /// One position through a visit: inputs, projections, then the rest.
    fn attention_half(
        &mut self,
        layer: usize,
        h: &[f32],
        blocks: &mut Vec<Vec<f32>>,
        msg: Option<&[Vec<f32>]>,
        visit: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let ins = self.attn_inputs(layer, h, blocks, msg);
        let proj = self.project_position(layer, &ins);
        self.attn_finish(layer, h, &ins, proj, blocks, visit, msg)
    }

    /// The FFN half of a visit: `post_mlp_conv` and the residual add.
    fn ffn_half(
        &mut self,
        layer: usize,
        residual: &[f32],
        moe_out: &[f32],
        visit: usize,
    ) -> Vec<f32> {
        let n_embd = self.cfg.n_embd;
        let mut y = moe_out.to_vec();
        {
            let lw = &self.layers[layer];
            let cache = &mut self.caches[visit];
            short_conv1d_4tap(&mut y, &lw.ffn_post_conv, &mut cache.ffn_conv, n_embd);
        }
        residual.iter().zip(&y).map(|(a, b)| a + b).collect()
    }

    /// One position through a visit: the two halves with the MoE between them.
    fn layer_forward(
        &mut self,
        layer: usize,
        h: &[f32],
        msg: Option<&[Vec<f32>]>,
        visit: usize,
    ) -> Vec<f32> {
        let t_visit = std::time::Instant::now();
        self.profile.visits += 1;
        let n_embd = self.cfg.n_embd;
        let mut blocks = std::mem::take(&mut self.block_states);
        let (residual, moe_in) = self.attention_half(layer, h, &mut blocks, msg, visit);
        self.block_states = blocks;
        let t_moe = std::time::Instant::now();
        let moe_out = Self::moe_forward_parts(
            &mut self.gpu,
            &self.moes[layer],
            &self.file,
            layer,
            n_embd,
            &moe_in,
            &mut self.profile,
        );
        let moe_time = t_moe.elapsed().as_secs_f64();
        self.profile.moe += moe_time;
        let out = self.ffn_half(layer, &residual, &moe_out, visit);
        self.profile.attention += t_visit.elapsed().as_secs_f64() - moe_time;
        out
    }

    /// The MoE half of a visit: expert GEMVs on the card when the layer's experts are
    /// resident and NVFP4, otherwise the verified host loop. The routing, the SwiGLU and
    /// the weighted accumulation are identical in both, so the host path stays the oracle.
    #[allow(clippy::too_many_arguments)]
    fn moe_forward_parts(
        gpu: &mut Option<Gpu>,
        moe: &Moe,
        file: &GgufFile,
        layer: usize,
        n_embd: usize,
        x: &[f32],
        profile: &mut Profile,
    ) -> Vec<f32> {
        let route = moe.route_pure(x);
        // Copy out what the launches need, then release the borrow so `run_batch` can take
        // `&mut gpu`.
        let resident = gpu
            .as_ref()
            .and_then(|g| g.experts.get(layer))
            .and_then(|e| e.as_ref());
        let gu_p = resident.and_then(|e| e.gate_up.as_ref()).map(snapshot);
        let dn_p = resident.and_then(|e| e.down.as_ref()).map(snapshot);
        let (gate_cap, up_cap) = moe.caps();
        let ff = moe.ff();

        // Down resident, gate/up not: run the gate/up half on the host and the down half on
        // the card. Before per-tensor residency these layers ran entirely on the host.
        if let (None, Some(dn)) = (&gu_p, &dn_p) {
            let dn = dn.clone();
            let t = std::time::Instant::now();
            // The host gate/up half is two thirds of the work for these layers, so it is
            // threaded across the routed experts exactly like `forward_route`; running it
            // serially made the per-tensor path *slower* than the all-host one.
            let tensor = moe.gate_up_tensor();
            let acts: Vec<Vec<f32>> = std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(route.indices.len());
                for &e in &route.indices {
                    handles.push(scope.spawn(move || {
                        let mut gu = vec![0f32; tensor.out_dim];
                        let mut row = Vec::new();
                        let (mut gate, mut up, mut act) =
                            (vec![0f32; ff], vec![0f32; ff], vec![0f32; ff]);
                        tensor.row_matvec(file, e, x, &mut gu, &mut row);
                        gate.copy_from_slice(&gu[..ff]);
                        up.copy_from_slice(&gu[ff..]);
                        swiglu_softcapped_public(&gate, &up, gate_cap, up_cap, &mut act);
                        act
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().expect("gate/up worker panicked"))
                    .collect()
            });
            let weights: Vec<*mut c_void> = route
                .indices
                .iter()
                .map(|&e| unsafe { (dn.ptr as *mut u8).add(e * dn.slab) as *mut c_void })
                .collect();
            let inputs: Vec<&[f32]> = acts.iter().map(|a| a.as_slice()).collect();
            let downs = gpu
                .as_mut()
                .expect("gpu active")
                .run_batch(dn.handle, &weights, &inputs, n_embd);
            let mut acc = vec![0f32; n_embd];
            for (slot, &e) in route.indices.iter().enumerate() {
                let scale = dn.outer[e];
                let w = route.weights[slot];
                for (a, m) in acc
                    .iter_mut()
                    .zip(&downs[slot * n_embd..(slot + 1) * n_embd])
                {
                    *a += w * m * scale;
                }
            }
            profile.gpu_experts += t.elapsed().as_secs_f64();
            return acc;
        }

        if gu_p.is_none() || dn_p.is_none() {
            return moe.forward_route(file, x, &route);
        }
        let (gu, dn) = (gu_p.unwrap(), dn_p.unwrap());

        let t_batch = std::time::Instant::now();
        let weights: Vec<*mut c_void> = route
            .indices
            .iter()
            .map(|&e| unsafe { (gu.ptr as *mut u8).add(e * gu.slab) as *mut c_void })
            .collect();
        let inputs: Vec<&[f32]> = (0..route.indices.len()).map(|_| x).collect();
        let gate_up = gpu
            .as_mut()
            .expect("gpu active")
            .run_batch(gu.handle, &weights, &inputs, gu.rows);

        let mut acts: Vec<Vec<f32>> = Vec::with_capacity(route.indices.len());
        let (mut g, mut u, mut act) = (vec![0f32; ff], vec![0f32; ff], vec![0f32; ff]);
        for (slot, &expert) in route.indices.iter().enumerate() {
            let scale = gu.outer[expert];
            let row = &gate_up[slot * gu.rows..(slot + 1) * gu.rows];
            for (dst, src) in g.iter_mut().zip(&row[..ff]) {
                *dst = src * scale;
            }
            for (dst, src) in u.iter_mut().zip(&row[ff..]) {
                *dst = src * scale;
            }
            swiglu_softcapped_public(&g, &u, gate_cap, up_cap, &mut act);
            acts.push(act.clone());
        }

        let weights: Vec<*mut c_void> = route
            .indices
            .iter()
            .map(|&e| unsafe { (dn.ptr as *mut u8).add(e * dn.slab) as *mut c_void })
            .collect();
        let inputs: Vec<&[f32]> = acts.iter().map(|a| a.as_slice()).collect();
        let downs = gpu
            .as_mut()
            .expect("gpu active")
            .run_batch(dn.handle, &weights, &inputs, n_embd);
        profile.gpu_experts += t_batch.elapsed().as_secs_f64();

        let mut acc = vec![0f32; n_embd];
        for (slot, &expert) in route.indices.iter().enumerate() {
            let scale = dn.outer[expert];
            let w = route.weights[slot];
            let row = &downs[slot * n_embd..(slot + 1) * n_embd];
            for (a, m) in acc.iter_mut().zip(row) {
                *a += w * m * scale;
            }
        }
        acc
    }

    /// Prefill a token sequence in chunks of `chunk` positions, returning the logits of the
    /// last position.
    ///
    /// The 48 visits, the block snapshots and the message context are all per position, so
    /// a chunk is carried as one vector per position throughout; only the expert GEMVs and
    /// (on the card) the projections are shared, which is where the amortisation lives.
    pub fn prefill(&mut self, tokens: &[u32], chunk: usize) -> Vec<Vec<f32>> {
        let chunk = chunk.clamp(1, CHUNK_MAX);
        let mut logits = Vec::with_capacity(tokens.len());
        let mut at = 0;
        while at < tokens.len() {
            let end = (at + chunk).min(tokens.len());
            logits.extend(self.forward_chunk(&tokens[at..end]));
            at = end;
        }
        logits
    }

    /// One chunk of positions through the whole stack, returning the logits of every
    /// position in it (the head is cheap next to a visit, and per-position logits are what
    /// perplexity needs).
    fn forward_chunk(&mut self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let n = tokens.len();
        let base_pos = self.pos;
        let eps = self.cfg.rms_eps;
        let mut h: Vec<Vec<f32>> = tokens.iter().map(|&t| self.embed(t)).collect();
        let mut blocks: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n];

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
        let visit_of = |layer: usize, pass: usize| -> usize {
            if layer < start {
                layer
            } else if layer < end {
                start + pass * (end - start) + (layer - start)
            } else {
                start + passes * (end - start) + (layer - end)
            }
        };

        for layer in 0..start {
            self.chunk_visit(
                layer,
                &mut h,
                &mut blocks,
                None,
                visit_of(layer, 0),
                base_pos,
            );
        }
        let pre_h = h.clone();
        let pre_blocks = blocks.clone();
        let num_pre = blocks.first().map(|b| b.len()).unwrap_or(0);

        // Pass 0 is a plain run over the looped block, with no messages.
        for layer in start..end {
            self.chunk_visit(
                layer,
                &mut h,
                &mut blocks,
                None,
                visit_of(layer, 0),
                base_pos,
            );
        }

        for pass in 1..passes {
            let prev_h = h.clone();
            let prev_blocks: Vec<Vec<Vec<f32>>> = blocks
                .iter()
                .map(|b| b.iter().skip((num_pre + 1).min(b.len())).cloned().collect())
                .collect();
            h = pre_h.clone();
            blocks = pre_blocks.clone();
            for layer in start..end {
                self.chunk_visit(
                    layer,
                    &mut h,
                    &mut blocks,
                    Some((&prev_blocks, &prev_h)),
                    visit_of(layer, pass),
                    base_pos,
                );
            }
        }

        for layer in end..self.cfg.n_layers {
            self.chunk_visit(
                layer,
                &mut h,
                &mut blocks,
                None,
                visit_of(layer, 0),
                base_pos,
            );
        }
        self.pos = base_pos + n;

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let final_hidden =
                block_attention(&self.head.query, 1, &blocks[i], &h[i], eps).remove(0);
            let mut normed = final_hidden;
            rmsnorm(&mut normed, &self.head.norm, self.head.eps);
            out.push(self.head.logits(&self.file, &normed));
        }
        out
    }

    /// One visit over a chunk: attention half per position, one batched MoE, FFN half per
    /// position.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn chunk_visit(
        &mut self,
        layer: usize,
        h: &mut [Vec<f32>],
        blocks: &mut [Vec<Vec<f32>>],
        prev: Option<(&Vec<Vec<Vec<f32>>>, &Vec<Vec<f32>>)>,
        visit: usize,
        base_pos: usize,
    ) {
        let n = h.len();
        let t_visit = std::time::Instant::now();
        self.profile.visits += n;
        let mut residuals: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut moe_ins: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut msgs: Vec<Option<Vec<Vec<f32>>>> = Vec::with_capacity(n);
        let mut ins_all: Vec<AttnInputs> = Vec::with_capacity(n);
        for i in 0..n {
            // The log-scale uses the token's absolute position.
            self.pos = base_pos + i + 1;
            let msg = prev.map(|(pb, ph)| self.messages(layer, &pb[i], &ph[i], i));
            ins_all.push(self.attn_inputs(layer, &h[i], &blocks[i], msg.as_deref()));
            msgs.push(msg);
        }
        let mut proj_all = self.project_chunk(layer, &ins_all);
        for i in 0..n {
            let mut blk = std::mem::take(&mut blocks[i]);
            let proj = std::mem::take(&mut proj_all[i]);
            let (residual, moe_in) = self.attn_finish(
                layer,
                &h[i],
                &ins_all[i],
                proj,
                &mut blk,
                visit,
                msgs[i].as_deref(),
            );
            blocks[i] = blk;
            residuals.push(residual);
            moe_ins.push(moe_in);
        }

        let n_embd = self.cfg.n_embd;
        let t_moe = std::time::Instant::now();
        let moe_outs = Self::moe_chunk_parts(
            &mut self.gpu,
            &mut self.moes[layer],
            &self.file,
            layer,
            n_embd,
            &moe_ins,
            &mut self.profile,
        );
        self.profile.moe += t_moe.elapsed().as_secs_f64();
        for i in 0..n {
            h[i] = self.ffn_half(layer, &residuals[i], &moe_outs[i], visit);
        }
        self.profile.attention += t_visit.elapsed().as_secs_f64() - t_moe.elapsed().as_secs_f64();
    }

    /// A chunk of positions through the MoE of one layer.
    ///
    /// Positions that routed to the same expert are gathered, so either path pays the
    /// expert's dequantisation/weight read once per expert instead of once per position.
    #[allow(clippy::too_many_arguments)]
    fn moe_chunk_parts(
        gpu: &mut Option<Gpu>,
        moe: &mut Moe,
        file: &GgufFile,
        layer: usize,
        n_embd: usize,
        xs: &[Vec<f32>],
        profile: &mut Profile,
    ) -> Vec<Vec<f32>> {
        let mut outs = vec![vec![0f32; n_embd]; xs.len()];
        let trace = std::env::var_os("SCORIA_TRACE_CHUNK").is_some();
        let t_entry = std::time::Instant::now();
        // Gathering only pays when positions *share* experts: with 128 experts and 8 routed
        // per position, a chunk of C positions makes `C * 8` draws whose distinct count is
        // 128 * (1 - (1 - 8/128)^C). Measured: at C=1 the per-position path is faster
        // (1.63 vs 1.86 s/token), at C=8 the gathered path is much faster (0.70 vs 1.47).
        // So: one position is decode, anything larger gathers.
        if xs.len() == 1 {
            for (i, x) in xs.iter().enumerate() {
                outs[i] = Self::moe_forward_parts(gpu, moe, file, layer, n_embd, x, profile);
            }
            if trace {
                eprintln!(
                    "  chunk n=1 layer {layer}: {:.4}s (per-position)",
                    t_entry.elapsed().as_secs_f64()
                );
            }
            return outs;
        }
        let gpu_ready = gpu
            .as_ref()
            .and_then(|g| g.experts.get(layer))
            .and_then(|e| e.as_ref())
            .map(|e| e.complete())
            .unwrap_or(false);
        if !gpu_ready {
            if std::env::var_os("SCORIA_TRACE_CHUNK").is_some() {
                eprintln!(
                    "  chunk moe layer {layer}: HOST (resident={})",
                    gpu.as_ref().map(|g| g.experts.len()).unwrap_or(0)
                );
            }
            moe.forward_chunk(file, xs, &mut outs);
            if trace {
                eprintln!(
                    "  chunk n={} layer {layer}: {:.4}s HOST",
                    xs.len(),
                    t_entry.elapsed().as_secs_f64()
                );
            }
            return outs;
        }

        let t_batch = std::time::Instant::now();
        let (gate_cap, up_cap) = moe.caps();
        let ff = moe.ff();
        let e = gpu.as_ref().unwrap().experts[layer].as_ref().unwrap();
        let (gu_snap, dn_snap) = (
            snapshot(e.gate_up.as_ref().unwrap()),
            snapshot(e.down.as_ref().unwrap()),
        );
        let (gu_rows, gu_k) = (gu_snap.rows, e.gate_up.as_ref().unwrap().k);
        let (dn_rows, dn_k) = (dn_snap.rows, e.down.as_ref().unwrap().k);
        let gu_slab = gu_snap.slab;
        let dn_slab = dn_snap.slab;
        let gu_ptr = gu_snap.ptr;
        let dn_ptr = dn_snap.ptr;
        let gu_outer = gu_snap.outer.clone();
        let dn_outer = dn_snap.outer.clone();

        // Routing for every position, then a gather per expert.
        let routes: Vec<crate::inkling2_moe::Route> =
            xs.iter().map(|x| moe.route_pure(x)).collect();
        let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); gu_outer.len()];
        for (pos, route) in routes.iter().enumerate() {
            for (slot, &expert) in route.indices.iter().enumerate() {
                buckets[expert].push((pos, route.weights[slot]));
            }
        }
        let mut groups: Vec<(usize, usize, usize)> = Vec::new();
        let mut flat: Vec<f32> = Vec::new();
        let mut offset = 0usize;
        for (expert, hits) in buckets.iter().enumerate() {
            if hits.is_empty() {
                continue;
            }
            groups.push((expert, offset, hits.len()));
            for (pos, _) in hits {
                flat.extend_from_slice(&xs[*pos]);
            }
            offset += hits.len();
        }
        let gate_up = gpu.as_mut().unwrap().run_groups(
            gu_snap.handle,
            gu_ptr,
            gu_slab,
            &groups,
            &flat,
            gu_k,
            gu_rows,
        );

        // SwiGLU per (position, expert), with the outer scale applied where the CPU does.
        let mut per_position: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); xs.len()];
        let (mut g, mut u, mut act) = (vec![0f32; ff], vec![0f32; ff], vec![0f32; ff]);
        let mut base = 0usize;
        for (expert, hits) in buckets.iter().enumerate() {
            if hits.is_empty() {
                continue;
            }
            let scale = gu_outer[expert];
            for (which, (pos, weight)) in hits.iter().enumerate() {
                let row = &gate_up[(base + which) * gu_rows..(base + which + 1) * gu_rows];
                for (dst, src) in g.iter_mut().zip(&row[..ff]) {
                    *dst = src * scale;
                }
                for (dst, src) in u.iter_mut().zip(&row[ff..]) {
                    *dst = src * scale;
                }
                swiglu_softcapped_public(&g, &u, gate_cap, up_cap, &mut act);
                per_position[*pos].push((expert, {
                    let mut a = act.clone();
                    a.push(*weight);
                    a
                }));
            }
            base += hits.len();
        }

        // Down phase: gather the activations the same way.
        // Group the gathered activations in expert order.
        let mut by_expert: Vec<Vec<(usize, &[f32], f32)>> = vec![Vec::new(); gu_outer.len()];
        for (pos, entries) in per_position.iter().enumerate() {
            for (expert, payload) in entries {
                let weight = payload[ff];
                by_expert[*expert].push((pos, &payload[..ff], weight));
            }
        }
        let mut flat_acts: Vec<f32> = Vec::new();
        let mut offsets: Vec<(usize, usize, usize)> = Vec::new();
        // The second gather starts at zero again: `offset` is an index into the batch
        // buffers, and carrying it over from the gate/up gather writes the down outputs past
        // the region that is downloaded, so the caller reads back stale gate/up values.
        offset = 0;
        for (expert, hits) in by_expert.iter().enumerate() {
            if hits.is_empty() {
                continue;
            }
            offsets.push((expert, offset, hits.len()));
            for (_, act_slice, _) in hits {
                flat_acts.extend_from_slice(act_slice);
            }
            offset += hits.len();
        }
        let downs = gpu.as_mut().unwrap().run_groups(
            dn_snap.handle,
            dn_ptr,
            dn_slab,
            &offsets,
            &flat_acts,
            dn_k,
            dn_rows,
        );
        let mut base = 0usize;
        for (expert, hits) in by_expert.iter().enumerate() {
            if hits.is_empty() {
                continue;
            }
            let scale = dn_outer[expert];
            for (which, (pos, _, weight)) in hits.iter().enumerate() {
                let row = &downs[(base + which) * dn_rows..(base + which + 1) * dn_rows];
                for (a, m) in outs[*pos].iter_mut().zip(row) {
                    *a += weight * m * scale;
                }
            }
            base += hits.len();
        }
        profile.gpu_experts += t_batch.elapsed().as_secs_f64();
        if trace {
            eprintln!(
                "  chunk n={} layer {layer}: {:.4}s GATHERED",
                xs.len(),
                t_entry.elapsed().as_secs_f64()
            );
        }
        outs
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
    fn kv_ring_keeps_exactly_the_window_and_maps_absolute_positions() {
        // A window of 4 over 10 positions: positions 0..5 are evicted, and absolute
        // position `t` still reads back its own values through the modulo slot.
        let dim = 2;
        let mut kv = KvStore::new(Some(4), dim);
        for t in 0..10usize {
            kv.push(&[t as f32, -1.0], &[t as f32, 100.0]);
            assert_eq!(kv.count, t + 1);
            let expect_first = (t + 1).saturating_sub(4);
            assert_eq!(kv.first(), expect_first, "first after {t}");
            // Everything retained reads back its own value, nothing older is reachable.
            for u in expect_first..=t {
                let slot = kv.slot(u);
                assert_eq!(kv.k[slot * dim], u as f32, "k[{u}] after {t}");
                assert_eq!(kv.v[slot * dim], u as f32, "v[{u}] after {t}");
            }
            assert!(kv.k.len() <= 4 * dim, "capacity must not exceed the window");
        }
        assert_eq!(kv.count, 10);
        assert_eq!(kv.first(), 6);
        assert_eq!(kv.slot(6), 2);
        assert_eq!(kv.slot(9), 1);

        // An unbounded store keeps everything and never evicts.
        let mut full = KvStore::new(None, dim);
        for t in 0..10usize {
            full.push(&[t as f32, 0.0], &[0.0, t as f32]);
        }
        assert_eq!(full.first(), 0);
        assert_eq!(full.count, 10);
        assert_eq!(full.k.len(), 10 * dim);
    }

    #[test]
    fn kv_ring_grows_lazily_without_repacking() {
        // The first 16 slots are allocated up front, then doubled up to the window. While
        // `count <= cap <= window` the slot index equals the absolute position, so growth
        // is a resize and no position has to be moved.
        let dim = 1;
        let mut kv = KvStore::new(Some(64), dim);
        kv.push(&[0.0], &[0.0]);
        assert_eq!(kv.k.len(), 16);
        for t in 1..64usize {
            kv.push(&[t as f32], &[t as f32]);
        }
        assert_eq!(kv.k.len(), 64, "grown to the window, not beyond");
        for (t, value) in kv.k.iter().enumerate() {
            assert_eq!(*value, t as f32, "k[{t}] survived the growth");
        }
        // The 65th position overwrites slot 0 and nothing else.
        kv.push(&[64.0], &[64.0]);
        assert_eq!(kv.k[0], 64.0);
        assert_eq!(kv.first(), 1);
        assert_eq!(kv.k.len(), 64);
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
