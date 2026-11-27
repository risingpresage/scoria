use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

use scoria_gguf::{GgmlType, GgufError, GgufFile, ModelConfig, QuantCodec, TensorInfo};
use scoria_hip_rt::{DeviceBuf, HipError, HipResult, HipStream};
use scoria_kernels::infer_names;

use crate::{
    EngineContext, ExecutionGraph, KvSlotKey, LayerSpec, LinearKernel, MixerKind, Stage,
    TensorStore, TensorTier,
};

pub type LayerTap<'a> = &'a mut dyn FnMut(&str, &[f32]);

pub struct GpuTensor {
    pub id: u64,
    pub buf: DeviceBuf,
    pub ggml_type: GgmlType,
    pub dims: Vec<u64>,
}

pub struct VramTensorStore {
    tensors: Vec<GpuTensor>,
    by_name: HashMap<String, usize>,
    tiers: HashMap<u64, TensorTier>,
}

impl VramTensorStore {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            by_name: HashMap::new(),
            tiers: HashMap::new(),
        }
    }

    pub fn upload(&mut self, name: &str, info: &TensorInfo, bytes: &[u8]) -> HipResult<usize> {
        if let Some(&idx) = self.by_name.get(name) {
            return Ok(idx);
        }
        let mut buf = DeviceBuf::alloc(bytes.len())?;
        buf.copy_from_host(bytes)?;
        let id = self.tensors.len() as u64;
        self.tiers.insert(
            id,
            TensorTier::Vram {
                device_offset: 0,
                bytes: bytes.len(),
            },
        );
        self.tensors.push(GpuTensor {
            id,
            buf,
            ggml_type: info.ggml_type,
            dims: info.dims.clone(),
        });
        let idx = self.tensors.len() - 1;
        self.by_name.insert(name.to_string(), idx);
        Ok(idx)
    }

    pub fn get(&self, idx: usize) -> &GpuTensor {
        &self.tensors[idx]
    }
}

impl Default for VramTensorStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorStore for VramTensorStore {
    fn tier_of(&self, tensor_id: u64) -> Option<&TensorTier> {
        self.tiers.get(&tensor_id)
    }
}

pub struct ReferenceLinearKernel;

impl LinearKernel for ReferenceLinearKernel {
    fn name(&self) -> &'static str {
        "reference_hip_f32"
    }

    fn supports(&self, codec: QuantCodec) -> bool {
        matches!(
            codec,
            QuantCodec::F32
                | QuantCodec::F16
                | QuantCodec::Q5_0
                | QuantCodec::Q8_0
                | QuantCodec::Q4K
                | QuantCodec::Q6K
        )
    }
}

pub struct KvCache {
    slots: HashMap<KvSlotKey, (DeviceBuf, DeviceBuf)>,
    pub max_seq: usize,
    pub kv_dim: usize,
}

impl KvCache {
    pub fn new(n_layers: usize, max_seq: usize, kv_dim: usize) -> HipResult<Self> {
        let mut slots = HashMap::with_capacity(n_layers);
        let bytes = max_seq * kv_dim * std::mem::size_of::<f32>();
        for layer_id in 0..n_layers {
            let k_buf = DeviceBuf::alloc(bytes)?;
            let v_buf = DeviceBuf::alloc(bytes)?;
            slots.insert(
                KvSlotKey {
                    layer_id,
                    loop_pass: 0,
                },
                (k_buf, v_buf),
            );
        }
        Ok(Self {
            slots,
            max_seq,
            kv_dim,
        })
    }

    pub fn get_mut(&mut self, key: KvSlotKey) -> Option<(&mut DeviceBuf, &mut DeviceBuf)> {
        self.slots.get_mut(&key).map(|(k, v)| (k, v))
    }
}

#[derive(Debug)]
pub enum ModelLoadError {
    Gguf(GgufError),
    Hip(HipError),
    Unsupported(String),
}

impl std::fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelLoadError::Gguf(e) => write!(f, "{e}"),
            ModelLoadError::Hip(e) => write!(f, "{e}"),
            ModelLoadError::Unsupported(s) => write!(f, "unsupported model feature: {s}"),
        }
    }
}

impl std::error::Error for ModelLoadError {}

impl From<GgufError> for ModelLoadError {
    fn from(e: GgufError) -> Self {
        Self::Gguf(e)
    }
}

impl From<HipError> for ModelLoadError {
    fn from(e: HipError) -> Self {
        Self::Hip(e)
    }
}

struct LayerWeights {
    attn_norm: usize,
    attn_q: usize,
    attn_q_bias: Option<usize>,
    attn_k: usize,
    attn_k_bias: Option<usize>,
    attn_v: usize,
    attn_v_bias: Option<usize>,
    attn_output: usize,
    ffn_norm: usize,
    ffn_gate: usize,
    ffn_up: usize,
    ffn_down: usize,
}

struct ScratchBuffers {
    max_batch: usize,
    token_ids: DeviceBuf,
    hidden: DeviceBuf,
    normed: DeviceBuf,
    q: DeviceBuf,
    k: DeviceBuf,
    v: DeviceBuf,
    attn_out: DeviceBuf,
    proj_out: DeviceBuf,
    gate: DeviceBuf,
    up: DeviceBuf,
    ffn_out: DeviceBuf,
    logits: DeviceBuf,
}

impl ScratchBuffers {
    fn alloc(cfg: &ModelConfig, max_batch: usize) -> HipResult<Self> {
        let f32_sz = std::mem::size_of::<f32>();
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        Ok(Self {
            max_batch,
            token_ids: DeviceBuf::alloc(max_batch * std::mem::size_of::<u32>())?,
            hidden: DeviceBuf::alloc(max_batch * cfg.n_embd * f32_sz)?,
            normed: DeviceBuf::alloc(max_batch * cfg.n_embd * f32_sz)?,
            q: DeviceBuf::alloc(max_batch * q_dim * f32_sz)?,
            k: DeviceBuf::alloc(max_batch * kv_dim * f32_sz)?,
            v: DeviceBuf::alloc(max_batch * kv_dim * f32_sz)?,
            attn_out: DeviceBuf::alloc(max_batch * q_dim * f32_sz)?,
            proj_out: DeviceBuf::alloc(max_batch * cfg.n_embd * f32_sz)?,
            gate: DeviceBuf::alloc(max_batch * cfg.n_ff * f32_sz)?,
            up: DeviceBuf::alloc(max_batch * cfg.n_ff * f32_sz)?,
            ffn_out: DeviceBuf::alloc(max_batch * cfg.n_embd * f32_sz)?,
            logits: DeviceBuf::alloc(cfg.vocab_size * f32_sz)?,
        })
    }
}

pub struct Qwen2Model {
    pub config: ModelConfig,
    pub graph: ExecutionGraph,
    pub store: VramTensorStore,
    pub kv_cache: KvCache,
    pub pos: usize,
    stream: HipStream,
    token_embd: usize,
    output_norm: usize,
    output_weight: usize,
    layers: Vec<LayerWeights>,
    scratch: ScratchBuffers,
}

fn upload_required(
    gguf: &GgufFile,
    store: &mut VramTensorStore,
    name: &str,
) -> Result<usize, ModelLoadError> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| GgufError::MissingMetadata(format!("tensor `{name}`")))?;
    let bytes = gguf.tensor_bytes(info)?;
    Ok(store.upload(name, info, bytes)?)
}

fn upload_optional(
    gguf: &GgufFile,
    store: &mut VramTensorStore,
    name: &str,
) -> Result<Option<usize>, ModelLoadError> {
    let Some(info) = gguf.tensor(name) else {
        return Ok(None);
    };
    let bytes = gguf.tensor_bytes(info)?;
    Ok(Some(store.upload(name, info, bytes)?))
}

impl Qwen2Model {
    pub fn load(
        _ctx: &EngineContext,
        gguf: &GgufFile,
        max_seq: usize,
    ) -> Result<Self, ModelLoadError> {
        let config = ModelConfig::from_gguf(gguf)?;
        if config.architecture != "qwen2" {
            return Err(ModelLoadError::Unsupported(format!(
                "expected architecture `qwen2`, got `{}`",
                config.architecture
            )));
        }

        let linear_kernel = ReferenceLinearKernel;
        let mut store = VramTensorStore::new();
        let token_embd = upload_required(gguf, &mut store, "token_embd.weight")?;
        let output_norm = upload_required(gguf, &mut store, "output_norm.weight")?;
        let output_weight = if gguf.has_tensor("output.weight") {
            upload_required(gguf, &mut store, "output.weight")?
        } else {
            token_embd
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        let mut layer_specs = Vec::with_capacity(config.n_layers);
        for l in 0..config.n_layers {
            let lw = LayerWeights {
                attn_norm: upload_required(gguf, &mut store, &format!("blk.{l}.attn_norm.weight"))?,
                attn_q: upload_required(gguf, &mut store, &format!("blk.{l}.attn_q.weight"))?,
                attn_q_bias: upload_optional(gguf, &mut store, &format!("blk.{l}.attn_q.bias"))?,
                attn_k: upload_required(gguf, &mut store, &format!("blk.{l}.attn_k.weight"))?,
                attn_k_bias: upload_optional(gguf, &mut store, &format!("blk.{l}.attn_k.bias"))?,
                attn_v: upload_required(gguf, &mut store, &format!("blk.{l}.attn_v.weight"))?,
                attn_v_bias: upload_optional(gguf, &mut store, &format!("blk.{l}.attn_v.bias"))?,
                attn_output: upload_required(
                    gguf,
                    &mut store,
                    &format!("blk.{l}.attn_output.weight"),
                )?,
                ffn_norm: upload_required(gguf, &mut store, &format!("blk.{l}.ffn_norm.weight"))?,
                ffn_gate: upload_required(gguf, &mut store, &format!("blk.{l}.ffn_gate.weight"))?,
                ffn_up: upload_required(gguf, &mut store, &format!("blk.{l}.ffn_up.weight"))?,
                ffn_down: upload_required(gguf, &mut store, &format!("blk.{l}.ffn_down.weight"))?,
            };
            for &idx in &[
                lw.attn_q,
                lw.attn_k,
                lw.attn_v,
                lw.attn_output,
                lw.ffn_gate,
                lw.ffn_up,
                lw.ffn_down,
            ] {
                let codec = store.get(idx).ggml_type.to_codec();
                if !linear_kernel.supports(codec) {
                    return Err(ModelLoadError::Unsupported(format!(
                        "unsupported linear codec {codec:?}"
                    )));
                }
            }
            layers.push(lw);
            layer_specs.push(LayerSpec {
                layer_id: l,
                mixer: MixerKind::FullAttn,
                moe: None,
            });
        }

        let graph = ExecutionGraph {
            stages: vec![Stage::Dense(layer_specs)],
        };
        let kv_dim = config.n_kv_heads * config.head_dim;
        let kv_cache = KvCache::new(config.n_layers, max_seq, kv_dim)?;
        let scratch = ScratchBuffers::alloc(&config, 256)?;
        let stream = HipStream::new()?;

        Ok(Self {
            config,
            graph,
            store,
            kv_cache,
            pos: 0,
            stream,
            token_embd,
            output_norm,
            output_weight,
            layers,
            scratch,
        })
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    pub fn forward_tokens(
        &mut self,
        ctx: &EngineContext,
        tokens: &[u32],
        mut tap: Option<LayerTap<'_>>,
    ) -> HipResult<Vec<f32>> {
        assert!(!tokens.is_empty(), "forward_tokens requires >= 1 token");
        assert!(
            self.pos + tokens.len() <= self.kv_cache.max_seq,
            "sequence length {} exceeds max_seq {}",
            self.pos + tokens.len(),
            self.kv_cache.max_seq
        );

        let total_chunks = tokens.len().div_ceil(self.scratch.max_batch);
        let steps = self.graph.expand_steps();
        let q_dim = self.config.n_heads * self.config.head_dim;
        let kv_dim = self.config.n_kv_heads * self.config.head_dim;

        for (chunk_idx, chunk) in tokens.chunks(self.scratch.max_batch).enumerate() {
            let m = chunk.len();
            let start_pos = self.pos;

            let tok_bytes = unsafe {
                std::slice::from_raw_parts(
                    chunk.as_ptr() as *const u8,
                    std::mem::size_of_val(chunk),
                )
            };
            self.scratch.token_ids.copy_from_host_at(0, tok_bytes)?;

            let embd = self.store.get(self.token_embd);
            launch_embed(
                ctx,
                &self.stream,
                embd,
                self.scratch.token_ids.as_ptr(),
                self.scratch.hidden.as_ptr(),
                m,
                self.config.n_embd,
            )?;

            for step in &steps {
                let l = step.kv_key.layer_id;
                let lw = &self.layers[l];

                launch_rmsnorm(
                    ctx,
                    &self.stream,
                    self.scratch.hidden.as_ptr(),
                    self.store.get(lw.attn_norm).buf.as_ptr(),
                    self.scratch.normed.as_ptr(),
                    m,
                    self.config.n_embd,
                    self.config.rms_eps,
                )?;
                maybe_tap(
                    &mut tap,
                    &self.stream,
                    &format!("attn_norm-{l}"),
                    &self.scratch.normed,
                    m * self.config.n_embd,
                )?;

                let q_bias = lw
                    .attn_q_bias
                    .map(|i| self.store.get(i).buf.as_ptr())
                    .unwrap_or(ptr::null_mut());
                let k_bias = lw
                    .attn_k_bias
                    .map(|i| self.store.get(i).buf.as_ptr())
                    .unwrap_or(ptr::null_mut());
                let v_bias = lw
                    .attn_v_bias
                    .map(|i| self.store.get(i).buf.as_ptr())
                    .unwrap_or(ptr::null_mut());

                launch_linear(
                    ctx,
                    &self.stream,
                    self.store.get(lw.attn_q),
                    self.scratch.normed.as_ptr(),
                    q_bias,
                    self.scratch.q.as_ptr(),
                    m,
                    q_dim,
                    self.config.n_embd,
                )?;
                launch_linear(
                    ctx,
                    &self.stream,
                    self.store.get(lw.attn_k),
                    self.scratch.normed.as_ptr(),
                    k_bias,
                    self.scratch.k.as_ptr(),
                    m,
                    kv_dim,
                    self.config.n_embd,
                )?;
                launch_linear(
                    ctx,
                    &self.stream,
                    self.store.get(lw.attn_v),
                    self.scratch.normed.as_ptr(),
                    v_bias,
                    self.scratch.v.as_ptr(),
                    m,
                    kv_dim,
                    self.config.n_embd,
                )?;

                launch_rope_neox(
                    ctx,
                    &self.stream,
                    self.scratch.q.as_ptr(),
                    self.scratch.k.as_ptr(),
                    m,
                    start_pos,
                    self.config.n_heads,
                    self.config.n_kv_heads,
                    self.config.head_dim,
                    self.config.rope_freq_base,
                )?;
                maybe_tap(
                    &mut tap,
                    &self.stream,
                    &format!("Qcur-{l}"),
                    &self.scratch.q,
                    m * q_dim,
                )?;
                maybe_tap(
                    &mut tap,
                    &self.stream,
                    &format!("Kcur-{l}"),
                    &self.scratch.k,
                    m * kv_dim,
                )?;
                maybe_tap(
                    &mut tap,
                    &self.stream,
                    &format!("Vcur-{l}"),
                    &self.scratch.v,
                    m * kv_dim,
                )?;

                let (k_cache, v_cache) = self
                    .kv_cache
                    .get_mut(step.kv_key)
                    .expect("missing KV slot for layer");
                launch_kv_append(
                    ctx,
                    &self.stream,
                    self.scratch.k.as_ptr(),
                    self.scratch.v.as_ptr(),
                    k_cache.as_ptr(),
                    v_cache.as_ptr(),
                    m,
                    start_pos,
                    kv_dim,
                )?;

                launch_gqa_attn(
                    ctx,
                    &self.stream,
                    self.scratch.q.as_ptr(),
                    k_cache.as_ptr(),
                    v_cache.as_ptr(),
                    self.scratch.attn_out.as_ptr(),
                    m,
                    start_pos,
                    self.config.n_heads,
                    self.config.n_kv_heads,
                    self.config.head_dim,
                )?;

                launch_linear(
                    ctx,
                    &self.stream,
                    self.store.get(lw.attn_output),
                    self.scratch.attn_out.as_ptr(),
                    ptr::null_mut(),
                    self.scratch.proj_out.as_ptr(),
                    m,
                    self.config.n_embd,
                    q_dim,
                )?;

                launch_add_inplace(
                    ctx,
                    &self.stream,
                    self.scratch.hidden.as_ptr(),
                    self.scratch.proj_out.as_ptr(),
                    m * self.config.n_embd,
                )?;
                maybe_tap(
                    &mut tap,
                    &self.stream,
                    &format!("ffn_inp-{l}"),
                    &self.scratch.hidden,
                    m * self.config.n_embd,
                )?;

                launch_rmsnorm(
                    ctx,
                    &self.stream,
                    self.scratch.hidden.as_ptr(),
                    self.store.get(lw.ffn_norm).buf.as_ptr(),
                    self.scratch.normed.as_ptr(),
                    m,
                    self.config.n_embd,
                    self.config.rms_eps,
                )?;

                launch_linear(
                    ctx,
                    &self.stream,
                    self.store.get(lw.ffn_gate),
                    self.scratch.normed.as_ptr(),
                    ptr::null_mut(),
                    self.scratch.gate.as_ptr(),
                    m,
                    self.config.n_ff,
                    self.config.n_embd,
                )?;
                launch_linear(
                    ctx,
                    &self.stream,
                    self.store.get(lw.ffn_up),
                    self.scratch.normed.as_ptr(),
                    ptr::null_mut(),
                    self.scratch.up.as_ptr(),
                    m,
                    self.config.n_ff,
                    self.config.n_embd,
                )?;
                launch_silu_mul(
                    ctx,
                    &self.stream,
                    self.scratch.gate.as_ptr(),
                    self.scratch.up.as_ptr(),
                    self.scratch.gate.as_ptr(),
                    m * self.config.n_ff,
                )?;
                launch_linear(
                    ctx,
                    &self.stream,
                    self.store.get(lw.ffn_down),
                    self.scratch.gate.as_ptr(),
                    ptr::null_mut(),
                    self.scratch.ffn_out.as_ptr(),
                    m,
                    self.config.n_embd,
                    self.config.n_ff,
                )?;
                launch_add_inplace(
                    ctx,
                    &self.stream,
                    self.scratch.hidden.as_ptr(),
                    self.scratch.ffn_out.as_ptr(),
                    m * self.config.n_embd,
                )?;
                maybe_tap(
                    &mut tap,
                    &self.stream,
                    &format!("l_out-{l}"),
                    &self.scratch.hidden,
                    m * self.config.n_embd,
                )?;
            }

            self.pos += m;

            if chunk_idx + 1 == total_chunks {
                let last_byte_off = (m - 1) * self.config.n_embd * std::mem::size_of::<f32>();
                let last_hidden_ptr =
                    unsafe { (self.scratch.hidden.as_ptr() as *const u8).add(last_byte_off) }
                        as *mut c_void;

                launch_rmsnorm(
                    ctx,
                    &self.stream,
                    last_hidden_ptr,
                    self.store.get(self.output_norm).buf.as_ptr(),
                    self.scratch.normed.as_ptr(),
                    1,
                    self.config.n_embd,
                    self.config.rms_eps,
                )?;
                maybe_tap(
                    &mut tap,
                    &self.stream,
                    "result_norm",
                    &self.scratch.normed,
                    self.config.n_embd,
                )?;

                launch_linear(
                    ctx,
                    &self.stream,
                    self.store.get(self.output_weight),
                    self.scratch.normed.as_ptr(),
                    ptr::null_mut(),
                    self.scratch.logits.as_ptr(),
                    1,
                    self.config.vocab_size,
                    self.config.n_embd,
                )?;
            }
        }

        self.stream.synchronize()?;
        let mut logits = vec![0f32; self.config.vocab_size];
        let logits_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                logits.as_mut_ptr() as *mut u8,
                self.config.vocab_size * std::mem::size_of::<f32>(),
            )
        };
        self.scratch.logits.copy_to_host(logits_bytes)?;
        Ok(logits)
    }

    pub fn generate_greedy(
        &mut self,
        ctx: &EngineContext,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
        ignore_eos: bool,
        tap: Option<LayerTap<'_>>,
    ) -> HipResult<GenerationResult> {
        self.reset();
        let t_prefill0 = Instant::now();
        let prefill_logits = self.forward_tokens(ctx, prompt_tokens, tap)?;
        let prefill_ms = t_prefill0.elapsed().as_secs_f64() * 1000.0;

        let mut generated = Vec::with_capacity(max_new_tokens);
        if max_new_tokens == 0 {
            return Ok(GenerationResult {
                generated,
                prefill_logits,
                prefill_ms,
                decode_ms: 0.0,
            });
        }

        let mut next_tok = argmax_f32(&prefill_logits);
        generated.push(next_tok);

        let t_decode0 = Instant::now();
        for _ in 1..max_new_tokens {
            if !ignore_eos && eos_token_id == Some(next_tok) {
                break;
            }
            let logits = self.forward_tokens(ctx, &[next_tok], None)?;
            next_tok = argmax_f32(&logits);
            generated.push(next_tok);
        }
        let decode_ms = t_decode0.elapsed().as_secs_f64() * 1000.0;

        Ok(GenerationResult {
            generated,
            prefill_logits,
            prefill_ms,
            decode_ms,
        })
    }
}

pub struct GenerationResult {
    pub generated: Vec<u32>,
    pub prefill_logits: Vec<f32>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

pub fn argmax_f32(xs: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

pub fn top_k_logits(xs: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut pairs: Vec<(u32, f32)> = xs.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    pairs.truncate(k);
    pairs
}

fn maybe_tap(
    tap: &mut Option<LayerTap<'_>>,
    stream: &HipStream,
    name: &str,
    buf: &DeviceBuf,
    n_floats: usize,
) -> HipResult<()> {
    let Some(cb) = tap.as_deref_mut() else {
        return Ok(());
    };
    stream.synchronize()?;
    let mut host = vec![0f32; n_floats];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            host.as_mut_ptr() as *mut u8,
            n_floats * std::mem::size_of::<f32>(),
        )
    };
    buf.copy_to_host_at(0, bytes)?;
    cb(name, &host);
    Ok(())
}

fn launch_embed(
    ctx: &EngineContext,
    stream: &HipStream,
    table: &GpuTensor,
    token_ids_ptr: *mut c_void,
    out_ptr: *mut c_void,
    n_tokens: usize,
    n_embd: usize,
) -> HipResult<()> {
    let kname = match table.ggml_type {
        GgmlType::F32 => infer_names::EMBED_LOOKUP_F32,
        GgmlType::F16 => infer_names::EMBED_LOOKUP_F16,
        GgmlType::Q8_0 => infer_names::EMBED_LOOKUP_Q8_0,
        GgmlType::Q4K => infer_names::EMBED_LOOKUP_Q4_K,
        other => panic!("unsupported embedding type {:?}", other),
    };
    let func = ctx.infer_module.function(kname)?;
    let mut tbl = table.buf.as_ptr();
    let mut tids = token_ids_ptr;
    let mut out = out_ptr;
    let mut nt = n_tokens as u32;
    let mut ne = n_embd as u32;
    let mut args: [*mut c_void; 5] = [
        (&mut tbl as *mut *mut c_void) as *mut c_void,
        (&mut tids as *mut *mut c_void) as *mut c_void,
        (&mut out as *mut *mut c_void) as *mut c_void,
        (&mut nt as *mut u32) as *mut c_void,
        (&mut ne as *mut u32) as *mut c_void,
    ];
    unsafe {
        func.launch([n_tokens as u32, 1, 1], [256, 1, 1], 0, stream, &mut args)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_rmsnorm(
    ctx: &EngineContext,
    stream: &HipStream,
    x_ptr: *mut c_void,
    w_ptr: *mut c_void,
    out_ptr: *mut c_void,
    n_tokens: usize,
    n_embd: usize,
    eps: f32,
) -> HipResult<()> {
    let func = ctx.infer_module.function(infer_names::RMSNORM_F32)?;
    let mut x = x_ptr;
    let mut w = w_ptr;
    let mut out = out_ptr;
    let mut nt = n_tokens as u32;
    let mut ne = n_embd as u32;
    let mut e = eps;
    let mut args: [*mut c_void; 6] = [
        (&mut x as *mut *mut c_void) as *mut c_void,
        (&mut w as *mut *mut c_void) as *mut c_void,
        (&mut out as *mut *mut c_void) as *mut c_void,
        (&mut nt as *mut u32) as *mut c_void,
        (&mut ne as *mut u32) as *mut c_void,
        (&mut e as *mut f32) as *mut c_void,
    ];
    unsafe {
        func.launch([n_tokens as u32, 1, 1], [256, 1, 1], 0, stream, &mut args)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_linear(
    ctx: &EngineContext,
    stream: &HipStream,
    weight: &GpuTensor,
    x_ptr: *mut c_void,
    bias_ptr: *mut c_void,
    y_ptr: *mut c_void,
    m: usize,
    n: usize,
    k: usize,
) -> HipResult<()> {
    let kname = match weight.ggml_type {
        GgmlType::F32 => infer_names::LINEAR_F32,
        GgmlType::F16 => infer_names::LINEAR_F16,
        GgmlType::Q5_0 => infer_names::LINEAR_Q5_0,
        GgmlType::Q8_0 => infer_names::LINEAR_Q8_0,
        GgmlType::Q4K => infer_names::LINEAR_Q4_K,
        GgmlType::Q6K => infer_names::LINEAR_Q6_K,
        other => panic!("unsupported linear type {:?}", other),
    };
    let func = ctx.infer_module.function(kname)?;
    let waves_per_wg = 256u32 / ctx.wave_size;
    let grid_x = (n as u32).div_ceil(waves_per_wg);
    let mut w = weight.buf.as_ptr();
    let mut x = x_ptr;
    let mut b = bias_ptr;
    let mut y = y_ptr;
    let mut mu = m as u32;
    let mut nu = n as u32;
    let mut ku = k as u32;
    let mut args: [*mut c_void; 7] = [
        (&mut w as *mut *mut c_void) as *mut c_void,
        (&mut x as *mut *mut c_void) as *mut c_void,
        (&mut b as *mut *mut c_void) as *mut c_void,
        (&mut y as *mut *mut c_void) as *mut c_void,
        (&mut mu as *mut u32) as *mut c_void,
        (&mut nu as *mut u32) as *mut c_void,
        (&mut ku as *mut u32) as *mut c_void,
    ];
    unsafe {
        func.launch([grid_x, m as u32, 1], [256, 1, 1], 0, stream, &mut args)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_rope_neox(
    ctx: &EngineContext,
    stream: &HipStream,
    q_ptr: *mut c_void,
    k_ptr: *mut c_void,
    m: usize,
    start_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    freq_base: f32,
) -> HipResult<()> {
    let func = ctx.infer_module.function(infer_names::ROPE_NEOX_F32)?;
    let block_x = ((head_dim / 2) as u32).clamp(32, 256);
    let mut q = q_ptr;
    let mut k = k_ptr;
    let mut mu = m as u32;
    let mut sp = start_pos as u32;
    let mut nh = n_heads as u32;
    let mut nkvh = n_kv_heads as u32;
    let mut hd = head_dim as u32;
    let mut fb = freq_base;
    let mut args: [*mut c_void; 8] = [
        (&mut q as *mut *mut c_void) as *mut c_void,
        (&mut k as *mut *mut c_void) as *mut c_void,
        (&mut mu as *mut u32) as *mut c_void,
        (&mut sp as *mut u32) as *mut c_void,
        (&mut nh as *mut u32) as *mut c_void,
        (&mut nkvh as *mut u32) as *mut c_void,
        (&mut hd as *mut u32) as *mut c_void,
        (&mut fb as *mut f32) as *mut c_void,
    ];
    unsafe {
        func.launch(
            [m as u32, (n_heads + n_kv_heads) as u32, 1],
            [block_x, 1, 1],
            0,
            stream,
            &mut args,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_kv_append(
    ctx: &EngineContext,
    stream: &HipStream,
    k_in: *mut c_void,
    v_in: *mut c_void,
    k_cache: *mut c_void,
    v_cache: *mut c_void,
    m: usize,
    start_pos: usize,
    kv_dim: usize,
) -> HipResult<()> {
    let func = ctx
        .infer_module
        .function(infer_names::KV_CACHE_APPEND_F32)?;
    let total = (m * kv_dim) as u32;
    let mut ki = k_in;
    let mut vi = v_in;
    let mut kc = k_cache;
    let mut vc = v_cache;
    let mut mu = m as u32;
    let mut sp = start_pos as u32;
    let mut kd = kv_dim as u32;
    let mut args: [*mut c_void; 7] = [
        (&mut ki as *mut *mut c_void) as *mut c_void,
        (&mut vi as *mut *mut c_void) as *mut c_void,
        (&mut kc as *mut *mut c_void) as *mut c_void,
        (&mut vc as *mut *mut c_void) as *mut c_void,
        (&mut mu as *mut u32) as *mut c_void,
        (&mut sp as *mut u32) as *mut c_void,
        (&mut kd as *mut u32) as *mut c_void,
    ];
    unsafe {
        func.launch(
            [total.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            stream,
            &mut args,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_gqa_attn(
    ctx: &EngineContext,
    stream: &HipStream,
    q_ptr: *mut c_void,
    k_cache: *mut c_void,
    v_cache: *mut c_void,
    out_ptr: *mut c_void,
    m: usize,
    start_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> HipResult<()> {
    let func = ctx.infer_module.function(infer_names::GQA_ATTN_F32)?;
    let block_x = if head_dim <= 128 { 128 } else { 256 };
    let mut q = q_ptr;
    let mut kc = k_cache;
    let mut vc = v_cache;
    let mut out = out_ptr;
    let mut mu = m as u32;
    let mut sp = start_pos as u32;
    let mut nh = n_heads as u32;
    let mut nkvh = n_kv_heads as u32;
    let mut hd = head_dim as u32;
    let mut args: [*mut c_void; 9] = [
        (&mut q as *mut *mut c_void) as *mut c_void,
        (&mut kc as *mut *mut c_void) as *mut c_void,
        (&mut vc as *mut *mut c_void) as *mut c_void,
        (&mut out as *mut *mut c_void) as *mut c_void,
        (&mut mu as *mut u32) as *mut c_void,
        (&mut sp as *mut u32) as *mut c_void,
        (&mut nh as *mut u32) as *mut c_void,
        (&mut nkvh as *mut u32) as *mut c_void,
        (&mut hd as *mut u32) as *mut c_void,
    ];
    unsafe {
        func.launch(
            [n_heads as u32, m as u32, 1],
            [block_x, 1, 1],
            0,
            stream,
            &mut args,
        )?;
    }
    Ok(())
}

fn launch_silu_mul(
    ctx: &EngineContext,
    stream: &HipStream,
    gate_ptr: *mut c_void,
    up_ptr: *mut c_void,
    out_ptr: *mut c_void,
    n: usize,
) -> HipResult<()> {
    let func = ctx.infer_module.function(infer_names::SILU_MUL_F32)?;
    let mut g = gate_ptr;
    let mut u = up_ptr;
    let mut o = out_ptr;
    let mut nu = n as u32;
    let mut args: [*mut c_void; 4] = [
        (&mut g as *mut *mut c_void) as *mut c_void,
        (&mut u as *mut *mut c_void) as *mut c_void,
        (&mut o as *mut *mut c_void) as *mut c_void,
        (&mut nu as *mut u32) as *mut c_void,
    ];
    unsafe {
        func.launch(
            [(n as u32).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            stream,
            &mut args,
        )?;
    }
    Ok(())
}

fn launch_add_inplace(
    ctx: &EngineContext,
    stream: &HipStream,
    dst_ptr: *mut c_void,
    src_ptr: *mut c_void,
    n: usize,
) -> HipResult<()> {
    let func = ctx.infer_module.function(infer_names::ADD_INPLACE_F32)?;
    let mut d = dst_ptr;
    let mut s = src_ptr;
    let mut nu = n as u32;
    let mut args: [*mut c_void; 3] = [
        (&mut d as *mut *mut c_void) as *mut c_void,
        (&mut s as *mut *mut c_void) as *mut c_void,
        (&mut nu as *mut u32) as *mut c_void,
    ];
    unsafe {
        func.launch(
            [(n as u32).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            stream,
            &mut args,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scoria_gguf::dequant::dequant_row;
    use std::path::Path;

    fn to_bytes<T>(slice: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
        }
    }

    fn to_bytes_mut<T>(slice: &mut [T]) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                slice.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(slice),
            )
        }
    }

    #[test]
    fn gpu_kernels_parity_against_cpu() {
        let ctx = EngineContext::init(0).expect("EngineContext::init(0)");
        let stream = HipStream::new().expect("HipStream::new");

        // 1. rmsnorm_f32
        let n_embd = 64usize;
        let x_in: Vec<f32> = (0..n_embd).map(|i| ((i as f32) - 31.5) * 0.05).collect();
        let w_in: Vec<f32> = (0..n_embd).map(|i| 0.8 + (i as f32) * 0.01).collect();
        let eps = 1e-6f32;
        let ss: f32 = x_in.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / n_embd as f32 + eps).sqrt();
        let cpu_norm: Vec<f32> = x_in.iter().zip(&w_in).map(|(&x, &w)| x * inv * w).collect();

        let mut d_x = DeviceBuf::alloc(n_embd * 4).unwrap();
        let mut d_w = DeviceBuf::alloc(n_embd * 4).unwrap();
        let d_out = DeviceBuf::alloc(n_embd * 4).unwrap();
        d_x.copy_from_host(to_bytes(&x_in)).unwrap();
        d_w.copy_from_host(to_bytes(&w_in)).unwrap();
        launch_rmsnorm(
            &ctx,
            &stream,
            d_x.as_ptr(),
            d_w.as_ptr(),
            d_out.as_ptr(),
            1,
            n_embd,
            eps,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let mut gpu_norm = vec![0f32; n_embd];
        d_out.copy_to_host(to_bytes_mut(&mut gpu_norm)).unwrap();
        let max_diff_norm = cpu_norm
            .iter()
            .zip(&gpu_norm)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff_norm < 1e-5, "rmsnorm max_diff={max_diff_norm}");

        // 2. Real sliced rows from Qwen2.5-0.5B-Instruct-Q4_K_M.gguf (Q5_0, Q8_0, Q4_K, Q6_K)
        let model_path = "/home/noorm/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";
        if Path::new(model_path).exists() {
            let gguf = GgufFile::open(model_path).expect("open 0.5B GGUF");
            let test_tensors = [
                "blk.0.attn_q.weight",   // Q5_0 (896, 896)
                "blk.0.attn_v.weight",   // Q8_0 (896, 128)
                "blk.0.ffn_down.weight", // Q6_K (4864, 896)
                "blk.12.ffn_gate.weight",
            ];
            for tname in test_tensors {
                let Some(info) = gguf.tensor(tname) else {
                    continue;
                };
                let k = info.dims[0] as usize;
                let n_rows = (info.dims[1] as usize).min(16);
                let row_bytes = info.ggml_type.tensor_byte_len(k).unwrap();
                let raw = &gguf.tensor_bytes(info).unwrap()[..n_rows * row_bytes];
                let x_vec: Vec<f32> = (0..k).map(|i| ((i % 29) as f32 - 14.0) * 0.03).collect();

                let mut cpu_y = vec![0f32; n_rows];
                let mut row_f32 = vec![0f32; k];
                for r in 0..n_rows {
                    dequant_row(
                        info.ggml_type,
                        &raw[r * row_bytes..(r + 1) * row_bytes],
                        &mut row_f32,
                    )
                    .unwrap();
                    cpu_y[r] = row_f32.iter().zip(&x_vec).map(|(a, b)| a * b).sum();
                }

                let mut store = VramTensorStore::new();
                let mut sub_info = info.clone();
                sub_info.dims = vec![k as u64, n_rows as u64];
                let tid = store.upload(tname, &sub_info, raw).unwrap();
                let mut d_in = DeviceBuf::alloc(k * 4).unwrap();
                let d_y = DeviceBuf::alloc(n_rows * 4).unwrap();
                d_in.copy_from_host(to_bytes(&x_vec)).unwrap();
                launch_linear(
                    &ctx,
                    &stream,
                    store.get(tid),
                    d_in.as_ptr(),
                    ptr::null_mut(),
                    d_y.as_ptr(),
                    1,
                    n_rows,
                    k,
                )
                .unwrap();
                stream.synchronize().unwrap();
                let mut gpu_y = vec![0f32; n_rows];
                d_y.copy_to_host(to_bytes_mut(&mut gpu_y)).unwrap();
                let max_diff = cpu_y
                    .iter()
                    .zip(&gpu_y)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_diff < 1e-4,
                    "linear parity failed for {tname} ({:?}): max_diff={max_diff}",
                    info.ggml_type
                );
            }
        }
    }
}
