use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

use scoria_gguf::{GgmlType, GgufError, GgufFile, ModelConfig, QuantCodec, TensorInfo};
use scoria_hip_rt::{DeviceBuf, HipError, HipEvent, HipGraphExec, HipResult, HipStream};

use crate::{
    EngineContext, ExecutionGraph, KvSlotKey, LayerSpec, LinearKernel, MixerKind, Stage,
    TensorStore, TensorTier,
};

pub type LayerTap<'a> = &'a mut dyn FnMut(&str, &[f32]);

trait KernelArg {
    fn as_arg_ptr(&mut self) -> *mut c_void;
}

impl KernelArg for *mut c_void {
    #[inline(always)]
    fn as_arg_ptr(&mut self) -> *mut c_void {
        (self as *mut *mut c_void) as *mut c_void
    }
}

impl KernelArg for u32 {
    #[inline(always)]
    fn as_arg_ptr(&mut self) -> *mut c_void {
        (self as *mut u32) as *mut c_void
    }
}

impl KernelArg for f32 {
    #[inline(always)]
    fn as_arg_ptr(&mut self) -> *mut c_void {
        (self as *mut f32) as *mut c_void
    }
}

macro_rules! hip_launch {
    ($ctx:expr, $stream:expr, $handle:expr, $grid:expr, $block:expr; $($val:expr),+ $(,)?) => {{
        hip_launch!(@step $ctx, $stream, $handle, $grid, $block, 0u32; []; $($val),+)
    }};
    (@step $ctx:expr, $stream:expr, $handle:expr, $grid:expr, $block:expr, $smem:expr; [$($ptr:ident),*]; $head:expr, $($tail:expr),+) => {{
        let mut arg = $head;
        let ptr = KernelArg::as_arg_ptr(&mut arg);
        hip_launch!(@step $ctx, $stream, $handle, $grid, $block, $smem; [$($ptr,)* ptr]; $($tail),+)
    }};
    (@step $ctx:expr, $stream:expr, $handle:expr, $grid:expr, $block:expr, $smem:expr; [$($ptr:ident),*]; $last:expr) => {{
        let mut arg = $last;
        let ptr = KernelArg::as_arg_ptr(&mut arg);
        let mut args = [$($ptr,)* ptr];
        unsafe {
            $ctx.infer_module.launch_handle(
                $handle,
                $grid,
                $block,
                $smem,
                $stream,
                &mut args,
            )
        }
    }};
}

/// Like [`hip_launch!`] but with an explicit dynamic shared-memory size.
macro_rules! hip_launch_smem {
    ($ctx:expr, $stream:expr, $handle:expr, $grid:expr, $block:expr, $smem:expr; $($val:expr),+ $(,)?) => {{
        hip_launch!(@step $ctx, $stream, $handle, $grid, $block, $smem; []; $($val),+)
    }};
}

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
    attn_q_norm: Option<usize>,
    attn_k_norm: Option<usize>,
    attn_output: usize,
    ffn_norm: usize,
    ffn_gate: usize,
    ffn_up: usize,
    ffn_down: usize,
}

struct ScratchBuffers {
    max_batch: usize,
    token_ids: DeviceBuf,
    pos: DeviceBuf,
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
    argmax_vals: DeviceBuf,
    argmax_idxs: DeviceBuf,
    argmax_out: DeviceBuf,
}

impl ScratchBuffers {
    fn alloc(cfg: &ModelConfig, max_batch: usize) -> HipResult<Self> {
        let f32_sz = std::mem::size_of::<f32>();
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        Ok(Self {
            max_batch,
            token_ids: DeviceBuf::alloc(max_batch * std::mem::size_of::<u32>())?,
            pos: DeviceBuf::alloc(std::mem::size_of::<u32>())?,
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
            argmax_vals: DeviceBuf::alloc(64 * f32_sz)?,
            argmax_idxs: DeviceBuf::alloc(64 * std::mem::size_of::<u32>())?,
            argmax_out: DeviceBuf::alloc(std::mem::size_of::<u32>())?,
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
        if config.architecture != "qwen2" && config.architecture != "qwen3" {
            return Err(ModelLoadError::Unsupported(format!(
                "expected architecture `qwen2` or `qwen3`, got `{}`",
                config.architecture
            )));
        }
        let is_qwen3 = config.architecture == "qwen3";

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
                attn_q_norm: upload_optional(
                    gguf,
                    &mut store,
                    &format!("blk.{l}.attn_q_norm.weight"),
                )?,
                attn_k_norm: upload_optional(
                    gguf,
                    &mut store,
                    &format!("blk.{l}.attn_k_norm.weight"),
                )?,
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
            if is_qwen3 && (lw.attn_q_norm.is_none() || lw.attn_k_norm.is_none()) {
                return Err(ModelLoadError::Unsupported(format!(
                    "qwen3 layer {l} is missing attn_q_norm/attn_k_norm"
                )));
            }
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
        tap: Option<LayerTap<'_>>,
    ) -> HipResult<Vec<f32>> {
        self.forward_tokens_to_device(ctx, tokens, tap)?;
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

    fn forward_tokens_to_device(
        &mut self,
        ctx: &EngineContext,
        tokens: &[u32],
        mut tap: Option<LayerTap<'_>>,
    ) -> HipResult<()> {
        assert!(!tokens.is_empty(), "forward_tokens requires >= 1 token");
        assert!(
            self.pos + tokens.len() <= self.kv_cache.max_seq,
            "sequence length {} exceeds max_seq {}",
            self.pos + tokens.len(),
            self.kv_cache.max_seq
        );

        let total_chunks = tokens.len().div_ceil(self.scratch.max_batch);
        for (chunk_idx, chunk) in tokens.chunks(self.scratch.max_batch).enumerate() {
            let m = chunk.len();
            let tok_bytes = unsafe {
                std::slice::from_raw_parts(
                    chunk.as_ptr() as *const u8,
                    std::mem::size_of_val(chunk),
                )
            };
            self.scratch.token_ids.copy_from_host_at(0, tok_bytes)?;
            self.write_device_pos()?;
            let is_last_chunk = chunk_idx + 1 == total_chunks;
            self.run_layers(ctx, m, is_last_chunk, &mut tap)?;
            self.pos += m;
        }
        Ok(())
    }

    /// Publishes the host-side sequence position into the device slot read by the
    /// RoPE / KV-append / attention kernels, so those launches can be replayed from a
    /// captured graph with no host round trip.
    fn write_device_pos(&mut self) -> HipResult<()> {
        let bytes = (self.pos as u32).to_le_bytes();
        self.scratch.pos.copy_from_host_at(0, &bytes)
    }

    /// One forward pass over `m` tokens already staged in `scratch.token_ids` and
    /// `scratch.pos`. Free of host synchronisation, so it is capture-safe.
    fn run_layers(
        &mut self,
        ctx: &EngineContext,
        m: usize,
        is_last_chunk: bool,
        tap: &mut Option<LayerTap<'_>>,
    ) -> HipResult<()> {
        let steps = self.graph.expand_steps();
        let q_dim = self.config.n_heads * self.config.head_dim;
        let kv_dim = self.config.n_kv_heads * self.config.head_dim;

        {
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

            let n_steps = steps.len();
            for (step_idx, step) in steps.iter().enumerate() {
                let l = step.kv_key.layer_id;
                let lw = &self.layers[l];

                if step_idx == 0 {
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
                }
                maybe_tap(
                    tap,
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

                if let (Some(qn), Some(kn)) = (lw.attn_q_norm, lw.attn_k_norm) {
                    launch_qk_rmsnorm(
                        ctx,
                        &self.stream,
                        self.scratch.q.as_ptr(),
                        self.scratch.k.as_ptr(),
                        self.store.get(qn).buf.as_ptr(),
                        self.store.get(kn).buf.as_ptr(),
                        m,
                        self.config.n_heads,
                        self.config.n_kv_heads,
                        self.config.head_dim,
                        self.config.rms_eps,
                    )?;
                }

                launch_rope_neox(
                    ctx,
                    &self.stream,
                    self.scratch.q.as_ptr(),
                    self.scratch.k.as_ptr(),
                    self.scratch.pos.as_ptr(),
                    m,
                    self.config.n_heads,
                    self.config.n_kv_heads,
                    self.config.head_dim,
                    self.config.rope_freq_base,
                )?;
                maybe_tap(
                    tap,
                    &self.stream,
                    &format!("Qcur-{l}"),
                    &self.scratch.q,
                    m * q_dim,
                )?;
                maybe_tap(
                    tap,
                    &self.stream,
                    &format!("Kcur-{l}"),
                    &self.scratch.k,
                    m * kv_dim,
                )?;
                maybe_tap(
                    tap,
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
                    self.scratch.pos.as_ptr(),
                    m,
                    kv_dim,
                )?;

                launch_gqa_attn(
                    ctx,
                    &self.stream,
                    self.scratch.q.as_ptr(),
                    k_cache.as_ptr(),
                    v_cache.as_ptr(),
                    self.scratch.attn_out.as_ptr(),
                    self.scratch.pos.as_ptr(),
                    m,
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

                launch_add_rmsnorm(
                    ctx,
                    &self.stream,
                    self.scratch.hidden.as_ptr(),
                    self.scratch.proj_out.as_ptr(),
                    self.store.get(lw.ffn_norm).buf.as_ptr(),
                    self.scratch.normed.as_ptr(),
                    m,
                    self.config.n_embd,
                    self.config.rms_eps,
                )?;
                maybe_tap(
                    tap,
                    &self.stream,
                    &format!("ffn_inp-{l}"),
                    &self.scratch.hidden,
                    m * self.config.n_embd,
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

                if step_idx + 1 < n_steps {
                    let next_l = steps[step_idx + 1].kv_key.layer_id;
                    launch_add_rmsnorm(
                        ctx,
                        &self.stream,
                        self.scratch.hidden.as_ptr(),
                        self.scratch.ffn_out.as_ptr(),
                        self.store.get(self.layers[next_l].attn_norm).buf.as_ptr(),
                        self.scratch.normed.as_ptr(),
                        m,
                        self.config.n_embd,
                        self.config.rms_eps,
                    )?;
                    maybe_tap(
                        tap,
                        &self.stream,
                        &format!("l_out-{l}"),
                        &self.scratch.hidden,
                        m * self.config.n_embd,
                    )?;
                } else if is_last_chunk && m == 1 {
                    launch_add_rmsnorm(
                        ctx,
                        &self.stream,
                        self.scratch.hidden.as_ptr(),
                        self.scratch.ffn_out.as_ptr(),
                        self.store.get(self.output_norm).buf.as_ptr(),
                        self.scratch.normed.as_ptr(),
                        1,
                        self.config.n_embd,
                        self.config.rms_eps,
                    )?;
                    maybe_tap(
                        tap,
                        &self.stream,
                        &format!("l_out-{l}"),
                        &self.scratch.hidden,
                        m * self.config.n_embd,
                    )?;
                } else {
                    launch_add_inplace(
                        ctx,
                        &self.stream,
                        self.scratch.hidden.as_ptr(),
                        self.scratch.ffn_out.as_ptr(),
                        m * self.config.n_embd,
                    )?;
                    maybe_tap(
                        tap,
                        &self.stream,
                        &format!("l_out-{l}"),
                        &self.scratch.hidden,
                        m * self.config.n_embd,
                    )?;
                }
            }

            if is_last_chunk {
                if m > 1 {
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
                }
                maybe_tap(
                    tap,
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

        Ok(())
    }

    /// Two-stage device argmax over `scratch.logits` into `scratch.argmax_out`.
    /// Capture-safe: no host synchronisation.
    fn launch_argmax(&self, ctx: &EngineContext) -> HipResult<()> {
        hip_launch!(
            ctx,
            &self.stream,
            ctx.infer.argmax_stage1_f32,
            [64, 1, 1],
            [256, 1, 1];
            self.scratch.logits.as_ptr(),
            self.scratch.argmax_vals.as_ptr(),
            self.scratch.argmax_idxs.as_ptr(),
            self.config.vocab_size as u32
        )?;
        hip_launch!(
            ctx,
            &self.stream,
            ctx.infer.argmax_stage2_f32,
            [1, 1, 1],
            [64, 1, 1];
            self.scratch.argmax_vals.as_ptr(),
            self.scratch.argmax_idxs.as_ptr(),
            self.scratch.argmax_out.as_ptr(),
            64u32
        )
    }

    /// Closes the decode loop on-device: sample -> `token_ids`, `pos += 1`.
    fn launch_decode_advance(&self, ctx: &EngineContext) -> HipResult<()> {
        hip_launch!(
            ctx,
            &self.stream,
            ctx.infer.decode_advance_u32,
            [1, 1, 1],
            [1, 1, 1];
            self.scratch.argmax_out.as_ptr(),
            self.scratch.token_ids.as_ptr(),
            self.scratch.pos.as_ptr()
        )
    }

    fn argmax_device_logits(&self, ctx: &EngineContext) -> HipResult<u32> {
        self.launch_argmax(ctx)?;
        self.stream.synchronize()?;
        let mut tok_bytes = [0u8; 4];
        self.scratch.argmax_out.copy_to_host(&mut tok_bytes)?;
        Ok(u32::from_le_bytes(tok_bytes))
    }

    /// Captures one single-token decode step (layers + argmax + advance) as a HIP
    /// graph. Requires `scratch.token_ids` and the device position to already hold
    /// the first decode input; `decode_graph_next` then feeds it back each replay.
    pub fn capture_decode_graph(&mut self, ctx: &EngineContext) -> HipResult<HipGraphExec> {
        self.write_device_pos()?;
        self.stream.synchronize()?;
        self.stream.begin_capture()?;
        let mut no_tap: Option<LayerTap<'_>> = None;
        let captured = self
            .run_layers(ctx, 1, true, &mut no_tap)
            .and_then(|()| self.launch_argmax(ctx))
            .and_then(|()| self.launch_decode_advance(ctx));
        let graph = self.stream.end_capture()?;
        captured?;
        graph.instantiate()
    }

    /// Seeds the on-device decode input slot (token) so the first graph replay reads it.
    pub fn seed_decode_token(&mut self, _ctx: &EngineContext, token: u32) -> HipResult<()> {
        self.scratch
            .token_ids
            .copy_from_host_at(0, &token.to_le_bytes())
    }

    /// Replays the captured decode graph and returns the sampled token.
    pub fn decode_graph_next(
        &mut self,
        graph: &HipGraphExec,
        max_seq: usize,
    ) -> HipResult<Option<u32>> {
        if self.pos + 1 > max_seq {
            return Ok(None);
        }
        graph.launch(&self.stream)?;
        self.stream.synchronize()?;
        let mut tok_bytes = [0u8; 4];
        self.scratch.argmax_out.copy_to_host(&mut tok_bytes)?;
        self.pos += 1;
        Ok(Some(u32::from_le_bytes(tok_bytes)))
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
        if max_new_tokens > 1 {
            // Seed the on-device input slot with the prefill sample, then replay one
            // captured decode step per token.
            self.scratch
                .token_ids
                .copy_from_host_at(0, &next_tok.to_le_bytes())?;
            let graph = self.capture_decode_graph(ctx)?;
            while generated.len() < max_new_tokens {
                if !ignore_eos && eos_token_id == Some(next_tok) {
                    break;
                }
                let Some(tok) = self.decode_graph_next(&graph, self.kv_cache.max_seq)? else {
                    break;
                };
                next_tok = tok;
                generated.push(next_tok);
            }
        }
        let decode_ms = t_decode0.elapsed().as_secs_f64() * 1000.0;

        Ok(GenerationResult {
            generated,
            prefill_logits,
            prefill_ms,
            decode_ms,
        })
    }

    pub fn profile_decode_step(
        &mut self,
        ctx: &EngineContext,
        token: u32,
    ) -> HipResult<DecodeProfile> {
        let mut prof = DecodeProfile::default();
        let t_wall0 = Instant::now();
        let q_dim = self.config.n_heads * self.config.head_dim;
        let kv_dim = self.config.n_kv_heads * self.config.head_dim;

        let ev = || -> HipResult<(HipEvent, HipEvent)> { Ok((HipEvent::new()?, HipEvent::new()?)) };
        macro_rules! timed {
            ($field:ident, $body:expr) => {{
                let (e0, e1) = ev()?;
                e0.record(&self.stream)?;
                $body?;
                e1.record(&self.stream)?;
                e1.synchronize()?;
                prof.$field += HipEvent::elapsed_ms(&e0, &e1)? as f64;
            }};
        }

        let tok_bytes = token.to_le_bytes();
        self.scratch.token_ids.copy_from_host_at(0, &tok_bytes)?;
        self.write_device_pos()?;
        let embd = self.store.get(self.token_embd);
        timed!(
            other_ms,
            launch_embed(
                ctx,
                &self.stream,
                embd,
                self.scratch.token_ids.as_ptr(),
                self.scratch.hidden.as_ptr(),
                1,
                self.config.n_embd,
            )
        );

        let steps = self.graph.expand_steps();
        let n_steps = steps.len();
        for (step_idx, step) in steps.iter().enumerate() {
            let l = step.kv_key.layer_id;
            let lw = &self.layers[l];

            if step_idx == 0 {
                timed!(
                    rmsnorm_ms,
                    launch_rmsnorm(
                        ctx,
                        &self.stream,
                        self.scratch.hidden.as_ptr(),
                        self.store.get(lw.attn_norm).buf.as_ptr(),
                        self.scratch.normed.as_ptr(),
                        1,
                        self.config.n_embd,
                        self.config.rms_eps,
                    )
                );
            }

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

            for (w_idx, bias_p, out_p, out_dim) in [
                (lw.attn_q, q_bias, self.scratch.q.as_ptr(), q_dim),
                (lw.attn_k, k_bias, self.scratch.k.as_ptr(), kv_dim),
                (lw.attn_v, v_bias, self.scratch.v.as_ptr(), kv_dim),
            ] {
                let wt = self.store.get(w_idx);
                let is_q6 = wt.ggml_type == GgmlType::Q6K;
                if is_q6 {
                    timed!(
                        q6k_ms,
                        launch_linear(
                            ctx,
                            &self.stream,
                            wt,
                            self.scratch.normed.as_ptr(),
                            bias_p,
                            out_p,
                            1,
                            out_dim,
                            self.config.n_embd,
                        )
                    );
                } else {
                    timed!(
                        gemv_other_ms,
                        launch_linear(
                            ctx,
                            &self.stream,
                            wt,
                            self.scratch.normed.as_ptr(),
                            bias_p,
                            out_p,
                            1,
                            out_dim,
                            self.config.n_embd,
                        )
                    );
                }
            }

            if let (Some(qn), Some(kn)) = (lw.attn_q_norm, lw.attn_k_norm) {
                timed!(
                    other_ms,
                    launch_qk_rmsnorm(
                        ctx,
                        &self.stream,
                        self.scratch.q.as_ptr(),
                        self.scratch.k.as_ptr(),
                        self.store.get(qn).buf.as_ptr(),
                        self.store.get(kn).buf.as_ptr(),
                        1,
                        self.config.n_heads,
                        self.config.n_kv_heads,
                        self.config.head_dim,
                        self.config.rms_eps,
                    )
                );
            }

            timed!(
                other_ms,
                launch_rope_neox(
                    ctx,
                    &self.stream,
                    self.scratch.q.as_ptr(),
                    self.scratch.k.as_ptr(),
                    self.scratch.pos.as_ptr(),
                    1,
                    self.config.n_heads,
                    self.config.n_kv_heads,
                    self.config.head_dim,
                    self.config.rope_freq_base,
                )
            );

            let (k_cache, v_cache) = self.kv_cache.get_mut(step.kv_key).unwrap();
            timed!(
                other_ms,
                launch_kv_append(
                    ctx,
                    &self.stream,
                    self.scratch.k.as_ptr(),
                    self.scratch.v.as_ptr(),
                    k_cache.as_ptr(),
                    v_cache.as_ptr(),
                    self.scratch.pos.as_ptr(),
                    1,
                    kv_dim,
                )
            );

            timed!(
                attn_ms,
                launch_gqa_attn(
                    ctx,
                    &self.stream,
                    self.scratch.q.as_ptr(),
                    k_cache.as_ptr(),
                    v_cache.as_ptr(),
                    self.scratch.attn_out.as_ptr(),
                    self.scratch.pos.as_ptr(),
                    1,
                    self.config.n_heads,
                    self.config.n_kv_heads,
                    self.config.head_dim,
                )
            );

            let wo = self.store.get(lw.attn_output);
            timed!(
                gemv_other_ms,
                launch_linear(
                    ctx,
                    &self.stream,
                    wo,
                    self.scratch.attn_out.as_ptr(),
                    ptr::null_mut(),
                    self.scratch.proj_out.as_ptr(),
                    1,
                    self.config.n_embd,
                    q_dim,
                )
            );
            timed!(
                rmsnorm_ms,
                launch_add_rmsnorm(
                    ctx,
                    &self.stream,
                    self.scratch.hidden.as_ptr(),
                    self.scratch.proj_out.as_ptr(),
                    self.store.get(lw.ffn_norm).buf.as_ptr(),
                    self.scratch.normed.as_ptr(),
                    1,
                    self.config.n_embd,
                    self.config.rms_eps,
                )
            );

            for (w_idx, out_p) in [
                (lw.ffn_gate, self.scratch.gate.as_ptr()),
                (lw.ffn_up, self.scratch.up.as_ptr()),
            ] {
                let wt = self.store.get(w_idx);
                timed!(
                    gemv_other_ms,
                    launch_linear(
                        ctx,
                        &self.stream,
                        wt,
                        self.scratch.normed.as_ptr(),
                        ptr::null_mut(),
                        out_p,
                        1,
                        self.config.n_ff,
                        self.config.n_embd,
                    )
                );
            }
            timed!(
                other_ms,
                launch_silu_mul(
                    ctx,
                    &self.stream,
                    self.scratch.gate.as_ptr(),
                    self.scratch.up.as_ptr(),
                    self.scratch.gate.as_ptr(),
                    self.config.n_ff,
                )
            );
            let wd = self.store.get(lw.ffn_down);
            if wd.ggml_type == GgmlType::Q6K {
                timed!(
                    q6k_ms,
                    launch_linear(
                        ctx,
                        &self.stream,
                        wd,
                        self.scratch.gate.as_ptr(),
                        ptr::null_mut(),
                        self.scratch.ffn_out.as_ptr(),
                        1,
                        self.config.n_embd,
                        self.config.n_ff,
                    )
                );
            } else {
                timed!(
                    gemv_other_ms,
                    launch_linear(
                        ctx,
                        &self.stream,
                        wd,
                        self.scratch.gate.as_ptr(),
                        ptr::null_mut(),
                        self.scratch.ffn_out.as_ptr(),
                        1,
                        self.config.n_embd,
                        self.config.n_ff,
                    )
                );
            }
            if step_idx + 1 < n_steps {
                let next_l = steps[step_idx + 1].kv_key.layer_id;
                timed!(
                    rmsnorm_ms,
                    launch_add_rmsnorm(
                        ctx,
                        &self.stream,
                        self.scratch.hidden.as_ptr(),
                        self.scratch.ffn_out.as_ptr(),
                        self.store.get(self.layers[next_l].attn_norm).buf.as_ptr(),
                        self.scratch.normed.as_ptr(),
                        1,
                        self.config.n_embd,
                        self.config.rms_eps,
                    )
                );
            } else {
                timed!(
                    rmsnorm_ms,
                    launch_add_rmsnorm(
                        ctx,
                        &self.stream,
                        self.scratch.hidden.as_ptr(),
                        self.scratch.ffn_out.as_ptr(),
                        self.store.get(self.output_norm).buf.as_ptr(),
                        self.scratch.normed.as_ptr(),
                        1,
                        self.config.n_embd,
                        self.config.rms_eps,
                    )
                );
            }
        }

        self.pos += 1;
        timed!(
            lm_head_ms,
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
            )
        );
        let _ = self.argmax_device_logits(ctx)?;
        prof.wall_ms = t_wall0.elapsed().as_secs_f64() * 1000.0;
        Ok(prof)
    }
}

#[derive(Debug, Default)]
pub struct DecodeProfile {
    pub wall_ms: f64,
    pub gemv_other_ms: f64,
    pub q6k_ms: f64,
    pub lm_head_ms: f64,
    pub rmsnorm_ms: f64,
    pub attn_ms: f64,
    pub other_ms: f64,
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
    let handle = match table.ggml_type {
        GgmlType::F32 => ctx.infer.embed_f32,
        GgmlType::F16 => ctx.infer.embed_f16,
        GgmlType::Q8_0 => ctx.infer.embed_q8_0,
        GgmlType::Q4K => ctx.infer.embed_q4_k,
        GgmlType::Q6K => ctx.infer.embed_q6_k,
        other => panic!("unsupported embedding type {:?}", other),
    };
    hip_launch!(
        ctx,
        stream,
        handle,
        [n_tokens as u32, 1, 1],
        [256, 1, 1];
        table.buf.as_ptr(),
        token_ids_ptr,
        out_ptr,
        n_tokens as u32,
        n_embd as u32
    )
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
    hip_launch!(
        ctx,
        stream,
        ctx.infer.rmsnorm_f32,
        [n_tokens as u32, 1, 1],
        [256, 1, 1];
        x_ptr,
        w_ptr,
        out_ptr,
        n_tokens as u32,
        n_embd as u32,
        eps
    )
}

/// `SCORIA_Q4K_ROWS=2|4|8` forces the Q4_K GEMV rows per workgroup; unset keeps the
/// adaptive default. Used by the C-A-C sweep in `harness/ledger.jsonl`.
fn q4k_rows_override() -> Option<u32> {
    static OVERRIDE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("SCORIA_Q4K_ROWS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|r| matches!(r, 2 | 4 | 8))
    })
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
    if m == 1 {
        // Q4_K blocks are 16 superblock-groups wide; grow the rows per workgroup when
        // `k` is short so every group stays busy (see `gemv_q4_k_rows_impl`).
        let q4k_rows: u32 = match q4k_rows_override() {
            Some(r) => r,
            None if k < 2048 => 8,
            None => 4,
        };
        let rows_per_wg = 128u32 / ctx.wave_size;
        let gemv_cfg = match weight.ggml_type {
            GgmlType::F16 => Some((ctx.infer.gemv_f16_f32, n as u32, 1u32, 0usize)),
            GgmlType::Q5_0 => Some((
                ctx.infer.gemv_q5_0_f32,
                (n as u32).div_ceil(rows_per_wg),
                1u32,
                0usize,
            )),
            GgmlType::Q8_0 => Some((
                ctx.infer.gemv_q8_0_f32,
                (n as u32).div_ceil(rows_per_wg),
                1u32,
                0usize,
            )),
            GgmlType::Q4K => {
                let handle = match q4k_rows {
                    2 => ctx.infer.gemv_q4_k_f32,
                    4 => ctx.infer.gemv_q4_k_r4_f32,
                    _ => ctx.infer.gemv_q4_k_r8_f32,
                };
                Some((handle, (n as u32).div_ceil(q4k_rows), q4k_rows, 0usize))
            }
            GgmlType::Q6K => Some((
                ctx.infer.gemv_q6_k_f32,
                (n as u32).div_ceil(2),
                1u32,
                0usize,
            )),
            _ => None,
        };
        if let Some((handle, grid_x, _rows, shared_bytes)) = gemv_cfg {
            return hip_launch_smem!(
                ctx,
                stream,
                handle,
                [grid_x, 1, 1],
                [128, 1, 1],
                shared_bytes as u32;
                weight.buf.as_ptr(),
                x_ptr,
                bias_ptr,
                y_ptr,
                n as u32,
                k as u32
            );
        }
    }

    // NB: the m < 8 scalar path is kept deliberately. WMMA rounds activations and
    // weights to FP16, which costs ~1.5e-3 cosine on 2-7 token prompts (0.9999 ->
    // 0.9984 on the 7B) even though greedy tokens still match; the parity gate wins.
    if m >= 8 && k.is_multiple_of(32) && ctx.arch.is_rdna4() {
        // Wide 64x128 tile has ~2x the arithmetic intensity, but needs >=128 blocks to
        // fill the 64 CUs; smaller matrices keep the 32x64 tile instead.
        let wide_blocks = (n as u32).div_ceil(128) as u64 * (m as u32).div_ceil(64) as u64;
        let (wmma_handle, tile_n, tile_m) = if wide_blocks >= 128 {
            let h = match weight.ggml_type {
                GgmlType::F16 => ctx.infer.gemm_wmma_f16,
                GgmlType::Q5_0 => ctx.infer.gemm_wmma_q5_0,
                GgmlType::Q8_0 => ctx.infer.gemm_wmma_q8_0,
                GgmlType::Q4K => ctx.infer.gemm_wmma_q4_k,
                GgmlType::Q6K => ctx.infer.gemm_wmma_q6_k,
                _ => None,
            };
            (h, 128u32, 64u32)
        } else {
            let h = match weight.ggml_type {
                GgmlType::F16 => ctx.infer.gemm_wmma_narrow_f16,
                GgmlType::Q5_0 => ctx.infer.gemm_wmma_narrow_q5_0,
                GgmlType::Q8_0 => ctx.infer.gemm_wmma_narrow_q8_0,
                GgmlType::Q4K => ctx.infer.gemm_wmma_narrow_q4_k,
                GgmlType::Q6K => ctx.infer.gemm_wmma_narrow_q6_k,
                _ => None,
            };
            (h, 64u32, 32u32)
        };
        if let Some(handle) = wmma_handle {
            return hip_launch!(
                ctx,
                stream,
                handle,
                [(n as u32).div_ceil(tile_n), (m as u32).div_ceil(tile_m), 1],
                [256, 1, 1];
                weight.buf.as_ptr(),
                x_ptr,
                bias_ptr,
                y_ptr,
                m as u32,
                n as u32,
                k as u32
            );
        }
    }

    let handle = match weight.ggml_type {
        GgmlType::F32 => ctx.infer.linear_f32,
        GgmlType::F16 => ctx.infer.linear_f16,
        GgmlType::Q5_0 => ctx.infer.linear_q5_0,
        GgmlType::Q8_0 => ctx.infer.linear_q8_0,
        GgmlType::Q4K => ctx.infer.linear_q4_k,
        GgmlType::Q6K => ctx.infer.linear_q6_k,
        other => panic!("unsupported linear type {:?}", other),
    };
    let waves_per_wg = 256u32 / ctx.wave_size;
    let grid_x = (n as u32).div_ceil(waves_per_wg);
    hip_launch!(
        ctx,
        stream,
        handle,
        [grid_x, m as u32, 1],
        [256, 1, 1];
        weight.buf.as_ptr(),
        x_ptr,
        bias_ptr,
        y_ptr,
        m as u32,
        n as u32,
        k as u32
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_qk_rmsnorm(
    ctx: &EngineContext,
    stream: &HipStream,
    q_ptr: *mut c_void,
    k_ptr: *mut c_void,
    q_w_ptr: *mut c_void,
    k_w_ptr: *mut c_void,
    n_tokens: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    eps: f32,
) -> HipResult<()> {
    hip_launch!(
        ctx,
        stream,
        ctx.infer.qk_rmsnorm_f32,
        [n_tokens as u32, (n_heads + n_kv_heads) as u32, 1],
        [ctx.wave_size, 1, 1];
        q_ptr,
        k_ptr,
        q_w_ptr,
        k_w_ptr,
        n_tokens as u32,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        eps
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_rope_neox(
    ctx: &EngineContext,
    stream: &HipStream,
    q_ptr: *mut c_void,
    k_ptr: *mut c_void,
    pos_ptr: *mut c_void,
    m: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    freq_base: f32,
) -> HipResult<()> {
    let block_x = ((head_dim / 2) as u32).clamp(32, 256);
    hip_launch!(
        ctx,
        stream,
        ctx.infer.rope_neox_f32,
        [m as u32, (n_heads + n_kv_heads) as u32, 1],
        [block_x, 1, 1];
        q_ptr,
        k_ptr,
        pos_ptr,
        m as u32,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32,
        freq_base
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_kv_append(
    ctx: &EngineContext,
    stream: &HipStream,
    k_in: *mut c_void,
    v_in: *mut c_void,
    k_cache: *mut c_void,
    v_cache: *mut c_void,
    pos_ptr: *mut c_void,
    m: usize,
    kv_dim: usize,
) -> HipResult<()> {
    let total = (m * kv_dim) as u32;
    hip_launch!(
        ctx,
        stream,
        ctx.infer.kv_cache_append_f32,
        [total.div_ceil(256), 1, 1],
        [256, 1, 1];
        k_in,
        v_in,
        k_cache,
        v_cache,
        pos_ptr,
        m as u32,
        kv_dim as u32
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_gqa_attn(
    ctx: &EngineContext,
    stream: &HipStream,
    q_ptr: *mut c_void,
    k_cache: *mut c_void,
    v_cache: *mut c_void,
    out_ptr: *mut c_void,
    pos_ptr: *mut c_void,
    m: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> HipResult<()> {
    hip_launch!(
        ctx,
        stream,
        ctx.infer.gqa_attn_f32,
        [n_heads as u32, m as u32, 1],
        [128, 1, 1];
        q_ptr,
        k_cache,
        v_cache,
        out_ptr,
        pos_ptr,
        m as u32,
        n_heads as u32,
        n_kv_heads as u32,
        head_dim as u32
    )
}

fn launch_silu_mul(
    ctx: &EngineContext,
    stream: &HipStream,
    gate_ptr: *mut c_void,
    up_ptr: *mut c_void,
    out_ptr: *mut c_void,
    n: usize,
) -> HipResult<()> {
    let items = if n.is_multiple_of(4) { n / 4 } else { n };
    hip_launch!(
        ctx,
        stream,
        ctx.infer.silu_mul_f32,
        [(items as u32).div_ceil(256), 1, 1],
        [256, 1, 1];
        gate_ptr,
        up_ptr,
        out_ptr,
        n as u32
    )
}

fn launch_add_inplace(
    ctx: &EngineContext,
    stream: &HipStream,
    dst_ptr: *mut c_void,
    src_ptr: *mut c_void,
    n: usize,
) -> HipResult<()> {
    hip_launch!(
        ctx,
        stream,
        ctx.infer.add_inplace_f32,
        [(n as u32).div_ceil(256), 1, 1],
        [256, 1, 1];
        dst_ptr,
        src_ptr,
        n as u32
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_add_rmsnorm(
    ctx: &EngineContext,
    stream: &HipStream,
    x_ptr: *mut c_void,
    res_ptr: *mut c_void,
    w_ptr: *mut c_void,
    out_ptr: *mut c_void,
    n_tokens: usize,
    n_embd: usize,
    eps: f32,
) -> HipResult<()> {
    hip_launch!(
        ctx,
        stream,
        ctx.infer.add_rmsnorm_f32,
        [n_tokens as u32, 1, 1],
        [256, 1, 1];
        x_ptr,
        res_ptr,
        w_ptr,
        out_ptr,
        n_tokens as u32,
        n_embd as u32,
        eps
    )
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

    fn f32_to_f16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = bits & 0x007f_ffff;
        if exp <= 0 {
            sign
        } else if exp >= 31 {
            sign | 0x7c00
        } else {
            sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
        }
    }

    #[test]
    fn probe_wmma_f16_layout() {
        let ctx = EngineContext::init(0).expect("EngineContext::init(0)");
        if !ctx.arch.is_rdna4() {
            return;
        }
        let stream = HipStream::new().expect("HipStream::new");
        let mut a_f32 = [0f32; 256];
        let mut b_f32 = [0f32; 256];
        let mut a_h = [0u16; 256];
        let mut b_h = [0u16; 256];
        for m in 0..16 {
            for k in 0..16 {
                let va = ((m * 3 + k) % 11) as f32 * 0.25;
                let vb = ((m * 5 + k * 2) % 13) as f32 * 0.25;
                a_f32[m * 16 + k] = va;
                b_f32[m * 16 + k] = vb;
                a_h[m * 16 + k] = f32_to_f16_bits(va);
                b_h[m * 16 + k] = f32_to_f16_bits(vb);
            }
        }
        let mut cpu = [[0f32; 16]; 16];
        for m in 0..16 {
            for n in 0..16 {
                let mut s = 0.0f32;
                for k in 0..16 {
                    s += a_f32[m * 16 + k] * b_f32[n * 16 + k];
                }
                cpu[m][n] = s;
            }
        }

        let mut d_a = DeviceBuf::alloc(256 * 2).unwrap();
        let mut d_b = DeviceBuf::alloc(256 * 2).unwrap();
        let d_c = DeviceBuf::alloc(256 * 4).unwrap();
        d_a.copy_from_host(to_bytes(&a_h)).unwrap();
        d_b.copy_from_host(to_bytes(&b_h)).unwrap();

        let probe_fn = ctx
            .infer_module
            .resolve(scoria_kernels::infer_names::WMMA_PROBE_F16)
            .unwrap();
        let mut pa = d_a.as_ptr();
        let mut pb = d_b.as_ptr();
        let mut pc = d_c.as_ptr();
        let mut args: [*mut c_void; 3] = [
            (&mut pa as *mut *mut c_void) as *mut c_void,
            (&mut pb as *mut *mut c_void) as *mut c_void,
            (&mut pc as *mut *mut c_void) as *mut c_void,
        ];
        unsafe {
            ctx.infer_module
                .launch_handle(probe_fn, [1, 1, 1], [32, 1, 1], 0, &stream, &mut args)
                .unwrap();
        }
        stream.synchronize().unwrap();
        let mut gpu_c = [0f32; 256];
        d_c.copy_to_host(to_bytes_mut(&mut gpu_c)).unwrap();

        // Check both candidate mappings on gfx1201 Wave32 16x16x16:
        // Map A: row = (lane >> 4) * 8 + i, col = lane & 15
        // Map B: row = i * 2 + (lane >> 4), col = lane & 15
        let mut err_a = 0.0f32;
        let mut err_b = 0.0f32;
        for lane in 0..32 {
            let col = lane & 15;
            let half = lane >> 4;
            for i in 0..8 {
                let got = gpu_c[lane * 8 + i];
                err_a = err_a.max((got - cpu[half * 8 + i][col]).abs());
                err_b = err_b.max((got - cpu[i * 2 + half][col]).abs());
            }
        }
        eprintln!("[wmma_probe_f16] err_a={err_a:.6} err_b={err_b:.6}");
        assert!(
            err_a < 1e-3 || err_b < 1e-3,
            "neither WMMA C layout matched: err_a={err_a}, err_b={err_b}"
        );
    }

    #[test]
    fn gpu_wmma_prefill_parity() {
        let ctx = EngineContext::init(0).expect("EngineContext::init(0)");
        if !ctx.arch.is_rdna4() {
            return;
        }
        let stream = HipStream::new().expect("HipStream::new");

        // 1. Synthetic F16 check (M=20, N=70, K=64) with bias
        {
            let (m, n, k) = (20usize, 70usize, 64usize);
            let w_f32: Vec<f32> = (0..n * k)
                .map(|i| (((i * 7 + 3) % 23) as f32 - 11.0) * 0.05)
                .collect();
            let w_f16: Vec<u16> = w_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
            let x_in: Vec<f32> = (0..m * k)
                .map(|i| (((i * 5 + 1) % 19) as f32 - 9.0) * 0.04)
                .collect();
            let bias_in: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.01).collect();

            let mut cpu_y = vec![0f32; m * n];
            for tok in 0..m {
                for row in 0..n {
                    let mut acc = bias_in[row];
                    for c in 0..k {
                        acc += w_f32[row * k + c] * x_in[tok * k + c];
                    }
                    cpu_y[tok * n + row] = acc;
                }
            }

            let mut store = VramTensorStore::new();
            let w_info = TensorInfo {
                name: "synth_f16".to_string(),
                dims: vec![k as u64, n as u64],
                ggml_type: GgmlType::F16,
                offset: 0,
            };
            let tid = store
                .upload("synth_f16", &w_info, to_bytes(&w_f16))
                .unwrap();
            let mut d_x = DeviceBuf::alloc(m * k * 4).unwrap();
            let mut d_b = DeviceBuf::alloc(n * 4).unwrap();
            let d_y = DeviceBuf::alloc(m * n * 4).unwrap();
            d_x.copy_from_host(to_bytes(&x_in)).unwrap();
            d_b.copy_from_host(to_bytes(&bias_in)).unwrap();

            launch_linear(
                &ctx,
                &stream,
                store.get(tid),
                d_x.as_ptr(),
                d_b.as_ptr(),
                d_y.as_ptr(),
                m,
                n,
                k,
            )
            .unwrap();
            stream.synchronize().unwrap();
            let mut gpu_y = vec![0f32; m * n];
            d_y.copy_to_host(to_bytes_mut(&mut gpu_y)).unwrap();
            let max_diff = cpu_y
                .iter()
                .zip(&gpu_y)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 5e-2,
                "WMMA F16 parity failed: max_diff={max_diff}"
            );
        }

        // 2. Real quantized weights (Q5_0, Q8_0, Q4_K, Q6_K) at M=20 and M=64
        let model_path = "/home/noorm/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";
        if Path::new(model_path).exists() {
            let gguf = GgufFile::open(model_path).expect("open 0.5B GGUF");
            let test_tensors = [
                "blk.0.attn_q.weight",
                "blk.0.attn_v.weight",
                "blk.0.ffn_down.weight",
                "blk.12.ffn_gate.weight",
                "blk.23.attn_v.weight",
            ];
            for m in [20usize, 64usize] {
                for tname in test_tensors {
                    let Some(info) = gguf.tensor(tname) else {
                        continue;
                    };
                    let k = info.dims[0] as usize;
                    let n_rows = (info.dims[1] as usize).min(96);
                    let row_bytes = info.ggml_type.tensor_byte_len(k).unwrap();
                    let raw = &gguf.tensor_bytes(info).unwrap()[..n_rows * row_bytes];
                    let x_in: Vec<f32> = (0..m * k)
                        .map(|i| (((i * 11 + 7) % 29) as f32 - 14.0) * 0.02)
                        .collect();

                    let mut deq_w = vec![0f32; n_rows * k];
                    for r in 0..n_rows {
                        dequant_row(
                            info.ggml_type,
                            &raw[r * row_bytes..(r + 1) * row_bytes],
                            &mut deq_w[r * k..(r + 1) * k],
                        )
                        .unwrap();
                    }

                    let mut cpu_y = vec![0f32; m * n_rows];
                    for tok in 0..m {
                        let x_row = &x_in[tok * k..(tok + 1) * k];
                        for r in 0..n_rows {
                            let w_row = &deq_w[r * k..(r + 1) * k];
                            cpu_y[tok * n_rows + r] =
                                w_row.iter().zip(x_row).map(|(a, b)| a * b).sum();
                        }
                    }

                    let mut store = VramTensorStore::new();
                    let mut sub_info = info.clone();
                    sub_info.dims = vec![k as u64, n_rows as u64];
                    let tid = store.upload(tname, &sub_info, raw).unwrap();
                    let mut d_x = DeviceBuf::alloc(m * k * 4).unwrap();
                    let d_y = DeviceBuf::alloc(m * n_rows * 4).unwrap();
                    d_x.copy_from_host(to_bytes(&x_in)).unwrap();

                    launch_linear(
                        &ctx,
                        &stream,
                        store.get(tid),
                        d_x.as_ptr(),
                        ptr::null_mut(),
                        d_y.as_ptr(),
                        m,
                        n_rows,
                        k,
                    )
                    .unwrap();
                    stream.synchronize().unwrap();

                    let mut gpu_y = vec![0f32; m * n_rows];
                    d_y.copy_to_host(to_bytes_mut(&mut gpu_y)).unwrap();
                    let max_diff = cpu_y
                        .iter()
                        .zip(&gpu_y)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_diff < 5e-2,
                        "WMMA parity failed for {tname} ({:?}, m={m}): max_diff={max_diff}",
                        info.ggml_type
                    );
                }
            }
        }
    }
}
