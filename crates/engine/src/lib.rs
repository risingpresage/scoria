pub mod qwen2;

use scoria_gguf::QuantCodec;
use scoria_hip_rt::{
    detect_arch, set_device, warp_size, HipModule, HipResult, KernelHandle, TargetArch,
};
use scoria_kernels::{gate_hsaco_for_arch, infer_hsaco_for_arch, infer_names};

macro_rules! define_infer_kernels {
    (
        required {
            $($req_field:ident => $req_sym:expr),* $(,)?
        }
        rdna4_optional {
            $($opt_field:ident => $opt_sym:expr),* $(,)?
        }
    ) => {
        pub struct InferKernels {
            $(pub $req_field: KernelHandle,)*
            $(pub $opt_field: Option<KernelHandle>,)*
        }

        impl InferKernels {
            fn resolve_all(m: &HipModule, _arch: &TargetArch) -> HipResult<Self> {
                Ok(Self {
                    $($req_field: m.resolve($req_sym)?,)*
                    $($opt_field: if _arch.is_rdna4() {
                        Some(m.resolve($opt_sym)?)
                    } else {
                        None
                    },)*
                })
            }
        }
    };
}

define_infer_kernels! {
    required {
        embed_f32 => infer_names::EMBED_LOOKUP_F32,
        embed_f16 => infer_names::EMBED_LOOKUP_F16,
        embed_q8_0 => infer_names::EMBED_LOOKUP_Q8_0,
        embed_q4_k => infer_names::EMBED_LOOKUP_Q4_K,
        embed_q6_k => infer_names::EMBED_LOOKUP_Q6_K,
        rmsnorm_f32 => infer_names::RMSNORM_F32,
        qk_rmsnorm_f32 => infer_names::QK_RMSNORM_F32,
        linear_f32 => infer_names::LINEAR_F32,
        linear_f16 => infer_names::LINEAR_F16,
        linear_q5_0 => infer_names::LINEAR_Q5_0,
        linear_q8_0 => infer_names::LINEAR_Q8_0,
        linear_q4_k => infer_names::LINEAR_Q4_K,
        linear_q6_k => infer_names::LINEAR_Q6_K,
        rope_neox_f32 => infer_names::ROPE_NEOX_F32,
        kv_cache_append_f32 => infer_names::KV_CACHE_APPEND_F32,
        gqa_attn_f32 => infer_names::GQA_ATTN_F32,
        silu_mul_f32 => infer_names::SILU_MUL_F32,
        add_inplace_f32 => infer_names::ADD_INPLACE_F32,
        add_rmsnorm_f32 => infer_names::ADD_RMSNORM_F32,
        decode_advance_u32 => infer_names::DECODE_ADVANCE_U32,
        argmax_stage1_f32 => infer_names::ARGMAX_STAGE1_F32,
        argmax_stage2_f32 => infer_names::ARGMAX_STAGE2_F32,
        gemv_f16_f32 => infer_names::GEMV_F16_F32,
        gemv_q5_0_f32 => infer_names::GEMV_Q5_0_F32,
        gemv_q8_0_f32 => infer_names::GEMV_Q8_0_F32,
        gemv_q4_k_f32 => infer_names::GEMV_Q4_K_F32,
        gemv_q4_k_r4_f32 => infer_names::GEMV_Q4_K_R4_F32,
        gemv_q4_k_r8_f32 => infer_names::GEMV_Q4_K_R8_F32,
        gemv_q6_k_f32 => infer_names::GEMV_Q6_K_F32,
    }
    rdna4_optional {
        gemm_wmma_f16 => infer_names::GEMM_WMMA_F16,
        gemm_wmma_q5_0 => infer_names::GEMM_WMMA_Q5_0,
        gemm_wmma_q8_0 => infer_names::GEMM_WMMA_Q8_0,
        gemm_wmma_q4_k => infer_names::GEMM_WMMA_Q4_K,
        gemm_wmma_q6_k => infer_names::GEMM_WMMA_Q6_K,
        gemm_wmma_narrow_f16 => infer_names::GEMM_WMMA_NARROW_F16,
        gemm_wmma_narrow_q5_0 => infer_names::GEMM_WMMA_NARROW_Q5_0,
        gemm_wmma_narrow_q8_0 => infer_names::GEMM_WMMA_NARROW_Q8_0,
        gemm_wmma_narrow_q4_k => infer_names::GEMM_WMMA_NARROW_Q4_K,
        gemm_wmma_narrow_q6_k => infer_names::GEMM_WMMA_NARROW_Q6_K,
    }
}

pub struct EngineContext {
    pub device_id: i32,
    pub arch: TargetArch,
    pub wave_size: u32,
    pub gate_module: HipModule,
    pub infer_module: HipModule,
    pub infer: InferKernels,
}

impl EngineContext {
    pub fn init(device_id: i32) -> HipResult<Self> {
        set_device(device_id)?;
        let arch = detect_arch(device_id)?;
        let wave_size = warp_size(device_id)?;
        let hsaco = gate_hsaco_for_arch(arch.as_str()).unwrap_or_else(|| {
            panic!(
                "no compiled HSACO kernels for detected AMDGPU arch `{}`",
                arch.as_str()
            )
        });
        let gate_module = HipModule::from_hsaco(hsaco)?;
        let infer_hsaco = infer_hsaco_for_arch(arch.as_str()).unwrap_or_else(|| {
            panic!(
                "no compiled inference HSACO kernels for detected AMDGPU arch `{}`",
                arch.as_str()
            )
        });
        let infer_module = HipModule::from_hsaco(infer_hsaco)?;
        let infer = InferKernels::resolve_all(&infer_module, &arch)?;
        Ok(Self {
            device_id,
            arch,
            wave_size,
            gate_module,
            infer_module,
            infer,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearOp {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub codec: QuantCodec,
}

pub trait LinearKernel {
    fn name(&self) -> &'static str;
    fn supports(&self, codec: QuantCodec) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixerKind {
    FullAttn,
    SlidingWindowAttn(usize),
    LinearRecurrent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerSpec {
    pub layer_id: usize,
    pub mixer: MixerKind,
    pub moe: Option<MoeSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeSpec {
    pub top_k: usize,
    pub num_experts: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage {
    Dense(Vec<LayerSpec>),
    Looped {
        layers: Vec<LayerSpec>,
        passes: usize,
        cross_pass_gate: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionGraph {
    pub stages: Vec<Stage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvSlotKey {
    pub layer_id: usize,
    pub loop_pass: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandedStep {
    pub logical_step: usize,
    pub kv_key: KvSlotKey,
    pub mixer: MixerKind,
    pub moe: Option<MoeSpec>,
    pub cross_pass_gate: bool,
}

impl ExecutionGraph {
    pub fn expand_steps(&self) -> Vec<ExpandedStep> {
        let mut steps = Vec::new();
        for stage in &self.stages {
            match stage {
                Stage::Dense(layers) => {
                    for layer in layers {
                        let logical_step = steps.len();
                        steps.push(ExpandedStep {
                            logical_step,
                            kv_key: KvSlotKey {
                                layer_id: layer.layer_id,
                                loop_pass: 0,
                            },
                            mixer: layer.mixer,
                            moe: layer.moe,
                            cross_pass_gate: false,
                        });
                    }
                }
                Stage::Looped {
                    layers,
                    passes,
                    cross_pass_gate,
                } => {
                    for pass in 0..*passes {
                        for layer in layers {
                            let logical_step = steps.len();
                            steps.push(ExpandedStep {
                                logical_step,
                                kv_key: KvSlotKey {
                                    layer_id: layer.layer_id,
                                    loop_pass: pass,
                                },
                                mixer: layer.mixer,
                                moe: layer.moe,
                                cross_pass_gate: *cross_pass_gate && pass > 0,
                            });
                        }
                    }
                }
            }
        }
        steps
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorTier {
    Vram {
        device_offset: u64,
        bytes: usize,
    },
    PinnedRam {
        host_offset: u64,
        bytes: usize,
    },
    NvmeExtent {
        drive_id: u8,
        offset: u64,
        bytes: usize,
    },
}

pub trait TensorStore {
    fn tier_of(&self, tensor_id: u64) -> Option<&TensorTier>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_inkling2_mini_looped_schedule() {
        let make_layers = |range: std::ops::Range<usize>| -> Vec<LayerSpec> {
            range
                .map(|id| LayerSpec {
                    layer_id: id,
                    mixer: if id % 6 == 5 {
                        MixerKind::FullAttn
                    } else {
                        MixerKind::SlidingWindowAttn(1024)
                    },
                    moe: Some(MoeSpec {
                        top_k: 8,
                        num_experts: 128,
                    }),
                })
                .collect()
        };

        let graph = ExecutionGraph {
            stages: vec![
                Stage::Dense(make_layers(0..6)),
                Stage::Looped {
                    layers: make_layers(6..18),
                    passes: 3,
                    cross_pass_gate: true,
                },
                Stage::Dense(make_layers(18..24)),
            ],
        };

        let steps = graph.expand_steps();
        assert_eq!(steps.len(), 48);
        assert_eq!(
            steps[6].kv_key,
            KvSlotKey {
                layer_id: 6,
                loop_pass: 0
            }
        );
        assert!(!steps[6].cross_pass_gate);
        assert_eq!(
            steps[18].kv_key,
            KvSlotKey {
                layer_id: 6,
                loop_pass: 1
            }
        );
        assert!(steps[18].cross_pass_gate);
        assert_eq!(
            steps[47].kv_key,
            KvSlotKey {
                layer_id: 23,
                loop_pass: 0
            }
        );
    }
}
