use scoria_gguf::QuantCodec;
use scoria_hip_rt::{detect_arch, set_device, warp_size, HipModule, HipResult, TargetArch};
use scoria_kernels::gate_hsaco_for_arch;

pub struct EngineContext {
    pub device_id: i32,
    pub arch: TargetArch,
    pub wave_size: u32,
    pub gate_module: HipModule,
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
        Ok(Self {
            device_id,
            arch,
            wave_size,
            gate_module,
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
