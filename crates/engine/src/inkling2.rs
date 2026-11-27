//! `inkling2_loop` architecture contract.
//!
//! Inkling 2 Mini is not a Qwen variant: it has no RoPE, per-layer head counts, a
//! sliding-window mask per layer, relative-position biases, short convolutions on K and
//! V, attention gates and sinks, block-wise attention residuals, `SiTU` activations, a
//! 128-expert top-8 MoE, and a 12-layer block executed three times with learned
//! per-projection loop gates.
//!
//! This module does not run the model. It states what a checkpoint must contain and in
//! what order the layers execute, so kernels get written against a checked contract.
//! `scoria arch-check <gguf>` prints the report.
//!
//! Everything here was derived from the checkpoint itself (`scoria inspect`) and the
//! Apache-2.0 reference in `satellitedown/inkling-2-mini-abliterated-ingine`
//! (`tools/inkling_ref.py`, `native/DESIGN.md`). Shape and presence facts are checked
//! against the GGUF at runtime; nothing is assumed.

use scoria_gguf::{GgmlType, GgufFile, ModelConfig};

pub const ARCH: &str = "inkling2_loop";

/// Which layers a tensor is expected on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Every decoder layer.
    All,
    /// Sliding-window layers only (`sliding_window_pattern[layer] != 0`).
    Sliding,
    /// Full-history layers only (the complement of `Sliding`).
    Full,
    /// Layers inside the looped block (`loop.start..loop.end`).
    Looped,
}

/// One tensor role inside a decoder block.
#[derive(Debug, Clone, Copy)]
pub struct Role {
    pub suffix: &'static str,
    pub policy: Policy,
    /// Present on most layers but not all, so absence is reported, not failed.
    pub optional: bool,
    pub expert: bool,
}

const fn role(suffix: &'static str, policy: Policy) -> Role {
    Role {
        suffix,
        policy,
        optional: false,
        expert: false,
    }
}

const fn opt_role(suffix: &'static str, policy: Policy) -> Role {
    Role {
        suffix,
        policy,
        optional: true,
        expert: false,
    }
}

const fn expert_role(suffix: &'static str) -> Role {
    Role {
        suffix,
        policy: Policy::All,
        optional: false,
        expert: true,
    }
}

/// The contract. The two layer variants are real: sliding-window layers carry one fused
/// `attn_norm`, while full-history layers normalise q, k and v separately and add
/// `attn_{k,v}_conv_proj` on top of the shared short convolutions.
pub const ROLES: &[Role] = &[
    role("attn_norm.weight", Policy::Sliding),
    role("attn_q_inp_norm.weight", Policy::Full),
    role("attn_k_inp_norm.weight", Policy::Full),
    role("attn_v_inp_norm.weight", Policy::Full),
    role("attn_q.weight", Policy::All),
    role("attn_k.weight", Policy::All),
    role("attn_v.weight", Policy::All),
    role("attn_output.weight", Policy::All),
    role("attn_gate.weight", Policy::All),
    role("attn_sink_proj.weight", Policy::All),
    role("attn_rel.weight", Policy::All),
    role("attn_rel_bias.weight", Policy::All),
    role("attn_rel_clip.weight", Policy::All),
    role("attn_q_clip.weight", Policy::All),
    role("attn_k_conv.weight", Policy::All),
    role("attn_v_conv.weight", Policy::All),
    role("attn_k_conv_proj.weight", Policy::Full),
    role("attn_v_conv_proj.weight", Policy::Full),
    role("attn_post_conv.weight", Policy::All),
    opt_role("attn_res_query.weight", Policy::All),
    role("ffn_norm.weight", Policy::All),
    role("ffn_gate_inp.weight", Policy::All),
    expert_role("ffn_gate_up_exps.weight"),
    opt_role("ffn_gate_up_exps.scale", Policy::All),
    expert_role("ffn_down_exps.weight"),
    opt_role("ffn_down_exps.scale", Policy::All),
    role("ffn_post_conv.weight", Policy::All),
    role("ffn_res_query.weight", Policy::All),
    role("exp_probs_b.bias", Policy::All),
    // Learned per-projection gates for the looped block (`loop.gate_rank`).
    role("loop_msg_query.weight", Policy::Looped),
    role("loop_q_ctrl_gain.weight", Policy::Looped),
    role("loop_q_ctrl_forget.weight", Policy::Looped),
    role("loop_q_ctrl_down.weight", Policy::Looped),
    role("loop_k_ctrl_gain.weight", Policy::Looped),
    role("loop_k_ctrl_forget.weight", Policy::Looped),
    role("loop_k_ctrl_down.weight", Policy::Looped),
    role("loop_v_ctrl_gain.weight", Policy::Looped),
    role("loop_v_ctrl_forget.weight", Policy::Looped),
    role("loop_v_ctrl_down.weight", Policy::Looped),
    role("loop_ffn_ctrl_gain.weight", Policy::Looped),
    role("loop_ffn_ctrl_forget.weight", Policy::Looped),
    role("loop_ffn_ctrl_down.weight", Policy::Looped),
];

/// Tensors outside `blk.*`.
pub const GLOBAL_TENSORS: &[&str] = &[
    "token_embd.weight",
    "token_embd_norm.weight",
    "output.weight",
    "output_norm.weight",
    "output_res_query.weight",
];

/// One execution of one decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Visit {
    pub visit: usize,
    pub layer: usize,
    pub pass: usize,
}

/// Hyperparameters the forward pass needs beyond `ModelConfig`.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2Params {
    pub relative_length: usize,
    pub relative_extent: usize,
    pub relative_clip: f32,
    pub qk_clip: f32,
    pub log_scale_alpha: f32,
    pub log_scale_floor: usize,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub attn_res_block: usize,
    pub final_logit_softcap: f32,
    pub shortconv_cache: usize,
    pub expert_ff: usize,
    pub expert_weights_scale: f32,
    pub loop_gate_rank: usize,
}

impl Inkling2Params {
    pub fn from_gguf(file: &GgufFile, cfg: &ModelConfig) -> Self {
        let k = |name: &str| format!("{ARCH}.{name}");
        let f32_of = |name: &str, default: f32| file.meta_f32(&k(name)).unwrap_or(default);
        let us_of = |name: &str, default: usize| {
            file.meta_u32(&k(name))
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        Self {
            relative_length: us_of("attention.relative_length", 0),
            relative_extent: us_of("attention.relative_extent", 0),
            relative_clip: f32_of("attention.relative_clip", 0.0),
            qk_clip: f32_of("attention.qk_clip", 0.0),
            log_scale_alpha: f32_of("attention.log_scale_alpha", 0.0),
            log_scale_floor: us_of("attention.log_scale_floor", 0),
            situ_beta: f32_of("activation.situ_beta", 4.0),
            situ_linear_beta: f32_of("activation.situ_linear_beta", 25.0),
            attn_res_block: us_of("attn_res.block_size", 0),
            final_logit_softcap: f32_of("final_logit_softcapping", 0.0),
            shortconv_cache: us_of("shortconv.l_cache", 0),
            expert_ff: us_of("expert_feed_forward_length", cfg.n_ff),
            expert_weights_scale: f32_of("expert_weights_scale", 1.0),
            loop_gate_rank: us_of("loop.gate_rank", 0),
        }
    }
}

/// The visit order: `0..loop_start`, then the looped block `loop_passes` times, then the
/// tail. Each visit owns its KV slots, which `ModelConfig::kv_budget` already counts.
pub fn schedule(cfg: &ModelConfig) -> Vec<Visit> {
    let mut visits = Vec::new();
    let looped = matches!(
        (cfg.loop_count, cfg.loop_start, cfg.loop_end),
        (Some(p), Some(s), Some(e)) if p > 1 && e > s && e <= cfg.n_layers
    );
    let (passes, start, end) = if looped {
        (
            cfg.loop_count.unwrap(),
            cfg.loop_start.unwrap(),
            cfg.loop_end.unwrap(),
        )
    } else {
        (1, 0, cfg.n_layers)
    };
    for layer in 0..start {
        visits.push(Visit {
            visit: visits.len(),
            layer,
            pass: 0,
        });
    }
    for pass in 0..passes {
        for layer in start..end {
            visits.push(Visit {
                visit: visits.len(),
                layer,
                pass,
            });
        }
    }
    for layer in end..cfg.n_layers {
        visits.push(Visit {
            visit: visits.len(),
            layer,
            pass: 0,
        });
    }
    visits
}

/// Whether a role belongs on `layer`.
pub fn expected_on(role: &Role, cfg: &ModelConfig, layer: usize) -> bool {
    match role.policy {
        Policy::All => true,
        Policy::Sliding => cfg
            .sliding_window_pattern
            .as_ref()
            .and_then(|p| p.get(layer).copied())
            .is_some_and(|v| v != 0),
        Policy::Full => cfg
            .sliding_window_pattern
            .as_ref()
            .and_then(|p| p.get(layer).copied())
            .is_some_and(|v| v == 0),
        Policy::Looped => {
            let in_block = cfg
                .loop_start
                .zip(cfg.loop_end)
                .is_some_and(|(s, e)| layer >= s && layer < e);
            in_block || !cfg.loop_count.is_some_and(|p| p > 1)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub suffix: &'static str,
    /// Layers where the role was expected.
    pub expected: usize,
    pub present: usize,
    /// Expected but absent.
    pub missing: usize,
    pub types: Vec<String>,
    /// Types the engine cannot dequantise yet.
    pub unsupported: Vec<String>,
}

impl Check {
    pub fn is_ok(&self) -> bool {
        self.missing == 0 && self.unsupported.is_empty()
    }
}

fn type_name(t: GgmlType) -> String {
    format!("{t:?}")
}

/// Types `crates/gguf/src/dequant.rs` can turn into f32 today.
pub fn engine_supports(t: GgmlType) -> bool {
    matches!(
        t,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::Q5_0
            | GgmlType::Q8_0
            | GgmlType::Q4K
            | GgmlType::Q6K
    )
}

/// Checks every role across every layer, and returns tensor names that match neither a
/// role nor a global name, so a piece of the architecture cannot go unnoticed.
pub fn check(file: &GgufFile, cfg: &ModelConfig) -> (Vec<Check>, Vec<String>) {
    let mut checks = Vec::with_capacity(ROLES.len());
    for r in ROLES {
        let (mut expected, mut present, mut missing) = (0usize, 0usize, 0usize);
        let (mut types, mut unsupported) = (Vec::new(), Vec::new());
        for layer in 0..cfg.n_layers {
            let name = format!("blk.{layer}.{}", r.suffix);
            match file.tensor(&name) {
                Some(t) => {
                    present += 1;
                    let tn = type_name(t.ggml_type);
                    if !types.contains(&tn) {
                        types.push(tn.clone());
                    }
                    if !engine_supports(t.ggml_type) && !unsupported.contains(&tn) {
                        unsupported.push(tn);
                    }
                }
                None => {
                    if expected_on(r, cfg, layer) {
                        expected += 1;
                        if !r.optional {
                            missing += 1;
                        }
                    }
                }
            }
        }
        if present > 0 {
            expected = expected.max(present.min(expected + missing));
        }
        checks.push(Check {
            suffix: r.suffix,
            expected: expected + present,
            present,
            missing,
            types,
            unsupported,
        });
    }

    let mut known: Vec<String> = Vec::new();
    for r in ROLES {
        for layer in 0..cfg.n_layers {
            known.push(format!("blk.{layer}.{}", r.suffix));
        }
    }
    for g in GLOBAL_TENSORS {
        known.push((*g).to_string());
    }
    let mut unknown: Vec<String> = file
        .tensors
        .iter()
        .map(|t| t.name.clone())
        .filter(|n| !known.iter().any(|k| k == n))
        .collect();
    unknown.sort();
    (checks, unknown)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ModelConfig {
        ModelConfig {
            architecture: ARCH.to_string(),
            n_layers: 24,
            context_length: 1 << 20,
            n_embd: 2304,
            n_ff: 1152,
            n_heads: 24,
            n_kv_heads: 8,
            head_dim: 128,
            vocab_size: 200058,
            rope_freq_base: 10000.0,
            rms_eps: 1e-6,
            tied_embeddings: false,
            kv_heads_per_layer: None,
            sliding_window: Some(1024),
            // layers 5, 11, 17, 23 are full-history
            sliding_window_pattern: Some(
                (0..24)
                    .map(|l| u8::from(!matches!(l, 5 | 11 | 17 | 23)))
                    .collect(),
            ),
            n_experts: Some(128),
            n_experts_used: Some(8),
            loop_count: Some(3),
            loop_start: Some(6),
            loop_end: Some(18),
        }
    }

    #[test]
    fn schedule_matches_the_looped_execution_order() {
        let s = schedule(&cfg());
        assert_eq!(s.len(), 48);
        assert_eq!((s[0].layer, s[0].pass), (0, 0));
        assert_eq!((s[5].layer, s[5].pass), (5, 0));
        assert_eq!((s[6].layer, s[6].pass), (6, 0));
        assert_eq!((s[17].layer, s[17].pass), (17, 0));
        assert_eq!((s[18].layer, s[18].pass), (6, 1));
        assert_eq!((s[41].layer, s[41].pass), (17, 2));
        assert_eq!((s[42].layer, s[42].pass), (18, 0));
        assert_eq!((s[47].layer, s[47].pass), (23, 0));
        for (i, v) in s.iter().enumerate() {
            assert_eq!(v.visit, i);
        }
    }

    #[test]
    fn schedule_without_a_loop_is_identity() {
        let mut c = cfg();
        c.loop_count = None;
        c.loop_start = None;
        c.loop_end = None;
        let s = schedule(&c);
        assert_eq!(s.len(), 24);
        assert!(s.iter().all(|v| v.pass == 0 && v.layer == v.visit));
    }

    #[test]
    fn layer_variants_are_exclusive() {
        let c = cfg();
        let norm = ROLES
            .iter()
            .find(|r| r.suffix == "attn_norm.weight")
            .unwrap();
        let q_inp = ROLES
            .iter()
            .find(|r| r.suffix == "attn_q_inp_norm.weight")
            .unwrap();
        // Sliding-window layers have the fused norm and no per-projection norms.
        assert!(expected_on(norm, &c, 0));
        assert!(!expected_on(q_inp, &c, 0));
        // The four full-history layers are the other way round.
        assert!(!expected_on(norm, &c, 5));
        assert!(expected_on(q_inp, &c, 5));
        assert_eq!((0..24).filter(|&l| expected_on(q_inp, &c, l)).count(), 4);
        assert_eq!((0..24).filter(|&l| expected_on(norm, &c, l)).count(), 20);
    }

    #[test]
    fn loop_gates_only_on_the_looped_block() {
        let c = cfg();
        let gate = ROLES
            .iter()
            .find(|r| r.suffix == "loop_q_ctrl_gain.weight")
            .unwrap();
        assert!(!expected_on(gate, &c, 5));
        assert!(expected_on(gate, &c, 6));
        assert!(expected_on(gate, &c, 17));
        assert!(!expected_on(gate, &c, 18));
    }

    #[test]
    fn contract_covers_the_architecture() {
        // 29 per-block roles + 13 loop gates + the two expert scales.
        assert_eq!(ROLES.len(), 42);
        assert_eq!(ROLES.iter().filter(|r| r.expert).count(), 2);
        // attn_res_query plus the two expert scale companions.
        assert_eq!(ROLES.iter().filter(|r| r.optional).count(), 3);
    }
}
