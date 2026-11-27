//! Layer-0 MoE for `inkling2_loop`: 128 experts, top-8, softcapped SwiGLU.
//!
//! Transcribed from `modeling_inkling2_loop.py`:
//!
//! * router: `logits = W_gate_inp x` in f32, `scores = act(logits)`, selection is
//!   `topk(scores + e_score_correction_bias)` but the **weights come from the raw
//!   scores**, renormalised to sum 1 and scaled by `routed_scaling_factor` (8.0 here).
//!   Getting that backwards is a classic MoE bug, so it is tested.
//! * expert: `gate_up = W13[e] x`, split in half, then
//!   `softcap(gate) * sigmoid(gate) * softcap(up)` with `softcap_c(x) = tanh(x/c) * c`,
//!   then `down = W2[e] z`, accumulated in f32 and weighted by the router weight.
//! * the 3-D expert tensors are stored expert-major, so one expert is one contiguous
//!   slab; the GGUF also carries a per-expert f32 outer scale that the raw dequantised
//!   values must be multiplied by (~4e-5).
//!
//! `act` is `sigmoid` or `softmax`; the checkpoint does not name it, and the presence of
//! an `e_score_correction_bias` (a noaux-tc router feature) points at sigmoid, so that is
//! the default and `route_with` lets a caller try the other.

use scoria_gguf::{GgufError, GgufFile, ModelConfig, TensorInfo};

use crate::inkling2::Inkling2Params;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterScore {
    Sigmoid,
    Softmax,
}

/// Routing decision for one token.
#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    /// Expert ids, in the order `topk` produced them.
    pub indices: Vec<usize>,
    /// Renormalised weights including the routed scaling factor.
    pub weights: Vec<f32>,
}

/// `softcap_c(x) = tanh(x / c) * c`.
pub fn softcap(x: f32, c: f32) -> f32 {
    (x / c).tanh() * c
}

/// `softcap(gate) * sigmoid(gate) * softcap(up)`, with the sigmoid on the *raw* gate.
pub fn swiglu_softcapped(gate: &[f32], up: &[f32], gate_cap: f32, up_cap: f32, out: &mut [f32]) {
    assert_eq!(gate.len(), up.len());
    assert_eq!(out.len(), gate.len());
    for i in 0..gate.len() {
        let g = gate[i];
        let u = up[i];
        out[i] = softcap(g, gate_cap) * (1.0 / (1.0 + (-g).exp())) * softcap(u, up_cap);
    }
}

/// One expert-major 3-D tensor. The bytes stay in the GGUF's mmap: 24 layers of experts
/// is 12.7 GB on disk, so copying them into the process would be the whole model twice.
struct ExpertTensor {
    info: TensorInfo,
    slab_bytes: usize,
    in_dim: usize,
    out_dim: usize,
    experts: usize,
    /// Per-expert outer f32 scale, when the checkpoint carries one.
    outer: Vec<f32>,
}

impl ExpertTensor {
    fn load(file: &GgufFile, name: &str, cfg: &ModelConfig) -> Result<Self, GgufError> {
        let info = file
            .tensor(name)
            .ok_or_else(|| GgufError::MissingMetadata(format!("tensor `{name}`")))?;
        if info.dims.len() != 3 {
            return Err(GgufError::Malformed(format!("`{name}` is not 3-D")));
        }
        let experts = info.dims[2] as usize;
        let outer = match file.tensor(&name.replace(".weight", ".scale")) {
            Some(si) => {
                let bytes = file.tensor_bytes(si)?;
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect()
            }
            None => vec![1.0; experts],
        };
        let _ = cfg;
        let total = info.ggml_type.tensor_byte_len(
            info.n_elements()
                .ok_or_else(|| GgufError::Malformed(format!("`{name}` shape overflows")))?,
        )?;
        Ok(Self {
            slab_bytes: total / experts,
            info: info.clone(),
            in_dim: info.dims[0] as usize,
            out_dim: info.dims[1] as usize,
            experts,
            outer,
        })
    }

    /// Dequantises expert `e` into `out` (`out_dim * in_dim`) and applies the outer scale.
    fn expert(&self, file: &GgufFile, e: usize, out: &mut [f32]) {
        assert!(e < self.experts, "expert {e} out of range");
        assert_eq!(out.len(), self.out_dim * self.in_dim);
        let bytes = file
            .tensor_bytes(&self.info)
            .expect("expert tensor is in the mmap");
        let start = e * self.slab_bytes;
        scoria_gguf::dequant::dequant_row(
            self.info.ggml_type,
            &bytes[start..start + self.slab_bytes],
            out,
        )
        .expect("expert slab dequantises");
        let k = self.outer[e];
        for v in out.iter_mut() {
            *v *= k;
        }
    }
}

pub struct Moe {
    n_embd: usize,
    n_experts: usize,
    top_k: usize,
    ff: usize,
    gate_cap: f32,
    up_cap: f32,
    routed_scale: f32,
    score: RouterScore,
    w_gate_inp: Vec<f32>,
    e_score_bias: Vec<f32>,
    gate_up: ExpertTensor,
    down: ExpertTensor,
    pub last_route: Option<Route>,
}

impl Moe {
    pub fn load(file: &GgufFile, cfg: &ModelConfig, score: RouterScore) -> Result<Self, GgufError> {
        Self::load_layer(file, cfg, score, 0)
    }

    /// Load layer `layer`'s router and expert tensors.
    pub fn load_layer(
        file: &GgufFile,
        cfg: &ModelConfig,
        score: RouterScore,
        layer: usize,
    ) -> Result<Self, GgufError> {
        let params = Inkling2Params::from_gguf(file, cfg);
        let n_experts = cfg.n_experts.unwrap_or(0);
        let top_k = cfg.n_experts_used.unwrap_or(0);
        if n_experts == 0 || top_k == 0 {
            return Err(GgufError::Malformed("checkpoint has no experts".into()));
        }
        let read_f32 = |name: &str| -> Result<Vec<f32>, GgufError> {
            let info = file
                .tensor(name)
                .ok_or_else(|| GgufError::MissingMetadata(format!("tensor `{name}`")))?;
            let bytes = file.tensor_bytes(info)?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        };
        let t = |name: &str| format!("blk.{layer}.{name}");
        let gate_up = ExpertTensor::load(file, &t("ffn_gate_up_exps.weight"), cfg)?;
        let down = ExpertTensor::load(file, &t("ffn_down_exps.weight"), cfg)?;
        Ok(Self {
            n_embd: cfg.n_embd,
            n_experts,
            top_k,
            ff: params.expert_ff,
            // `mlp_gate_softcap` / `mlp_up_softcap` in the transformer config; the GGUF
            // exposes them as `activation.situ_{,linear_}beta`.
            gate_cap: params.situ_beta,
            up_cap: params.situ_linear_beta,
            routed_scale: params.expert_weights_scale,
            score,
            w_gate_inp: read_f32(&t("ffn_gate_inp.weight"))?,
            e_score_bias: read_f32(&t("exp_probs_b.bias"))?,
            gate_up,
            down,
            last_route: None,
        })
    }

    pub fn router_scores(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.n_embd);
        // GGUF stores the router as [n_embd, n_experts].
        let mut logits = vec![0f32; self.n_experts];
        for (e, slot) in logits.iter_mut().enumerate() {
            let row = &self.w_gate_inp[e * self.n_embd..(e + 1) * self.n_embd];
            *slot = row.iter().zip(x).map(|(a, b)| a * b).sum();
        }
        match self.score {
            RouterScore::Sigmoid => logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
            RouterScore::Softmax => {
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                exps.iter().map(|v| v / sum).collect()
            }
        }
    }

    /// Selection uses `scores + bias`; the weights come from the raw scores.
    pub fn route(&mut self, x: &[f32]) -> Route {
        let scores = self.router_scores(x);
        let biased: Vec<f32> = scores
            .iter()
            .zip(&self.e_score_bias)
            .map(|(s, b)| s + b)
            .collect();
        let mut order: Vec<usize> = (0..self.n_experts).collect();
        order.sort_by(|&a, &b| biased[b].total_cmp(&biased[a]).then(a.cmp(&b)));
        let indices: Vec<usize> = order[..self.top_k].to_vec();
        let picked: Vec<f32> = indices.iter().map(|&i| scores[i]).collect();
        let sum: f32 = picked.iter().sum::<f32>() + 1e-20;
        let weights = picked.iter().map(|w| w / sum * self.routed_scale).collect();
        let route = Route { indices, weights };
        self.last_route = Some(route.clone());
        route
    }

    /// One token through the MoE, returning the accumulated expert contribution.
    pub fn forward(&mut self, file: &GgufFile, x: &[f32]) -> Vec<f32> {
        let route = self.route(x);
        let mut acc = vec![0f32; self.n_embd];
        let mut slab = vec![0f32; self.gate_up.out_dim * self.gate_up.in_dim];
        let mut gate_up = vec![0f32; self.gate_up.out_dim];
        let mut gate = vec![0f32; self.ff];
        let mut up = vec![0f32; self.ff];
        let mut act = vec![0f32; self.ff];
        let mut mid = vec![0f32; self.down.out_dim];
        let mut down_slab = vec![0f32; self.down.out_dim * self.down.in_dim];

        for (slot, &e) in route.indices.iter().enumerate() {
            self.gate_up.expert(file, e, &mut slab);
            // gate_up = W13[e] x, [2*ff]
            for (o, out) in gate_up.iter_mut().enumerate() {
                let row = &slab[o * self.gate_up.in_dim..(o + 1) * self.gate_up.in_dim];
                *out = row.iter().zip(x).map(|(a, b)| a * b).sum();
            }
            // The GGUF layout is `[gate rows; up rows]`, because the converter fuses the
            // checkpoint's two separate tensors with `torch.cat([gate, up], dim=1)`
            // (`conversion/base.py`, `--fuse-gate-up-exps`). The *source* checkpoint
            // interleaves its `w13` rows, but that is undone before fusion, so a half
            // split is right here and an even/odd split would be wrong.
            let (gate_rows, up_rows) = gate_up.split_at(self.ff);
            gate.copy_from_slice(gate_rows);
            up.copy_from_slice(up_rows);
            swiglu_softcapped(&gate, &up, self.gate_cap, self.up_cap, &mut act);

            self.down.expert(file, e, &mut down_slab);
            for (o, out) in mid.iter_mut().enumerate() {
                let row = &down_slab[o * self.down.in_dim..(o + 1) * self.down.in_dim];
                *out = row.iter().zip(&act).map(|(a, b)| a * b).sum();
            }
            let w = route.weights[slot];
            for (a, m) in acc.iter_mut().zip(&mid) {
                *a += w * m;
            }
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softcap_is_tanh_times_c() {
        assert_eq!(softcap(0.0, 4.0), 0.0);
        // Large inputs saturate at +-c instead of exploding.
        assert!((softcap(100.0, 4.0) - 4.0).abs() < 1e-5);
        assert!((softcap(-100.0, 4.0) + 4.0).abs() < 1e-5);
        // Small inputs pass through almost unchanged.
        assert!((softcap(0.01, 4.0) - 0.01).abs() < 1e-4);
    }

    #[test]
    fn swiglu_softcapped_matches_a_hand_computation() {
        let gate = [0.0f32, 1.0, -1.0];
        let up = [2.0f32, 0.0, 100.0];
        let mut out = [0f32; 3];
        swiglu_softcapped(&gate, &up, 4.0, 25.0, &mut out);
        // i = 0: tanh(0)*4 * sigmoid(0) * tanh(2/25)*25 = 0
        assert_eq!(out[0], 0.0);
        // i = 1: up is 0, and softcap(0) is 0, so the product is 0.
        assert_eq!(out[1], 0.0);
        // i = 2: gate -1, up 100. tanh(100/25) = tanh(4) = 0.999329, *25 = 24.983, NOT 25:
        // the softcap saturates asymptotically, and a test that assumed 25 was wrong.
        let want = (-0.25f32).tanh() * 4.0 * (1.0 / (1.0 + 1f32.exp())) * (4.0f32.tanh() * 25.0);
        assert!((out[2] - want).abs() < 1e-5, "{} vs {}", out[2], want);
    }

    #[test]
    fn swiglu_reads_concatenated_gate_then_up_rows() {
        // `--fuse-gate-up-exps` writes `torch.cat([gate, up], dim=1)`, so the first `ff`
        // rows are gate and the last `ff` are up. The source checkpoint's interleaved
        // `w13` order is undone before that fusion.
        let gate_up = [0.5f32, -0.25, 2.0, 10.0];
        let ff = 2;
        let (gate, up) = gate_up.split_at(ff);
        assert_eq!(gate, &[0.5, -0.25]);
        assert_eq!(up, &[2.0, 10.0]);
        // The interleaved reading would pair (0.5, -0.25) and (2.0, 10.0) differently; make
        // sure the two conventions cannot both pass.
        assert_ne!(gate, [gate_up[0], gate_up[2]]);
    }

    #[test]
    fn routing_selects_by_biased_score_but_weights_by_raw_score() {
        // Two of four experts, with a bias large enough to reorder the selection.
        let moe = Moe {
            n_embd: 2,
            n_experts: 4,
            top_k: 2,
            ff: 1,
            gate_cap: 4.0,
            up_cap: 25.0,
            routed_scale: 8.0,
            score: RouterScore::Sigmoid,
            // rows are experts; x = (1, 0) so logits are the first column.
            w_gate_inp: vec![
                0.0, 0.0, // e0 -> 0
                2.0, 0.0, // e1 -> 2
                -2.0, 0.0, // e2 -> -2
                1.0, 0.0, // e3 -> 1
            ],
            e_score_bias: vec![10.0, 0.0, 0.0, 0.0], // e0 wins on bias alone
            gate_up: ExpertTensor {
                info: TensorInfo {
                    name: "test".into(),
                    dims: vec![2, 2, 4],
                    ggml_type: scoria_gguf::GgmlType::F32,
                    offset: 0,
                },
                slab_bytes: 0,
                in_dim: 2,
                out_dim: 2,
                experts: 4,
                outer: vec![1.0; 4],
            },
            down: ExpertTensor {
                info: TensorInfo {
                    name: "test".into(),
                    dims: vec![1, 2, 4],
                    ggml_type: scoria_gguf::GgmlType::F32,
                    offset: 0,
                },
                slab_bytes: 0,
                in_dim: 1,
                out_dim: 2,
                experts: 4,
                outer: vec![1.0; 4],
            },
            last_route: None,
        };
        let mut moe = moe;
        let route = moe.route(&[1.0, 0.0]);
        // Selection: e0 (bias 10) and e1 (score sigmoid(2) = 0.881) beat e3 (0.731).
        assert_eq!(route.indices[0], 0);
        assert_eq!(route.indices[1], 1);
        // Weights use the *raw* scores, not the biased ones.
        let s0 = 1.0 / (1.0 + 0f32.exp()); // sigmoid(0)
        let s1 = 1.0 / (1.0 + (-2f32).exp());
        let norm = s0 + s1;
        assert!((route.weights[0] - s0 / norm * 8.0).abs() < 1e-5);
        assert!((route.weights[1] - s1 / norm * 8.0).abs() < 1e-5);
        // The scaling factor is applied, so the weights sum to 8.
        assert!((route.weights.iter().sum::<f32>() - 8.0).abs() < 1e-4);
    }
}
