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
pub(crate) struct ExpertTensor {
    pub(crate) info: TensorInfo,
    pub(crate) slab_bytes: usize,
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
    pub(crate) experts: usize,
    /// Per-expert outer f32 scale, when the checkpoint carries one.
    pub(crate) outer: Vec<f32>,
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

    /// `y[o] = outer[e] * (W[e] row o . x)`, dequantising **one row at a time**.
    ///
    /// Materialising the whole expert as f32 costs `out_dim * in_dim * 4` bytes of write
    /// traffic per expert -- 32 MB for this model, 12 GB per token across 8 experts and 48
    /// visits -- and it was the largest single cost. A row is 2304 values (9 KB of f32),
    /// which stays in cache, so the same dequantisation work runs against a third of the
    /// memory traffic.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn row_matvec(
        &self,
        file: &GgufFile,
        e: usize,
        x: &[f32],
        out: &mut [f32],
        row: &mut Vec<f32>,
    ) {
        assert!(e < self.experts, "expert {e} out of range");
        assert_eq!(x.len(), self.in_dim, "row_matvec input");
        assert_eq!(out.len(), self.out_dim, "row_matvec output");
        let bytes = file
            .tensor_bytes(&self.info)
            .expect("expert tensor is in the mmap");
        let row_bytes = self
            .info
            .ggml_type
            .tensor_byte_len(self.in_dim)
            .expect("expert row size");
        let base = e * self.slab_bytes;
        let scale = self.outer[e];
        row.resize(self.in_dim, 0.0);
        for (o, slot) in out.iter_mut().enumerate() {
            let start = base + o * row_bytes;
            scoria_gguf::dequant::dequant_row(
                self.info.ggml_type,
                &bytes[start..start + row_bytes],
                row,
            )
            .expect("expert row dequantises");
            *slot = row.iter().zip(x).map(|(a, b)| a * b).sum::<f32>() * scale;
        }
    }

    /// `outs[p][o] = outer[e] * (W[e] row o . xs[p])`, dequantising each row **once** and
    /// dotting it against every input.
    ///
    /// This is the whole point of a batched prefill: dequantisation measured at half the
    /// cost of a single-position expert GEMV, so amortising it across the positions that
    /// share an expert removes that half for all but the first.
    fn row_matvec_m(
        &self,
        file: &GgufFile,
        e: usize,
        xs: &[&[f32]],
        outs: &mut [Vec<f32>],
        row: &mut Vec<f32>,
    ) {
        assert_eq!(xs.len(), outs.len(), "one output per input");
        assert!(e < self.experts, "expert {e} out of range");
        let bytes = file
            .tensor_bytes(&self.info)
            .expect("expert tensor is in the mmap");
        let row_bytes = self
            .info
            .ggml_type
            .tensor_byte_len(self.in_dim)
            .expect("expert row size");
        let base = e * self.slab_bytes;
        let scale = self.outer[e];
        row.resize(self.in_dim, 0.0);
        for out in outs.iter_mut() {
            assert_eq!(out.len(), self.out_dim, "row_matvec_m output");
        }
        for o in 0..self.out_dim {
            let start = base + o * row_bytes;
            scoria_gguf::dequant::dequant_row(
                self.info.ggml_type,
                &bytes[start..start + row_bytes],
                row,
            )
            .expect("expert row dequantises");
            for (out, x) in outs.iter_mut().zip(xs) {
                assert_eq!(x.len(), self.in_dim, "row_matvec_m input");
                out[o] = row.iter().zip(*x).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
        }
    }
}

/// Per-thread working set: a dequantised-row buffer, the expert output vectors and the
/// activation. Sized once and reused, because a token dequantises ~8 experts per layer visit
/// and 48 visits deep that is a lot of allocation churn.
struct Scratch {
    /// One dequantised row, reused for every row of both expert matrices.
    row: Vec<f32>,
    gate_up: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    mid: Vec<f32>,
}

impl Scratch {
    /// `row` must fit the widest input dimension of the two expert matrices.
    fn new(gate_up_out: usize, down_out: usize, ff: usize, row: usize) -> Self {
        Self {
            row: vec![0.0; row],
            gate_up: vec![0.0; gate_up_out],
            gate: vec![0.0; ff],
            up: vec![0.0; ff],
            act: vec![0.0; ff],
            mid: vec![0.0; down_out],
        }
    }
}

/// Reusable per-worker buffers for a chunked MoE. Allocating these per call dominated the
/// chunk path: 16 workers x ~73k floats per layer-visit is ~300 MB of zeroing per position,
/// which made the batched host MoE 9x slower per layer than the per-position one.
#[derive(Default)]
struct ChunkScratch {
    acc: Vec<Vec<f32>>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    gu: Vec<Vec<f32>>,
    dn: Vec<Vec<f32>>,
    acts: Vec<Vec<f32>>,
    row: Vec<f32>,
}

impl ChunkScratch {
    fn prepare(
        &mut self,
        positions: usize,
        n_embd: usize,
        ff: usize,
        out_dim: usize,
        dn_out: usize,
    ) {
        if self.acc.len() < positions {
            self.acc.resize(positions, Vec::new());
        }
        for row in self.acc.iter_mut() {
            row.clear();
            row.resize(n_embd, 0.0);
        }
        self.gate.clear();
        self.gate.resize(ff, 0.0);
        self.up.clear();
        self.up.resize(ff, 0.0);
        self.act.clear();
        self.act.resize(ff, 0.0);
        if self.gu.len() < positions {
            self.gu.resize(positions, Vec::new());
        }
        for row in self.gu.iter_mut() {
            row.clear();
            row.resize(out_dim, 0.0);
        }
        if self.dn.len() < positions {
            self.dn.resize(positions, Vec::new());
        }
        for row in self.dn.iter_mut() {
            row.clear();
            row.resize(dn_out, 0.0);
        }
        if self.acts.len() < positions {
            self.acts.resize(positions, Vec::new());
        }
        for row in self.acts.iter_mut() {
            row.clear();
            row.resize(ff, 0.0);
        }
    }
}

pub struct Moe {
    chunk_scratch: Vec<ChunkScratch>,
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
    /// One worker per thread; the routed experts are independent, so the top-k split
    /// across threads is the whole parallelism available here.
    threads: usize,
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
            chunk_scratch: Vec::new(),
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
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
        let route = self.route_pure(x);
        self.last_route = Some(route.clone());
        route
    }

    /// The same decision without recording it, for batched callers.
    pub fn route_pure(&self, x: &[f32]) -> Route {
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
        Route { indices, weights }
    }

    /// Worker count for the expert loop. One token routes to `top_k` experts and each is
    /// independent, so this is the only parallelism the MoE offers without splitting a
    /// single matvec.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// The SwiGLU softcaps, the expert width, and the two expert tensors, for a caller that
    /// wants to run the expert matvecs somewhere else (the GPU path).
    pub fn caps(&self) -> (f32, f32) {
        (self.gate_cap, self.up_cap)
    }

    pub fn ff(&self) -> usize {
        self.ff
    }

    pub(crate) fn gate_up_tensor(&self) -> &ExpertTensor {
        &self.gate_up
    }

    pub(crate) fn down_tensor(&self) -> &ExpertTensor {
        &self.down
    }

    /// One token through the MoE with the routing already decided: the host expert loop,
    /// threaded across the routed experts exactly like `forward`.
    pub fn forward_route(&self, file: &GgufFile, x: &[f32], route: &Route) -> Vec<f32> {
        let n_embd = self.n_embd;
        let ff = self.ff;
        let gate_cap = self.gate_cap;
        let up_cap = self.up_cap;
        let top_k = route.indices.len();
        let workers = self.threads.min(top_k).max(1);
        let gu_out = self.gate_up.out_dim;
        let dn_out = self.down.out_dim;
        let widest = self.gate_up.in_dim.max(self.down.in_dim);
        let per = top_k.div_ceil(workers);
        let gate_up = &self.gate_up;
        let down = &self.down;
        let indices = &route.indices;
        let weights = &route.weights;

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for worker in 0..workers {
                let lo = worker * per;
                let hi = ((worker + 1) * per).min(top_k);
                if lo >= hi {
                    break;
                }
                handles.push(scope.spawn(move || {
                    let mut sc = Scratch::new(gu_out, dn_out, ff, widest);
                    let mut acc = vec![0f32; n_embd];
                    for slot in lo..hi {
                        let e = indices[slot];
                        gate_up.row_matvec(file, e, x, &mut sc.gate_up, &mut sc.row);
                        let (gate_rows, up_rows) = sc.gate_up.split_at(ff);
                        sc.gate.copy_from_slice(gate_rows);
                        sc.up.copy_from_slice(up_rows);
                        swiglu_softcapped(&sc.gate, &sc.up, gate_cap, up_cap, &mut sc.act);
                        down.row_matvec(file, e, &sc.act, &mut sc.mid, &mut sc.row);
                        let w = weights[slot];
                        for (a, m) in acc.iter_mut().zip(&sc.mid) {
                            *a += w * m;
                        }
                    }
                    acc
                }));
            }
            let parts: Vec<Vec<f32>> = handles
                .into_iter()
                .map(|h| h.join().expect("expert worker panicked"))
                .collect();
            let mut acc = vec![0f32; n_embd];
            for part in parts {
                for (a, p) in acc.iter_mut().zip(&part) {
                    *a += p;
                }
            }
            acc
        })
    }

    /// A chunk of positions through the MoE.
    ///
    /// Routing is per position, so the positions that chose an expert are gathered and the
    /// expert is evaluated once for all of them. Dequantisation — half the cost of a
    /// single-position expert GEMV — is therefore paid once per expert instead of once per
    /// (position, expert), which is the entire reason to batch a prefill.
    pub fn forward_chunk(&mut self, file: &GgufFile, xs: &[Vec<f32>], outs: &mut [Vec<f32>]) {
        let n_embd = self.n_embd;
        let ff = self.ff;
        assert_eq!(xs.len(), outs.len());
        for out in outs.iter_mut() {
            out.resize(n_embd, 0.0);
            out.fill(0.0);
        }
        let routes: Vec<Route> = xs.iter().map(|x| self.route_pure(x)).collect();
        let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.n_experts];
        for (pos, route) in routes.iter().enumerate() {
            for (slot, &expert) in route.indices.iter().enumerate() {
                buckets[expert].push((pos, route.weights[slot]));
            }
        }

        let experts = self.n_experts;
        let gate_up_out = self.gate_up.out_dim;
        let dn_out = self.down.out_dim;
        // Only experts that were actually routed to have work: for a chunk of C positions
        // that is at most `C * top_k`, so spawning a thread per expert range wastes most of
        // the spawns. This was a third of the chunk path's MoE cost at chunk size 1.
        let busy = buckets.iter().filter(|b| !b.is_empty()).count();
        let workers = self.threads.min(experts).min(busy.max(1)).max(1);
        let per = experts.div_ceil(workers);
        let gate_cap = self.gate_cap;
        let up_cap = self.up_cap;
        let buckets = &buckets;
        while self.chunk_scratch.len() < workers {
            self.chunk_scratch.push(ChunkScratch::default());
        }
        let Moe {
            chunk_scratch,
            gate_up,
            down,
            ..
        } = self;
        let gate_up_ref: &ExpertTensor = gate_up;
        let down_ref: &ExpertTensor = down;
        let scratch = &mut chunk_scratch[..workers];
        for sc in scratch.iter_mut() {
            sc.prepare(xs.len(), n_embd, ff, gate_up_out, dn_out);
        }

        // Each worker owns a contiguous range of experts and its own accumulation buffer,
        // so positions touched by two experts are summed at the end rather than raced.
        let partials: Vec<Vec<Vec<f32>>> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for (worker, sc) in scratch.iter_mut().enumerate() {
                let lo = worker * per;
                let hi = ((worker + 1) * per).min(experts);
                if lo >= hi {
                    break;
                }
                handles.push(scope.spawn(move || {
                    for (expert, hits) in buckets.iter().enumerate().take(hi).skip(lo) {
                        if hits.is_empty() {
                            continue;
                        }
                        let inputs: Vec<&[f32]> =
                            hits.iter().map(|(pos, _)| xs[*pos].as_slice()).collect();
                        gate_up_ref.row_matvec_m(
                            file,
                            expert,
                            &inputs,
                            &mut sc.gu[..hits.len()],
                            &mut sc.row,
                        );

                        for which in 0..hits.len() {
                            sc.gate.copy_from_slice(&sc.gu[which][..ff]);
                            sc.up.copy_from_slice(&sc.gu[which][ff..]);
                            swiglu_softcapped(
                                &sc.gate,
                                &sc.up,
                                gate_cap,
                                up_cap,
                                &mut sc.acts[which],
                            );
                        }
                        let act_refs: Vec<&[f32]> =
                            sc.acts[..hits.len()].iter().map(|a| a.as_slice()).collect();
                        down_ref.row_matvec_m(
                            file,
                            expert,
                            &act_refs,
                            &mut sc.dn[..hits.len()],
                            &mut sc.row,
                        );

                        for (which, (pos, weight)) in hits.iter().enumerate() {
                            for (a, m) in sc.acc[*pos].iter_mut().zip(&sc.dn[which]) {
                                *a += weight * m;
                            }
                        }
                    }
                    std::mem::take(&mut sc.acc)
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().expect("chunk expert worker panicked"))
                .collect()
        });

        for part in partials {
            for (pos, row) in part.iter().enumerate() {
                for (a, m) in outs[pos].iter_mut().zip(row) {
                    *a += m;
                }
            }
        }
    }

    /// One token through the MoE, returning the accumulated expert contribution.
    pub fn forward(&mut self, file: &GgufFile, x: &[f32]) -> Vec<f32> {
        let route = self.route(x);
        self.forward_route(file, x, &route)
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
            chunk_scratch: Vec::new(),
            threads: 1,
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
