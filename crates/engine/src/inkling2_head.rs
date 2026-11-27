//! Output head for `inkling2_loop`: the final block-residual query, the norm, the
//! unembedding matvec and the logit softcap.
//!
//! From `modeling_inkling2_loop.py`:
//!
//! ```text
//! hidden = final_input(block_states, hidden)[0]   # 1-query attention, output_res_query
//! hidden = norm(hidden)                           # output_norm
//! logits = lm_head(hidden)[:unpadded_vocab]       # output.weight, NOT tied to the embedding
//! logits = tanh(logits / 30) * 30                 # final_logit_softcapping
//! ```
//!
//! The unembedding matrix is 200058 x 2304 (the GGUF pads it to 201024 rows), so it is
//! dequantised in row chunks rather than materialised: as f32 it would be 1.84 GB.

use scoria_gguf::{GgmlType, GgufError, GgufFile};

/// `tanh(x / c) * c`, the logit softcap. Asymptotically bounded by `c`.
pub fn softcap(x: f32, c: f32) -> f32 {
    if c <= 0.0 {
        return x;
    }
    (x / c).tanh() * c
}

/// The final block-residual: one learned query attending over the block snapshots plus the
/// current hidden state, non-causal, scaled by `hidden_size^-0.5`.
pub fn block_residual(
    query: &[f32],
    block_states: &[Vec<f32>],
    hidden: &[f32],
    eps: f32,
) -> Vec<f32> {
    let n = hidden.len();
    assert_eq!(query.len(), n);
    let mut stacked: Vec<Vec<f32>> = block_states.to_vec();
    stacked.push(hidden.to_vec());
    for v in stacked.iter_mut() {
        crate::inkling2_attn::unweighted_rmsnorm(v, eps);
    }
    let scale = (n as f32).sqrt().recip();
    let mut weights: Vec<f32> = stacked
        .iter()
        .map(|v| query.iter().zip(v).map(|(a, b)| a * b).sum::<f32>() * scale)
        .collect();
    crate::inkling2_attn::sink_softmax(&mut weights, f32::NEG_INFINITY);
    let mut out = vec![0f32; n];
    for (w, v) in weights.iter().zip(&stacked) {
        for (o, vi) in out.iter_mut().zip(v) {
            *o += w * vi;
        }
    }
    out
}

/// `output.weight` rows as f32, streamed in chunks of `rows`.
pub struct Unembed {
    ggml_type: GgmlType,
    bytes: Vec<u8>,
    row_bytes: usize,
    /// Real vocabulary size; the tensor is padded past this.
    pub vocab: usize,
}

impl Unembed {
    pub fn load(file: &GgufFile, n_embd: usize, vocab_unpadded: usize) -> Result<Self, GgufError> {
        let info = file
            .tensor("output.weight")
            .ok_or_else(|| GgufError::MissingMetadata("output.weight".into()))?;
        let row_bytes = info
            .ggml_type
            .tensor_byte_len(n_embd)
            .map_err(|_| GgufError::Malformed("output row size".into()))?;
        let rows = (info.n_elements().unwrap_or(0)) / n_embd.max(1);
        Ok(Self {
            ggml_type: info.ggml_type,
            bytes: file.tensor_bytes(info)?.to_vec(),
            row_bytes,
            vocab: vocab_unpadded.min(rows),
        })
    }

    /// `logits[v] = hidden . W[v]` for every real row, softcapped.
    pub fn logits(&self, hidden: &[f32], eps: f32, cap: f32) -> Vec<f32> {
        let n_embd = hidden.len();
        let mut normed = hidden.to_vec();
        let mean_sq = normed.iter().map(|v| v * v).sum::<f32>() / n_embd as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for v in normed.iter_mut() {
            *v *= inv;
        }

        let mut out = Vec::with_capacity(self.vocab);
        let mut row = vec![0f32; n_embd];
        for v in 0..self.vocab {
            let start = v * self.row_bytes;
            scoria_gguf::dequant::dequant_row(
                self.ggml_type,
                &self.bytes[start..start + self.row_bytes],
                &mut row,
            )
            .expect("unembedding row dequantises");
            let dot = row.iter().zip(&normed).map(|(a, b)| a * b).sum::<f32>();
            out.push(softcap(dot, cap));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softcap_bounds_and_preserves_small_values() {
        assert_eq!(softcap(0.0, 30.0), 0.0);
        // Asymptote, not a clamp: the value approaches c but never reaches it, and it is
        // odd. Asserting a decimal expansion by hand is how this test was wrong twice, so
        // assert the properties instead and compare against the definition once.
        assert!(softcap(100.0, 30.0) < 30.0 && softcap(100.0, 30.0) > 29.9);
        assert!(softcap(-100.0, 30.0) > -30.0 && softcap(-100.0, 30.0) < -29.9);
        assert!((softcap(100.0, 30.0) - (100.0f32 / 30.0).tanh() * 30.0).abs() < 1e-6);
        assert!(
            (softcap(-1.5, 30.0) + softcap(1.5, 30.0)).abs() < 1e-6,
            "odd"
        );
        // Monotone.
        assert!(softcap(0.5, 30.0) > softcap(0.4, 30.0));
        // Small logits pass through almost unchanged.
        assert!((softcap(0.3, 30.0) - 0.3).abs() < 1e-4);
        // A non-positive cap disables it rather than dividing by zero.
        assert_eq!(softcap(5.0, 0.0), 5.0);
    }

    #[test]
    fn block_residual_is_a_single_query_attention() {
        // Two snapshots plus the hidden state, query picking out the second snapshot.
        let eps = 1e-6;
        let block = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
        let hidden = vec![1.0f32, 1.0];
        let query = vec![0.0f32, 1.0];
        let out = block_residual(&query, &block, &hidden, eps);
        // The output is a convex combination of the normalised snapshots, so each element
        // must lie inside the range spanned by them. (RMSNorm fixes the RMS, not the norm,
        // so asserting a unit norm here was wrong.)
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for v in block.iter().chain(std::iter::once(&hidden)) {
            let mut n = v.clone();
            crate::inkling2_attn::unweighted_rmsnorm(&mut n, eps);
            for x in n {
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        assert!(
            out.iter().all(|x| *x >= lo - 1e-6 && *x <= hi + 1e-6),
            "out {out:?} in [{lo},{hi}]"
        );
        // The query points at the second snapshot, so its direction dominates.
        assert!(out[1] > out[0], "out {out:?}");
        // Determinism.
        let again = block_residual(&query, &block, &hidden, eps);
        assert_eq!(out, again);
    }

    #[test]
    fn logits_are_softcapped_and_deterministic() {
        // Two vocabulary rows of four f32 values, one row equal to the hidden state and one
        // equal to its negation, so the two logits must be opposite in sign.
        let hidden = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut bytes = Vec::new();
        for row in [
            hidden.clone(),
            hidden.iter().map(|v| -v).collect::<Vec<_>>(),
        ] {
            for v in row {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        let unembed = Unembed {
            ggml_type: GgmlType::F32,
            bytes,
            row_bytes: 4 * 4,
            vocab: 2,
        };
        let logits = unembed.logits(&hidden, 1e-6, 30.0);
        assert_eq!(logits.len(), 2);
        assert!(logits[0] > 0.0 && logits[1] < 0.0, "{logits:?}");
        assert!(logits.iter().all(|v| v.abs() <= 30.0));
        // f32 1.84 GB would be the full table; only `vocab` rows are touched.
        assert_eq!(unembed.logits(&hidden, 1e-6, 30.0), logits);
    }
}
