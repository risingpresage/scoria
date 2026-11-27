//! `inkling2_loop` attention core, as a CPU reference.
//!
//! Transcribed from the authoritative implementation,
//! `transformers/models/inkling2_loop/modeling_inkling2_loop.py`, because this
//! architecture is not a Qwen variant: no RoPE, a hidden-states-conditioned *relative
//! position bias*, per-head L2 clipping instead of a fixed scale, depthwise causal
//! short convolutions on K and V, a learned per-head attention sink, and an output gate.
//!
//! This module is deliberately host-only and dependency-free. It exists so the future
//! `v_wmma` kernel has something to be wrong against: `AGENTS.md` requires device code
//! to ship with a CPU reference, and writing the kernel first would leave the ordering
//! of clip/conv/gate/sink unknowable.
//!
//! Ordering, per the reference:
//!
//! ```text
//! q = W_q h                      -> clip(scale_q) -> per-head vectors
//! k = shortconv(W_k h)           -> clip
//! v = shortconv(W_v h)           -> rmsnorm (unweighted)
//! bias = gather(clip(scale_rel)(W_rel h) @ proj, q_pos - k_pos)
//! p = softmax([q.k^T * d^-0.5 + bias + mask, sink])[:-1]
//! out = W_o((p v) * sigmoid(W_gate h))
//! ```

/// Scales each head vector down to `threshold` when its L2 norm exceeds it.
pub fn l2_clip(x: &mut [f32], threshold: f32) {
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    let factor = (threshold / (norm + 1e-6)).min(1.0);
    for v in x.iter_mut() {
        *v *= factor;
    }
}

/// `l2_clip(x * scale)` with a learned per-element scale, as used for Q and the relative
/// states. `scale` must have the same layout as `x`.
pub fn scaled_l2_clip(x: &mut [f32], scale: &[f32], threshold: f32) {
    assert_eq!(x.len(), scale.len(), "scale must match the head layout");
    for (v, s) in x.iter_mut().zip(scale) {
        *v *= *s;
    }
    l2_clip(x, threshold);
}

/// `1 + alpha * ln(max((pos + 1) / floor, 1))`: exactly 1 until the floor, then grows
/// logarithmically. Applied to full-attention Q only.
pub fn log_scale(pos: usize, alpha: f32, floor: usize) -> f32 {
    let ratio = ((pos + 1) as f32 / floor.max(1) as f32).max(1.0);
    1.0 + alpha * ratio.ln()
}

/// Relative position bias for one query block against `kv_len` keys.
///
/// `rel_states` is `[heads][d_rel]` for the *last* query position only in the incremental
/// case; `proj` is `[d_rel][rel_extent]`. The bias for query `pos_q` and key `pos_k` is
/// `rel_states[h] . proj[:, (pos_q - pos_k)]`, and zero when that distance is outside
/// `[0, rel_extent)`.
#[allow(clippy::too_many_arguments)]
pub fn relative_bias(
    rel_states: &[f32],
    heads: usize,
    d_rel: usize,
    proj: &[f32],
    rel_extent: usize,
    pos_q: usize,
    pos_k: &[usize],
    out: &mut [f32],
) {
    assert_eq!(rel_states.len(), heads * d_rel);
    assert_eq!(proj.len(), d_rel * rel_extent);
    assert_eq!(out.len(), heads * pos_k.len());
    for h in 0..heads {
        // Project this head's relative state onto the bank of distance profiles.
        let mut logits = vec![0f32; rel_extent];
        for (d, logit) in logits.iter_mut().enumerate() {
            let mut acc = 0.0;
            for r in 0..d_rel {
                acc += rel_states[h * d_rel + r] * proj[r * rel_extent + d];
            }
            *logit = acc;
        }
        for (i, &s) in pos_k.iter().enumerate() {
            let value = match pos_q.checked_sub(s) {
                Some(dist) if dist < rel_extent => logits[dist],
                _ => 0.0,
            };
            out[h * pos_k.len() + i] = value;
        }
    }
}

/// Softmax over `logits` with one extra zero-valued sink logit that absorbs probability
/// mass and is then discarded.
pub fn sink_softmax(logits: &mut [f32], sink: f32) {
    let max = logits.iter().copied().fold(sink, f32::max);
    let mut sum = (sink - max).exp();
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in logits.iter_mut() {
        *v *= inv;
    }
}

/// Depthwise causal convolution with a 4-tap kernel, matching `sconv_kernel_size` and
/// `shortconv.l_cache` in the checkpoint. `cache` holds the previous `taps - 1` input
/// values per channel and is updated in place, so decode steps chain correctly.
pub fn short_conv1d_4tap(x: &mut [f32], weight: &[f32], cache: &mut [f32], channels: usize) {
    assert_eq!(weight.len(), 4 * channels, "weight is [taps][channels]");
    assert_eq!(cache.len(), 3 * channels, "cache is [taps-1][channels]");
    for c in 0..channels {
        let history = [cache[c], cache[channels + c], cache[2 * channels + c], x[c]];
        // Causal: output at t depends on t, t-1, t-2, t-3 with the newest tap last.
        let mut acc = 0.0;
        for (tap, &value) in history.iter().enumerate() {
            acc += weight[tap * channels + c] * value;
        }
        cache[c] = history[1];
        cache[channels + c] = history[2];
        cache[2 * channels + c] = history[3];
        x[c] = acc;
    }
}

/// The output gate: `attn * sigmoid(gate)`.
pub fn apply_output_gate(attn: &mut [f32], gate: &[f32]) {
    assert_eq!(attn.len(), gate.len());
    for (a, g) in attn.iter_mut().zip(gate) {
        *a *= 1.0 / (1.0 + (-*g).exp());
    }
}

/// Unweighted RMSNorm, as used on V: divide by the root-mean-square of the head vector
/// with no learned weight and no mean subtraction.
pub fn unweighted_rmsnorm(x: &mut [f32], eps: f32) {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Everything that happens to the raw projections after the matmuls and before attention:
///
/// * `q`: `scaled_l2_clip(scale_q, qk_clip)` per head,
/// * `k`: 4-tap causal short convolution, then `l2_clip(qk_clip)` per head,
/// * `v`: 4-tap causal short convolution, then unweighted RMSNorm per head,
/// * `gate`: `sigmoid`.
///
/// The convolutions run per channel across time and chain through `cache_k`/`cache_v`, so
/// this is the same code for prefill steps and decode steps. Ordering is transcribed from
/// the reference: conv *before* the head split, so the head reduction sees convolved
/// values.
#[allow(clippy::too_many_arguments)]
pub fn post_qkv(
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    gate: &mut [f32],
    scale_q: &[f32],
    conv_w_k: &[f32],
    conv_w_v: &[f32],
    cache_k: &mut [f32],
    cache_v: &mut [f32],
    heads: usize,
    head_dim: usize,
    kv_dim: usize,
    qk_clip: f32,
    eps: f32,
) {
    assert_eq!(q.len(), heads * head_dim);
    assert_eq!(gate.len(), q.len());
    assert_eq!(k.len(), kv_dim);
    assert_eq!(v.len(), kv_dim);

    // Per-head reductions first, on the raw projections, then the conv for k/v.
    for h in 0..heads {
        scaled_l2_clip(
            &mut q[h * head_dim..(h + 1) * head_dim],
            &scale_q[h * head_dim..(h + 1) * head_dim],
            qk_clip,
        );
    }
    short_conv1d_4tap(k, conv_w_k, cache_k, kv_dim);
    short_conv1d_4tap(v, conv_w_v, cache_v, kv_dim);
    let kv_heads = kv_dim / head_dim;
    for h in 0..kv_heads {
        l2_clip(&mut k[h * head_dim..(h + 1) * head_dim], qk_clip);
        unweighted_rmsnorm(&mut v[h * head_dim..(h + 1) * head_dim], eps);
    }
    for g in gate.iter_mut() {
        *g = 1.0 / (1.0 + (-*g).exp());
    }
}

/// One query head attending over a KV cache, exactly as the reference orders it:
/// `q.k^T * head_dim^-0.5 + relative_bias`, masked causal/window, softmax with the sink
/// absorbing mass, then the weighted sum of `v`.
///
/// `k_cache` and `v_cache` are `[positions][head_dim]` for this head only; `bias` is one
/// value per position for this head. Returns one `head_dim` vector.
pub fn decode_head(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    bias: &[f32],
    sink: f32,
    pos: usize,
    window: Option<usize>,
) -> Vec<f32> {
    let head_dim = q.len();
    assert_eq!(bias.len(), pos + 1, "one bias per visible position");
    assert_eq!(k_cache.len(), (pos + 1) * head_dim);
    assert_eq!(v_cache.len(), k_cache.len());

    let scale = (head_dim as f32).sqrt().recip();
    let mut logits = Vec::with_capacity(pos + 1);
    for t in 0..=pos {
        if !visible(pos, t, window) {
            logits.push(f32::NEG_INFINITY);
            continue;
        }
        let k = &k_cache[t * head_dim..(t + 1) * head_dim];
        let dot: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum();
        logits.push(dot * scale + bias[t]);
    }
    // A fully masked row would make every logit -inf; the sink then carries all the mass
    // and the output is zero, which is the reference behaviour rather than a NaN.
    sink_softmax(&mut logits, sink);
    let mut out = vec![0f32; head_dim];
    for (t, p) in logits.iter().enumerate() {
        if *p == 0.0 {
            continue;
        }
        let v = &v_cache[t * head_dim..(t + 1) * head_dim];
        for (o, vi) in out.iter_mut().zip(v) {
            *o += p * vi;
        }
    }
    out
}

/// Whether key `pos_k` is visible to query `pos_q`: causal, and inside the sliding window
/// when the layer has one. Attention layers with a window still see the whole prefix in
/// this architecture's full-attention variant, so `window == None` means unbounded.
pub fn visible(pos_q: usize, pos_k: usize, window: Option<usize>) -> bool {
    if pos_k > pos_q {
        return false;
    }
    match window {
        Some(w) => pos_q - pos_k < w,
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unweighted_rmsnorm_matches_the_definition() {
        // RMS of (3,4) is sqrt(12.5); dividing gives (3,4)/sqrt(12.5).
        let mut x = [3.0f32, 4.0];
        unweighted_rmsnorm(&mut x, 0.0);
        let inv = 1.0 / 12.5f32.sqrt();
        assert!((x[0] - 3.0 * inv).abs() < 1e-6);
        assert!((x[1] - 4.0 * inv).abs() < 1e-6);
        // A zero vector stays finite thanks to eps.
        let mut z = [0.0f32; 4];
        unweighted_rmsnorm(&mut z, 1e-6);
        assert!(z.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn post_qkv_clips_q_scales_and_norms_v_in_the_documented_order() {
        let (heads, head_dim) = (2usize, 2usize);
        let kv_dim = 2 * head_dim; // 4 kv heads? no: kv_dim/head_dim = 2 kv heads
        let mut q = vec![3.0f32, 4.0, 0.0, 0.0]; // head 0 norm 5, head 1 zero
        let mut k = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut v = vec![1.0f32, 1.0, 1.0, 1.0];
        let mut gate = vec![0.0f32, 100.0, -100.0, 0.0];
        let scale_q = vec![1.0f32; 4];
        let conv_w_k = vec![0.0f32; 4 * kv_dim]; // zero taps -> k becomes zero
        let conv_w_v = vec![0.0f32; 4 * kv_dim];
        let mut cache_k = vec![0.0f32; 3 * kv_dim];
        let mut cache_v = vec![0.0f32; 3 * kv_dim];
        post_qkv(
            &mut q,
            &mut k,
            &mut v,
            &mut gate,
            &scale_q,
            &conv_w_k,
            &conv_w_v,
            &mut cache_k,
            &mut cache_v,
            heads,
            head_dim,
            kv_dim,
            2.0,
            1e-6,
        );
        // q head 0 clipped to norm 2: (3,4)*0.4 = (1.2,1.6); head 1 stays zero and finite.
        assert!((q[0] - 1.2).abs() < 1e-6 && (q[1] - 1.6).abs() < 1e-6);
        assert_eq!(&q[2..4], &[0.0, 0.0]);
        // Zero conv taps wipe k, and l2_clip leaves zero alone.
        assert!(k.iter().all(|x| *x == 0.0));
        // v conv yields zero, and RMSNorm of a zero vector is zero, not NaN.
        assert!(v.iter().all(|x| *x == 0.0));
        // gate is a sigmoid.
        assert!((gate[0] - 0.5).abs() < 1e-6);
        assert!((gate[1] - 1.0).abs() < 1e-6);
        assert!(gate[2] < 1e-6);
    }

    #[test]
    fn post_qkv_convolution_moves_time_forward_through_the_cache() {
        // One head of width 1, taps [1,2,3,4] newest last, so streamed outputs are
        // 4x[t] + 3x[t-1] + 2x[t-2] + 1x[t-3].
        let (heads, head_dim, kv_dim) = (1usize, 1usize, 1usize);
        let conv_w = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut cache_k = vec![0.0f32; 3];
        let mut cache_v = vec![0.0f32; 3];
        let scale_q = vec![1.0f32];
        let mut seen = Vec::new();
        for step in 1..=4 {
            let mut q = vec![0.0f32];
            let mut k = vec![step as f32];
            let mut v = vec![0.0f32];
            let mut gate = vec![0.0f32];
            post_qkv(
                &mut q,
                &mut k,
                &mut v,
                &mut gate,
                &scale_q,
                &conv_w,
                &conv_w,
                &mut cache_k,
                &mut cache_v,
                heads,
                head_dim,
                kv_dim,
                1e9,
                1e-6,
            );
            seen.push(k[0]);
        }
        // step 1: 4*1 = 4 (after conv; l2_clip with a huge threshold leaves it).
        assert!((seen[0] - 4.0).abs() < 1e-5);
        // step 2: 4*2 + 3*1 = 11.
        assert!((seen[1] - 11.0).abs() < 1e-5);
        // step 4: 4*4 + 3*3 + 2*2 + 1*1 = 30.
        assert!((seen[3] - 30.0).abs() < 1e-5);
    }

    fn reference_decode(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        bias: &[f32],
        sink: f32,
        pos: usize,
        window: Option<usize>,
    ) -> Vec<f32> {
        // Deliberately naive: build the full score row, normalise, weight.
        let d = q.len();
        let scale = (d as f32).sqrt().recip();
        let mut scores: Vec<f32> = (0..=pos)
            .map(|t| {
                if !visible(pos, t, window) {
                    f32::NEG_INFINITY
                } else {
                    let dot: f32 = (0..d).map(|i| q[i] * k[t * d + i]).sum();
                    dot * scale + bias[t]
                }
            })
            .collect();
        let max = scores.iter().copied().fold(sink, f32::max);
        let denom: f32 = scores.iter().map(|s| (s - max).exp()).sum::<f32>() + (sink - max).exp();
        for s in scores.iter_mut() {
            *s = (*s - max).exp() / denom;
        }
        (0..d)
            .map(|i| (0..=pos).map(|t| scores[t] * v[t * d + i]).sum::<f32>())
            .collect()
    }

    #[test]
    fn decode_head_matches_a_naive_reference() {
        let d = 4;
        let pos = 5;
        let q: Vec<f32> = (0..d).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let k: Vec<f32> = (0..(pos + 1) * d)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.03)
            .collect();
        let v: Vec<f32> = (0..(pos + 1) * d)
            .map(|i| ((i * 5 % 13) as f32 - 6.0) * 0.02)
            .collect();
        let bias: Vec<f32> = (0..=pos).map(|t| 0.01 * t as f32).collect();
        let sink = -0.5;

        for window in [None, Some(6), Some(3), Some(1)] {
            let got = decode_head(&q, &k, &v, &bias, sink, pos, window);
            let want = reference_decode(&q, &k, &v, &bias, sink, pos, window);
            for (g, w) in got.iter().zip(&want) {
                assert!(
                    (g - w).abs() < 1e-6,
                    "window {window:?}: {got:?} vs {want:?}"
                );
            }
        }
    }

    #[test]
    fn decode_head_window_admits_exactly_the_last_n_positions() {
        let pos = 9;
        let q = [1.0f32];
        // Value equals the position index, so the output is the weighted mean position.
        let k = vec![0.0f32; pos + 1]; // uniform scores
        let v: Vec<f32> = (0..=pos).map(|t| t as f32).collect();
        let bias = vec![0.0f32; pos + 1];
        // Window 4 admits positions 6..9 -> mean 7.5.
        let out = decode_head(&q, &k, &v, &bias, f32::NEG_INFINITY, pos, Some(4));
        assert!((out[0] - 7.5).abs() < 1e-5, "got {}", out[0]);
        // Window 1 admits only the current position.
        let out = decode_head(&q, &k, &v, &bias, f32::NEG_INFINITY, pos, Some(1));
        assert!((out[0] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn decode_head_with_everything_masked_is_zero_not_nan() {
        let q = [1.0f32, 0.0];
        let k = vec![0.0f32; 4];
        let v = vec![1.0f32; 4];
        let bias = vec![0.0f32; 2];
        // Window 0 hides even the current position.
        let out = decode_head(&q, &k, &v, &bias, 0.0, 1, Some(0));
        assert!(out.iter().all(|o| *o == 0.0), "got {out:?}");
    }

    #[test]
    fn l2_clip_only_shrinks_oversized_heads() {
        // Hand computed: norm 5 -> 2/5 = 0.4, so (3,4) becomes (1.2,1.6).
        let mut x = [3.0f32, 4.0];
        l2_clip(&mut x, 2.0);
        assert!((x[0] - 1.2).abs() < 1e-6 && (x[1] - 1.6).abs() < 1e-6);
        // A short vector is untouched.
        let mut y = [0.3f32, 0.4];
        l2_clip(&mut y, 22.627);
        assert_eq!(y, [0.3, 0.4]);
        // A zero vector must not divide by zero.
        let mut z = [0.0f32; 4];
        l2_clip(&mut z, 1.0);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn scaled_clip_applies_the_learned_scale_first() {
        let mut x = [1.0f32, 0.0];
        scaled_l2_clip(&mut x, &[4.0, 1.0], 2.0);
        // scaled -> (4,0), norm 4, factor 0.5 -> (2,0)
        assert!((x[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn log_scale_is_one_below_the_floor_then_grows() {
        assert_eq!(log_scale(8191, 0.1, 8192), 1.0);
        assert_eq!(log_scale(0, 0.1, 8192), 1.0);
        // pos 16383 -> (16384/8192) = 2 -> 1 + 0.1*ln2
        let v = log_scale(16383, 0.1, 8192);
        assert!((v - (1.0 + 0.1 * 2f32.ln())).abs() < 1e-6);
        // Monotonic.
        assert!(log_scale(100_000, 0.1, 8192) > log_scale(50_000, 0.1, 8192));
    }

    #[test]
    fn relative_bias_gathers_by_backward_distance_and_zeroes_the_rest() {
        // Two heads, d_rel = 2, rel_extent = 4. proj[r][d] = r as 1,2 and 3,4.
        let proj = [
            1.0f32, 2.0, 3.0, 4.0, // r = 0 across distances 0..3
            10.0, 20.0, 30.0, 40.0, // r = 1
        ];
        let rel_states = [1.0f32, 0.0, 0.0, 1.0]; // head 0 picks r=0, head 1 picks r=1
        let pos_k = [2usize, 1, 0];
        let mut out = [0f32; 2 * 3];
        relative_bias(&rel_states, 2, 2, &proj, 4, 2, &pos_k, &mut out);
        // distances are 0, 1, 2 -> head 0 sees 1, 2, 3; head 1 sees 10, 20, 30.
        assert_eq!(&out[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&out[3..6], &[10.0, 20.0, 30.0]);

        // A key in the future (distance underflow) and one past the extent are zero.
        let pos_k = [5usize, 0];
        let mut out = [1f32; 4];
        relative_bias(&rel_states, 2, 2, &proj, 4, 2, &pos_k, &mut out);
        assert_eq!(out[0], 0.0, "future key must contribute no bias");
        assert_eq!(out[1], 3.0, "distance 2 is inside the extent");
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 30.0);

        // Boundary: distance extent-1 is visible, distance extent is masked. With
        // pos_q = 4 and keys [1, 0] those distances are 3 and 4.
        let head0 = &rel_states[0..2];
        let mut out = [0f32; 2];
        relative_bias(head0, 1, 2, &proj, 4, 4, &[1, 0], &mut out);
        assert_eq!(out[0], 4.0, "distance 3 is inside the extent");
        assert_eq!(out[1], 0.0, "distance 4 is outside the extent");

        // And with pos_q = 3 the same keys are at distances 2 and 0, both visible.
        let mut out = [0f32; 2];
        relative_bias(head0, 1, 2, &proj, 4, 3, &[1, 3], &mut out);
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 1.0);
    }

    #[test]
    fn sink_absorbs_mass_without_appearing_in_the_output() {
        let mut logits = [0.0f32, 0.0];
        sink_softmax(&mut logits, 0.0);
        // Three equal candidates: [1/3, 1/3], the sink taking the third.
        assert!((logits[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((logits.iter().sum::<f32>() - 2.0 / 3.0).abs() < 1e-6);
        // A very negative sink restores a plain two-way softmax.
        let mut logits = [0.0f32, 0.0];
        sink_softmax(&mut logits, -100.0);
        assert!((logits[0] - 0.5).abs() < 1e-6);
        // A very positive sink silences attention entirely.
        let mut logits = [0.0f32, 0.0];
        sink_softmax(&mut logits, 100.0);
        assert!(logits.iter().all(|p| *p < 1e-6));
    }

    #[test]
    fn short_conv_is_causal_and_chains_through_the_cache() {
        // One channel, taps [1, 2, 3, 4] newest last: y[t] = x[t]*4 + x[t-1]*3 + ...
        let weight = [1.0f32, 2.0, 3.0, 4.0];
        let mut cache = [0.0f32; 3];
        // Streaming one step at a time must equal processing a whole sequence at once.
        let seq = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut streamed = Vec::new();
        for &v in &seq {
            let mut x = [v];
            short_conv1d_4tap(&mut x, &weight, &mut cache, 1);
            streamed.push(x[0]);
        }
        // Hand computed for t=0: 4*1 = 4.
        assert_eq!(streamed[0], 4.0);
        // t=1: 4*2 + 3*1 = 11. t=2: 4*3 + 3*2 + 2*1 = 20.
        assert_eq!(streamed[1], 11.0);
        assert_eq!(streamed[2], 20.0);
        // t=4: 4*5 + 3*4 + 2*3 + 1*2 = 40.
        assert_eq!(streamed[4], 40.0);
    }

    #[test]
    fn output_gate_is_a_sigmoid() {
        let mut attn = [1.0f32, 1.0];
        apply_output_gate(&mut attn, &[0.0, 100.0]);
        assert!((attn[0] - 0.5).abs() < 1e-6);
        assert!((attn[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn visibility_is_causal_and_bounded_by_the_window() {
        assert!(visible(5, 5, None));
        assert!(!visible(5, 6, None));
        assert!(visible(1023, 0, Some(1024)));
        assert!(
            !visible(1024, 0, Some(1024)),
            "the window excludes distance 1024"
        );
        assert!(visible(1024, 1, Some(1024)));
        // The window never lets a future token in, even when it is close.
        assert!(!visible(3, 4, Some(1024)));
    }

    #[test]
    fn zero_window_hides_everything() {
        assert!(!visible(0, 0, Some(0)));
        assert!(!visible(10, 9, Some(0)));
    }
}
