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
/// case. The bias for query `pos_q` and key `pos_k` is
/// `rel_states[h] . bank[.., pos_q - pos_k]`, and zero when that distance is outside
/// `[0, rel_extent)`.
///
/// `bank` is the GGUF tensor as stored, which is the **transpose** of the reference's
/// `proj: [d_rel, rel_extent]`: the file carries it as a GGML matrix with `in = d_rel`
/// (dims `[d_rel, rel_extent]`, so the flat index of entry `(r, distance)` is
/// `distance * d_rel + r`). Reading it `[r * rel_extent + distance]` instead transposes a
/// 16 x 1024 table and turns every attention bias into noise.
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
                acc += rel_states[h * d_rel + r] * proj[d * d_rel + r];
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

/// Depthwise causal convolution with a 4-tap kernel, matching `inkling2_loop`'s **static**
/// `ShortConvolution`:
///
/// ```text
/// out[t] = (sum_j weight[c][j] * x[t - (3 - j)] + x[t]) / sqrt(kernel + 1)
/// ```
///
/// Every detail here was wrong once, and the tap order is the one that matters most:
///
/// * the tensor is a PyTorch `Conv1d` weight, so it is cross-correlation, **not**
///   convolution: kernel tap `j` multiplies the element `3 - j` back, i.e. tap 0 weights
///   the **oldest** token and tap 3 the current one. The trained taps confirm it: their
///   mean magnitudes are `0.03, 0.04, 0.10, 0.44` for `blk.0.ffn_post_conv`, a gradient
///   that only appears when the tap axis is read as the fastest-varying axis;
/// * the file layout is `weight[channel * 4 + tap]` (GGML `dims = [4, channels]`);
/// * the pre-convolution value is added back as a residual and the sum is divided by
///   `sqrt(kernel + 1)` = `sqrt(5)`. The *dynamic* variant divides by
///   `sqrt(2*kernel + 1)` instead.
///
/// `cache` holds the previous `taps - 1` inputs per channel, newest first, and is updated
/// in place so decode steps chain.
pub fn short_conv1d_4tap(x: &mut [f32], weight: &[f32], cache: &mut [f32], channels: usize) {
    assert_eq!(weight.len(), 4 * channels, "weight is [channels][taps]");
    assert_eq!(cache.len(), 3 * channels, "cache is [taps-1][channels]");
    let scale = (5.0f32).sqrt().recip();
    for c in 0..channels {
        let current = x[c];
        let lag1 = cache[c];
        let lag2 = cache[channels + c];
        let lag3 = cache[2 * channels + c];
        let w = &weight[c * 4..c * 4 + 4];
        let acc = w[0] * lag3 + w[1] * lag2 + w[2] * lag1 + w[3] * current;
        cache[2 * channels + c] = lag2;
        cache[channels + c] = lag1;
        cache[c] = current;
        x[c] = (acc + current) * scale;
    }
}

/// The **dynamic** short convolution used on the full-attention layers' K and V: the taps
/// are a static per-channel weight plus a per-token weight projected from the same input,
/// shared within each KV head.
///
/// ```text
/// w[c][j] = static[c * 4 + j] + dyn[(c / head_dim) * 4 + j]
/// out[t]  = (sum_j w[c][j] * x[t - (3 - j)] + x[t]) / sqrt(2 * kernel + 1)
/// ```
///
/// The same cross-correlation convention as the static convolution: tap 0 weights the
/// oldest token. `dyn` holds one tap vector per KV head, `kv_heads * 4` values, laid out
/// `[head][tap]`. The scale is `sqrt(2*kernel + 1)` = 3 for four taps.
pub fn short_conv1d_4tap_dynamic(
    x: &mut [f32],
    static_w: &[f32],
    dyn_w: &[f32],
    cache: &mut [f32],
    channels: usize,
    head_dim: usize,
) {
    assert_eq!(static_w.len(), 4 * channels);
    assert_eq!(cache.len(), 3 * channels);
    let kv_heads = channels / head_dim;
    assert_eq!(dyn_w.len(), 4 * kv_heads);
    let scale = (9.0f32).sqrt().recip();
    for c in 0..channels {
        let head = c / head_dim;
        let current = x[c];
        let lag1 = cache[c];
        let lag2 = cache[channels + c];
        let lag3 = cache[2 * channels + c];
        let s = &static_w[c * 4..c * 4 + 4];
        let d = &dyn_w[head * 4..head * 4 + 4];
        let acc = (s[0] + d[0]) * lag3
            + (s[1] + d[1]) * lag2
            + (s[2] + d[2]) * lag1
            + (s[3] + d[3]) * current;
        cache[2 * channels + c] = lag2;
        cache[channels + c] = lag1;
        cache[c] = current;
        x[c] = (acc + current) * scale;
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

        // Invariant: after an unweighted RMSNorm a head has norm sqrt(len), whatever the
        // input. This is what made the real layer-0 run trustworthy: 8 kv heads x 128 dims
        // has to come out at exactly sqrt(1024) = 32.
        for len in [4usize, 128, 1024] {
            let mut v: Vec<f32> = (0..len).map(|i| ((i * 37 % 41) as f32) - 20.0).collect();
            unweighted_rmsnorm(&mut v, 1e-6);
            let got = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((got - (len as f32).sqrt()).abs() < 1e-3, "len {len}: {got}");
        }
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
        // With zero taps the conv collapses to (0 + x)/sqrt5; k is then L2-clipped per
        // head at the threshold 2.0. Head 0 = (1,2)/sqrt5 has norm 1.0 and survives; head 1
        // = (3,4)/sqrt5 has norm 2.236 and is scaled to norm 2.
        let inv5 = 1.0 / 5f32.sqrt();
        assert!((k[0] - 1.0 * inv5).abs() < 1e-6, "k[0]={}", k[0]);
        assert!((k[1] - 2.0 * inv5).abs() < 1e-6, "k[1]={}", k[1]);
        let head1_norm = (k[2] * k[2] + k[3] * k[3]).sqrt();
        assert!(
            (head1_norm - 2.0).abs() < 1e-5,
            "clipped head norm {head1_norm}"
        );
        assert!(
            (k[2] / k[3] - 3.0 / 4.0).abs() < 1e-5,
            "direction preserved"
        );
        // v was all ones, so after /sqrt5 and an unweighted RMSNorm (which makes the RMS 1)
        // every element is exactly 1 again - the normalisation cancels the scaling.
        assert!(v.iter().all(|x| (*x - 1.0).abs() < 1e-5), "v={v:?}");
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
        // Hand computed with tap 0 on the oldest input and /sqrt5:
        // step 1: (4*1 + 1)/sqrt5 = 5/sqrt5
        let inv = 1.0 / 5f32.sqrt();
        assert!((seen[0] - 5.0 * inv).abs() < 1e-5, "{}", seen[0]);
        // step 2: (3*1 + 4*2 + 2)/sqrt5 = 13/sqrt5
        assert!((seen[1] - 13.0 * inv).abs() < 1e-5, "{}", seen[1]);
        // step 4: (1*1 + 2*2 + 3*3 + 4*4 + 4)/sqrt5 = 34/sqrt5
        assert!((seen[3] - 34.0 * inv).abs() < 1e-5, "{}", seen[3]);
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
        // Two heads, d_rel = 2, rel_extent = 4. The bank is stored distance-major, as the
        // GGUF stores it: bank[d * d_rel + r].
        let proj = [
            1.0f32, 10.0, // distance 0: r = 0, 1
            2.0, 20.0, // distance 1
            3.0, 30.0, // distance 2
            4.0, 40.0, // distance 3
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
        // Taps [1,2,3,4] on one channel, with tap 0 on the *oldest* input (PyTorch's
        // cross-correlation), then + residual and / sqrt(4+1) = /sqrt5.
        let weight = [1.0f32, 2.0, 3.0, 4.0];
        let inv = 1.0 / 5f32.sqrt();
        let mut cache = [0.0f32; 3];
        let seq = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut streamed = Vec::new();
        for &v in &seq {
            let mut x = [v];
            short_conv1d_4tap(&mut x, &weight, &mut cache, 1);
            streamed.push(x[0]);
        }
        // t=1: (4*1 + 1)/sqrt5 = 5/sqrt5
        assert!((streamed[0] - 5.0 * inv).abs() < 1e-6, "{}", streamed[0]);
        // t=2: (3*1 + 4*2 + 2)/sqrt5 = 13/sqrt5
        assert!((streamed[1] - 13.0 * inv).abs() < 1e-6, "{}", streamed[1]);
        // t=3: (2*1 + 3*2 + 4*3 + 3)/sqrt5 = 23/sqrt5
        assert!((streamed[2] - 23.0 * inv).abs() < 1e-6, "{}", streamed[2]);
        // t=4: (1*1 + 2*2 + 3*3 + 4*4 + 4)/sqrt5 = 34/sqrt5
        assert!((streamed[3] - 34.0 * inv).abs() < 1e-6, "{}", streamed[3]);
        // t=5: (1*2 + 2*3 + 3*4 + 4*5 + 5)/sqrt5 = 45/sqrt5
        assert!((streamed[4] - 45.0 * inv).abs() < 1e-6, "{}", streamed[4]);
        // Tap order is observable: with a cold cache the current token is weighted by
        // tap 3, so a weight of [0,0,0,4] gives (4*x + x)/sqrt5, not x/sqrt5.
        let mut x = [2.0f32];
        let mut cold = [0.0f32; 3];
        short_conv1d_4tap(&mut x, &[0.0, 0.0, 0.0, 4.0], &mut cold, 1);
        assert!((x[0] - 10.0 * inv).abs() < 1e-6, "{}", x[0]);
    }

    #[test]
    fn dynamic_conv_adds_per_head_taps_and_scales_by_three() {
        // Two channels, head_dim 1, so there are two heads: dyn taps differ per channel.
        let static_w = [0.0f32; 8];
        // head 0 tap 3 (the current token) = 1, everything else zero.
        let dyn_w = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let mut cache = [0.0f32; 6];
        let mut x = [5.0f32, 5.0];
        short_conv1d_4tap_dynamic(&mut x, &static_w, &dyn_w, &mut cache, 2, 1);
        // channel 0: (1*5 + 5)/sqrt9 = 10/3; channel 1: (0*5 + 5)/3 = 5/3.
        assert!((x[0] - 10.0 / 3.0).abs() < 1e-6, "{}", x[0]);
        assert!((x[1] - 5.0 / 3.0).abs() < 1e-6, "{}", x[1]);
        assert_eq!(
            &cache[..2],
            &[5.0, 5.0],
            "current token becomes the newest entry"
        );
        // A dynamic tap must change the answer versus the static convolution, which would
        // divide by sqrt(5) and ignore the projection entirely.
        let mut s = [5.0f32, 5.0];
        let mut c2 = [0.0f32; 6];
        short_conv1d_4tap(&mut s, &static_w, &mut c2, 2);
        assert!((s[0] - 5.0 * (5.0f32).sqrt().recip()).abs() < 1e-6);
        assert_ne!(s, x);
    }
}
