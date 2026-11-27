use std::ffi::c_void;
use std::ptr;

use scoria_engine::EngineContext;
use scoria_gguf::GgmlType;
use scoria_gguf::dequant::{dequant_row, f16_to_f32};
use scoria_hip_rt::{DeviceBuf, HipStream, KernelHandle};

fn to_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

fn to_bytes_mut<T>(slice: &mut [T]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u8, std::mem::size_of_val(slice))
    }
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 255 {
        return if mant == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00
        };
    }
    let exp_f16 = exp - 127 + 15;
    if exp_f16 >= 31 {
        return sign | 0x7c00;
    }
    if exp_f16 <= 0 {
        if exp_f16 < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (14 - exp_f16);
        return sign | ((m + 1) >> 1) as u16;
    }
    let half_mant = (mant + 0x1000) >> 13;
    if (half_mant & 0x0400) != 0 {
        return sign | (((exp_f16 + 1) as u16) << 10);
    }
    sign | ((exp_f16 as u16) << 10) | (half_mant as u16)
}

fn round_f16(v: f32) -> f32 {
    f16_to_f32(f32_to_f16_bits(v))
}

enum ArgVal {
    Ptr(*mut c_void),
    U32(u32),
    F32(f32),
}

impl ArgVal {
    fn as_ptr(&mut self) -> *mut c_void {
        match self {
            ArgVal::Ptr(p) => (p as *mut *mut c_void) as *mut c_void,
            ArgVal::U32(u) => (u as *mut u32) as *mut c_void,
            ArgVal::F32(f) => (f as *mut f32) as *mut c_void,
        }
    }
}

fn launch(
    ctx: &EngineContext,
    stream: &HipStream,
    handle: KernelHandle,
    grid: [u32; 3],
    block: [u32; 3],
    mut args: Vec<ArgVal>,
) {
    let mut ptrs: Vec<*mut c_void> = args.iter_mut().map(|a| a.as_ptr()).collect();
    unsafe {
        ctx.infer_module
            .launch_handle(handle, grid, block, 0, stream, &mut ptrs)
            .expect("kernel launch failed");
    }
}

#[test]
fn adversarial_gqa_attn_f16_fallback_and_fast_paths() {
    let ctx = EngineContext::init(0).expect("EngineContext::init(0)");
    let stream = HipStream::new().expect("HipStream::new");

    for &(head_dim, kv_len, m) in &[
        (32usize, 1usize, 1usize),
        (32usize, 17usize, 2usize),
        (96usize, 1usize, 1usize),
        (96usize, 17usize, 2usize),
        (128usize, 17usize, 2usize),
    ] {
        let n_heads = 4usize;
        let n_kv_heads = 2usize;
        let start_pos = kv_len - 1;
        let total_kv = start_pos + m;

        let q: Vec<f32> = (0..m * n_heads * head_dim)
            .map(|i| (((i * 13 + 5) % 29) as f32 - 14.0) * 0.05)
            .collect();
        let k_f16_bits: Vec<u16> = (0..total_kv * n_kv_heads * head_dim)
            .map(|i| f32_to_f16_bits((((i * 7 + 3) % 23) as f32 - 11.0) * 0.06))
            .collect();
        let v_f16_bits: Vec<u16> = (0..total_kv * n_kv_heads * head_dim)
            .map(|i| f32_to_f16_bits((((i * 11 + 1) % 31) as f32 - 15.0) * 0.04))
            .collect();

        // CPU reference using FP16-rounded K and V
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let kv_group = n_heads / n_kv_heads;
        let mut cpu_out = vec![0.0f32; m * n_heads * head_dim];
        for tok in 0..m {
            let valid_kv = start_pos + tok + 1;
            for h in 0..n_heads {
                let kvh = h / kv_group;
                let q_slice =
                    &q[(tok * n_heads + h) * head_dim..(tok * n_heads + h + 1) * head_dim];
                let mut scores = Vec::with_capacity(valid_kv);
                let mut max_s = f32::NEG_INFINITY;
                for t in 0..valid_kv {
                    let k_off = (t * n_kv_heads + kvh) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_slice[d] * f16_to_f32(k_f16_bits[k_off + d]);
                    }
                    let s = dot * scale;
                    max_s = max_s.max(s);
                    scores.push(s);
                }
                let mut l_sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    l_sum += *s;
                }
                let out_slice = &mut cpu_out
                    [(tok * n_heads + h) * head_dim..(tok * n_heads + h + 1) * head_dim];
                for (t, &w) in scores.iter().enumerate() {
                    let v_off = (t * n_kv_heads + kvh) * head_dim;
                    for d in 0..head_dim {
                        out_slice[d] += (w / l_sum) * f16_to_f32(v_f16_bits[v_off + d]);
                    }
                }
            }
        }

        let mut d_q = DeviceBuf::alloc(q.len() * 4).unwrap();
        let mut d_k = DeviceBuf::alloc(k_f16_bits.len() * 2).unwrap();
        let mut d_v = DeviceBuf::alloc(v_f16_bits.len() * 2).unwrap();
        let d_out = DeviceBuf::alloc(cpu_out.len() * 4).unwrap();
        let mut d_pos = DeviceBuf::alloc(4).unwrap();
        d_q.copy_from_host(to_bytes(&q)).unwrap();
        d_k.copy_from_host(to_bytes(&k_f16_bits)).unwrap();
        d_v.copy_from_host(to_bytes(&v_f16_bits)).unwrap();
        d_pos
            .copy_from_host(&(start_pos as u32).to_le_bytes())
            .unwrap();

        launch(
            &ctx,
            &stream,
            ctx.infer.gqa_attn_f16,
            [n_heads as u32, m as u32, 1],
            [ctx.wave_size * 4, 1, 1],
            vec![
                ArgVal::Ptr(d_q.as_ptr()),
                ArgVal::Ptr(d_k.as_ptr()),
                ArgVal::Ptr(d_v.as_ptr()),
                ArgVal::Ptr(d_out.as_ptr()),
                ArgVal::Ptr(d_pos.as_ptr()),
                ArgVal::U32(m as u32),
                ArgVal::U32(n_heads as u32),
                ArgVal::U32(n_kv_heads as u32),
                ArgVal::U32(head_dim as u32),
            ],
        );
        stream.synchronize().unwrap();
        let mut gpu_out = vec![0.0f32; cpu_out.len()];
        d_out.copy_to_host(to_bytes_mut(&mut gpu_out)).unwrap();
        let max_diff = cpu_out
            .iter()
            .zip(&gpu_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "gqa_attn_f16 parity failed for head_dim={head_dim} kv_len={kv_len} m={m}: max_diff={max_diff}"
        );
    }
}

#[test]
fn adversarial_argmax_sentinels_ties_and_stage2_loop() {
    let ctx = EngineContext::init(0).expect("EngineContext::init(0)");
    let stream = HipStream::new().expect("HipStream::new");

    let run_argmax = |vals: &[f32], grid_x: u32| -> u32 {
        let mut d_vals = DeviceBuf::alloc(vals.len() * 4).unwrap();
        let d_s1_v = DeviceBuf::alloc(grid_x as usize * 4).unwrap();
        let d_s1_i = DeviceBuf::alloc(grid_x as usize * 4).unwrap();
        let d_out = DeviceBuf::alloc(4).unwrap();
        d_vals.copy_from_host(to_bytes(vals)).unwrap();

        launch(
            &ctx,
            &stream,
            ctx.infer.argmax_stage1_f32,
            [grid_x, 1, 1],
            [256, 1, 1],
            vec![
                ArgVal::Ptr(d_vals.as_ptr()),
                ArgVal::Ptr(d_s1_v.as_ptr()),
                ArgVal::Ptr(d_s1_i.as_ptr()),
                ArgVal::U32(vals.len() as u32),
            ],
        );
        launch(
            &ctx,
            &stream,
            ctx.infer.argmax_stage2_f32,
            [1, 1, 1],
            [64, 1, 1],
            vec![
                ArgVal::Ptr(d_s1_v.as_ptr()),
                ArgVal::Ptr(d_s1_i.as_ptr()),
                ArgVal::Ptr(d_out.as_ptr()),
                ArgVal::U32(grid_x),
            ],
        );
        stream.synchronize().unwrap();
        let mut out = [0u32; 1];
        d_out.copy_to_host(to_bytes_mut(&mut out)).unwrap();
        out[0]
    };

    // 1. All -INFINITY -> index 0 must win
    let all_neginf = vec![f32::NEG_INFINITY; 300];
    assert_eq!(run_argmax(&all_neginf, 16), 0);

    // 2. All -3e38 except last index at -2e38 (below old -1e38 sentinel)
    let mut below_sentinel = vec![-3.0e38f32; 515];
    below_sentinel[514] = -2.0e38f32;
    assert_eq!(run_argmax(&below_sentinel, 32), 514);

    // 3. Equal-value ties -> lowest index wins
    let mut ties = vec![-5.0f32; 513];
    ties[7] = 42.0;
    ties[257] = 42.0;
    ties[512] = 42.0;
    assert_eq!(run_argmax(&ties, 16), 7);

    // 4. Direct stage2 with count = 130 > blockDim.x (64)
    let s2_vals: Vec<f32> = (0..130)
        .map(|i| if i == 129 { 99.5 } else { i as f32 * 0.5 })
        .collect();
    let s2_idxs: Vec<u32> = (0..130).map(|i| 1000 + i).collect();
    let mut d_v = DeviceBuf::alloc(s2_vals.len() * 4).unwrap();
    let mut d_i = DeviceBuf::alloc(s2_idxs.len() * 4).unwrap();
    let d_o = DeviceBuf::alloc(4).unwrap();
    d_v.copy_from_host(to_bytes(&s2_vals)).unwrap();
    d_i.copy_from_host(to_bytes(&s2_idxs)).unwrap();
    launch(
        &ctx,
        &stream,
        ctx.infer.argmax_stage2_f32,
        [1, 1, 1],
        [64, 1, 1],
        vec![
            ArgVal::Ptr(d_v.as_ptr()),
            ArgVal::Ptr(d_i.as_ptr()),
            ArgVal::Ptr(d_o.as_ptr()),
            ArgVal::U32(130),
        ],
    );
    stream.synchronize().unwrap();
    let mut stage2_out = [0u32; 1];
    d_o.copy_to_host(to_bytes_mut(&mut stage2_out)).unwrap();
    assert_eq!(stage2_out[0], 1129);
}

#[test]
fn adversarial_quantize_q8_1_non_multiple_of_32() {
    let ctx = EngineContext::init(0).expect("EngineContext::init(0)");
    if !ctx.arch.is_rdna4() {
        return;
    }
    let stream = HipStream::new().expect("HipStream::new");

    let k = 33usize;
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) - 16.0) * 0.125).collect();
    let n_blocks = k.div_ceil(32);

    let mut d_x = DeviceBuf::alloc(k * 4).unwrap();
    let d_out = DeviceBuf::alloc(n_blocks * 48).unwrap();
    d_x.copy_from_host(to_bytes(&x)).unwrap();

    launch(
        &ctx,
        &stream,
        ctx.infer.quantize_q8_1_f32,
        [n_blocks as u32, 1, 1],
        [ctx.wave_size, 1, 1],
        vec![
            ArgVal::Ptr(d_x.as_ptr()),
            ArgVal::Ptr(d_out.as_ptr()),
            ArgVal::U32(k as u32),
        ],
    );
    stream.synchronize().unwrap();
    let mut gpu_bytes = vec![0u8; n_blocks * 48];
    d_out.copy_to_host(&mut gpu_bytes).unwrap();

    for b in 0..n_blocks {
        let mut blk_x = [0.0f32; 32];
        for (lane, slot) in blk_x.iter_mut().enumerate() {
            let idx = b * 32 + lane;
            if idx < k {
                *slot = x[idx];
            }
        }
        let amax = blk_x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d_ref = amax / 127.0;
        let inv_d = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        let mut sum_q = 0i32;
        let mut q_ref = [0i8; 32];
        for lane in 0..32 {
            let q = (blk_x[lane] * inv_d).round_ties_even().clamp(-127.0, 127.0) as i8;
            q_ref[lane] = q;
            sum_q += q as i32;
        }
        let s_ref = d_ref * sum_q as f32;

        let blk = &gpu_bytes[b * 48..(b + 1) * 48];
        for lane in 0..32 {
            let q_gpu = blk[lane] as i8;
            assert!(
                (q_gpu as i32 - q_ref[lane] as i32).abs() <= 1,
                "q8_1 mismatch at blk={b} lane={lane}: gpu={q_gpu} ref={}",
                q_ref[lane]
            );
        }
        let d_gpu = f32::from_le_bytes(blk[32..36].try_into().unwrap());
        let s_gpu = f32::from_le_bytes(blk[36..40].try_into().unwrap());
        assert!(
            (d_gpu - d_ref).abs() < 1e-6,
            "blk={b} d: {d_gpu} vs {d_ref}"
        );
        assert!(
            (s_gpu - s_ref).abs() < 1e-4,
            "blk={b} s: {s_gpu} vs {s_ref}"
        );
    }
}

#[test]
fn adversarial_gemm_wmma_q4_k_odd_shapes_and_zero_m() {
    let ctx = EngineContext::init(0).expect("EngineContext::init(0)");
    let Some(handle) = ctx.infer.gemm_wmma_q4_k else {
        return;
    };
    let stream = HipStream::new().expect("HipStream::new");

    // Helper constructing valid 144-byte Q4_K blocks for a given shape (n, k=256)
    let build_q4k_weights = |n: usize, k: usize| -> (Vec<u8>, Vec<f32>) {
        assert_eq!(k % 256, 0);
        let nb = k / 256;
        let mut raw = vec![0u8; n * nb * 144];
        for row in 0..n {
            for b in 0..nb {
                let off = (row * nb + b) * 144;
                let d_bits = f32_to_f16_bits(0.02 + ((row + b) % 5) as f32 * 0.005);
                let dmin_bits = f32_to_f16_bits(0.01 + ((row * 3 + b) % 4) as f32 * 0.003);
                raw[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
                raw[off + 2..off + 4].copy_from_slice(&dmin_bits.to_le_bytes());
                for j in 0..12 {
                    raw[off + 4 + j] = (((row * 7 + b * 3 + j) % 63) + 1) as u8;
                }
                for j in 0..128 {
                    let lo = ((row + b + j) % 15) as u8;
                    let hi = ((row * 3 + b + j * 5) % 15) as u8;
                    raw[off + 16 + j] = lo | (hi << 4);
                }
            }
        }
        let mut deq = vec![0.0f32; n * k];
        for row in 0..n {
            dequant_row(
                GgmlType::Q4K,
                &raw[row * nb * 144..(row + 1) * nb * 144],
                &mut deq[row * k..(row + 1) * k],
            )
            .unwrap();
        }
        (raw, deq)
    };

    for &(m, n, k) in &[(1usize, 1usize, 256usize), (17, 65, 256), (65, 129, 256)] {
        let (w_raw, w_deq) = build_q4k_weights(n, k);
        let x: Vec<f32> = (0..m * k)
            .map(|i| (((i * 11 + 3) % 29) as f32 - 14.0) * 0.02)
            .collect();
        let bias: Vec<f32> = (0..n).map(|i| (i as f32 - 10.0) * 0.01).collect();

        let mut cpu_y = vec![0.0f32; m * n];
        for tok in 0..m {
            for row in 0..n {
                let mut acc = bias[row];
                for c in 0..k {
                    acc += w_deq[row * k + c] * x[tok * k + c];
                }
                cpu_y[tok * n + row] = acc;
            }
        }

        let mut d_w = DeviceBuf::alloc(w_raw.len()).unwrap();
        let mut d_x = DeviceBuf::alloc(x.len() * 4).unwrap();
        let mut d_b = DeviceBuf::alloc(bias.len() * 4).unwrap();
        let d_y = DeviceBuf::alloc(cpu_y.len() * 4).unwrap();
        d_w.copy_from_host(&w_raw).unwrap();
        d_x.copy_from_host(to_bytes(&x)).unwrap();
        d_b.copy_from_host(to_bytes(&bias)).unwrap();

        launch(
            &ctx,
            &stream,
            handle,
            [(n as u32).div_ceil(128), (m as u32).div_ceil(64), 1],
            [256, 1, 1],
            vec![
                ArgVal::Ptr(d_w.as_ptr()),
                ArgVal::Ptr(d_x.as_ptr()),
                ArgVal::Ptr(d_b.as_ptr()),
                ArgVal::Ptr(d_y.as_ptr()),
                ArgVal::U32(m as u32),
                ArgVal::U32(n as u32),
                ArgVal::U32(k as u32),
            ],
        );
        stream.synchronize().unwrap();
        let mut gpu_y = vec![0.0f32; cpu_y.len()];
        d_y.copy_to_host(to_bytes_mut(&mut gpu_y)).unwrap();
        let max_diff = cpu_y
            .iter()
            .zip(&gpu_y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-1,
            "gemm_wmma_q4_k parity failed at m={m} n={n} k={k}: max_diff={max_diff}"
        );
    }

    // m = 0 must be a safe no-op leaving sentinel buffer untouched
    let (w_raw, _) = build_q4k_weights(128, 256);
    let mut d_w = DeviceBuf::alloc(w_raw.len()).unwrap();
    let d_x = DeviceBuf::alloc(256 * 4).unwrap();
    let mut d_y = DeviceBuf::alloc(128 * 4).unwrap();
    d_w.copy_from_host(&w_raw).unwrap();
    let sentinel = vec![123.25f32; 128];
    d_y.copy_from_host(to_bytes(&sentinel)).unwrap();
    launch(
        &ctx,
        &stream,
        handle,
        [1, 1, 1],
        [256, 1, 1],
        vec![
            ArgVal::Ptr(d_w.as_ptr()),
            ArgVal::Ptr(d_x.as_ptr()),
            ArgVal::Ptr(ptr::null_mut()),
            ArgVal::Ptr(d_y.as_ptr()),
            ArgVal::U32(0),
            ArgVal::U32(128),
            ArgVal::U32(256),
        ],
    );
    stream.synchronize().unwrap();
    let mut out = vec![0.0f32; 128];
    d_y.copy_to_host(to_bytes_mut(&mut out)).unwrap();
    assert_eq!(out, sentinel);
}

#[test]
fn adversarial_qk_norm_rope_kv_f32_and_f16() {
    let ctx = EngineContext::init(0).expect("EngineContext::init(0)");
    let stream = HipStream::new().expect("HipStream::new");
    let freq_base = 1_000_000.0f32;
    let eps = 1e-6f32;

    for &(head_dim, pos, use_f16) in &[
        (128usize, 0u32, false),
        (128usize, 16384u32, false),
        (64usize, 16384u32, false),
        (128usize, 16384u32, true),
        (64usize, 16384u32, true),
    ] {
        let m = 2usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;

        let q_init: Vec<f32> = (0..m * n_heads * head_dim)
            .map(|i| (((i * 13 + 7) % 37) as f32 - 18.0) * 0.05)
            .collect();
        let k_in: Vec<f32> = (0..m * n_kv_heads * head_dim)
            .map(|i| (((i * 17 + 3) % 31) as f32 - 15.0) * 0.04)
            .collect();
        let v_in: Vec<f32> = (0..m * n_kv_heads * head_dim)
            .map(|i| (((i * 19 + 5) % 29) as f32 - 14.0) * 0.03)
            .collect();
        let q_norm_w: Vec<f32> = (0..head_dim).map(|i| 0.75 + i as f32 * 0.004).collect();
        let k_norm_w: Vec<f32> = (0..head_dim).map(|i| 1.10 - i as f32 * 0.003).collect();

        // CPU f64 reference
        let half = head_dim / 2;
        let mut cpu_q = q_init.clone();
        let cache_slots = pos as usize + m;
        let mut cpu_kc = vec![0.0f32; cache_slots * n_kv_heads * head_dim];
        let mut cpu_vc = vec![0.0f32; cache_slots * n_kv_heads * head_dim];

        for tok in 0..m {
            let p = pos as f64 + tok as f64;
            for h in 0..n_heads {
                let off = (tok * n_heads + h) * head_dim;
                let s = &mut cpu_q[off..off + head_dim];
                let ss: f64 = s.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let inv = 1.0 / (ss / head_dim as f64 + eps as f64).sqrt();
                for i in 0..half {
                    let x0 = s[i] as f64 * inv * q_norm_w[i] as f64;
                    let x1 = s[i + half] as f64 * inv * q_norm_w[i + half] as f64;
                    let inv_freq = (freq_base as f64).powf(-2.0 * i as f64 / head_dim as f64);
                    let theta = p * inv_freq;
                    let (sv, cv) = theta.sin_cos();
                    s[i] = (x0 * cv - x1 * sv) as f32;
                    s[i + half] = (x0 * sv + x1 * cv) as f32;
                }
            }
            for kvh in 0..n_kv_heads {
                let in_off = (tok * n_kv_heads + kvh) * head_dim;
                let dst_off = ((pos as usize + tok) * n_kv_heads + kvh) * head_dim;
                let ks = &k_in[in_off..in_off + head_dim];
                let ss: f64 = ks.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let inv = 1.0 / (ss / head_dim as f64 + eps as f64).sqrt();
                for i in 0..half {
                    let x0 = ks[i] as f64 * inv * k_norm_w[i] as f64;
                    let x1 = ks[i + half] as f64 * inv * k_norm_w[i + half] as f64;
                    let inv_freq = (freq_base as f64).powf(-2.0 * i as f64 / head_dim as f64);
                    let theta = p * inv_freq;
                    let (sv, cv) = theta.sin_cos();
                    let o0 = (x0 * cv - x1 * sv) as f32;
                    let o1 = (x0 * sv + x1 * cv) as f32;
                    cpu_kc[dst_off + i] = if use_f16 { round_f16(o0) } else { o0 };
                    cpu_kc[dst_off + i + half] = if use_f16 { round_f16(o1) } else { o1 };
                }
                for d in 0..head_dim {
                    let v = v_in[in_off + d];
                    cpu_vc[dst_off + d] = if use_f16 { round_f16(v) } else { v };
                }
            }
        }

        let elem_bytes = if use_f16 { 2 } else { 4 };
        let mut d_q = DeviceBuf::alloc(q_init.len() * 4).unwrap();
        let mut d_kin = DeviceBuf::alloc(k_in.len() * 4).unwrap();
        let mut d_vin = DeviceBuf::alloc(v_in.len() * 4).unwrap();
        let mut d_qw = DeviceBuf::alloc(q_norm_w.len() * 4).unwrap();
        let mut d_kw = DeviceBuf::alloc(k_norm_w.len() * 4).unwrap();
        let mut d_kc = DeviceBuf::alloc(cpu_kc.len() * elem_bytes).unwrap();
        let mut d_vc = DeviceBuf::alloc(cpu_vc.len() * elem_bytes).unwrap();
        let mut d_pos = DeviceBuf::alloc(4).unwrap();
        d_kc.zero_async(&stream).unwrap();
        d_vc.zero_async(&stream).unwrap();
        d_q.copy_from_host(to_bytes(&q_init)).unwrap();
        d_kin.copy_from_host(to_bytes(&k_in)).unwrap();
        d_vin.copy_from_host(to_bytes(&v_in)).unwrap();
        d_qw.copy_from_host(to_bytes(&q_norm_w)).unwrap();
        d_kw.copy_from_host(to_bytes(&k_norm_w)).unwrap();
        d_pos.copy_from_host(&pos.to_le_bytes()).unwrap();

        let handle = if use_f16 {
            ctx.infer.qk_norm_rope_kv_f16
        } else {
            ctx.infer.qk_norm_rope_kv_f32
        };
        launch(
            &ctx,
            &stream,
            handle,
            [m as u32, (n_heads + n_kv_heads) as u32, 1],
            [ctx.wave_size, 1, 1],
            vec![
                ArgVal::Ptr(d_q.as_ptr()),
                ArgVal::Ptr(d_kin.as_ptr()),
                ArgVal::Ptr(d_vin.as_ptr()),
                ArgVal::Ptr(d_qw.as_ptr()),
                ArgVal::Ptr(d_kw.as_ptr()),
                ArgVal::Ptr(d_kc.as_ptr()),
                ArgVal::Ptr(d_vc.as_ptr()),
                ArgVal::Ptr(d_pos.as_ptr()),
                ArgVal::U32(m as u32),
                ArgVal::U32(n_heads as u32),
                ArgVal::U32(n_kv_heads as u32),
                ArgVal::U32(head_dim as u32),
                ArgVal::F32(freq_base),
                ArgVal::F32(eps),
            ],
        );
        stream.synchronize().unwrap();

        let mut gpu_q = vec![0.0f32; cpu_q.len()];
        d_q.copy_to_host(to_bytes_mut(&mut gpu_q)).unwrap();
        let q_diff = cpu_q
            .iter()
            .zip(&gpu_q)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // At pos=16384, multiplying a 24-bit FP32 inv_freq by 16384 leaves ~10 fractional bits
        // of angle (~1e-3 rad) relative to f64 sin/cos; verify exact <1e-5 at pos=0 and <2e-3 at pos=16384.
        let q_tol_f64 = if pos == 0 { 1e-5 } else { 2e-3 };
        assert!(
            q_diff < q_tol_f64,
            "qk_norm_rope_kv Q diff vs f64 too large (head_dim={head_dim}, pos={pos}, f16={use_f16}): {q_diff}"
        );

        let k_diff = if use_f16 {
            let mut gpu_k_bits = vec![0u16; cpu_kc.len()];
            d_kc.copy_to_host(to_bytes_mut(&mut gpu_k_bits)).unwrap();
            cpu_kc
                .iter()
                .zip(&gpu_k_bits)
                .map(|(&a, &b)| (a - f16_to_f32(b)).abs())
                .fold(0.0f32, f32::max)
        } else {
            let mut gpu_k = vec![0.0f32; cpu_kc.len()];
            d_kc.copy_to_host(to_bytes_mut(&mut gpu_k)).unwrap();
            cpu_kc
                .iter()
                .zip(&gpu_k)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };
        let k_tol = if use_f16 {
            5e-3
        } else if pos == 0 {
            1e-5
        } else {
            2e-3
        };
        assert!(
            k_diff < k_tol,
            "qk_norm_rope_kv K diff vs f64 too large (head_dim={head_dim}, pos={pos}, f16={use_f16}): {k_diff}"
        );
    }
}
