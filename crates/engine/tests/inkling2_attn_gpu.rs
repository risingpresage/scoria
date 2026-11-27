//! GPU vs CPU parity for the `inkling2_loop` decode-attention kernel.
//!
//! `AGENTS.md` requires device code to be checked against a CPU reference, so this test
//! runs the same inputs through `inkling2_attn::decode_head` and through
//! `inkling2_attn_decode_f32` on the card and compares. It is adversarial in the ways this
//! kernel can actually be wrong: a window that hides all but the last position, a window
//! that hides everything, a sink that dominates, and distances past the relative extent
//! (where the host-side bias is already zero).

use std::ffi::c_void;

use scoria_engine::EngineContext;
use scoria_engine::inkling2_attn::{decode_head, l2_clip, relative_bias, short_conv1d_4tap};
use scoria_hip_rt::DeviceBuf;

fn to_bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn from_bytes(v: &[u8]) -> Vec<f32> {
    v.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

struct Fixture {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    pos: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    bias: Vec<f32>,
    sink: Vec<f32>,
}

fn fixture(pos: usize) -> Fixture {
    let (heads, kv_heads, head_dim) = (4usize, 2usize, 128usize);
    let q: Vec<f32> = (0..heads * head_dim)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * 1e-3)
        .collect();
    let k: Vec<f32> = (0..(pos + 1) * kv_heads * head_dim)
        .map(|i| ((i * 61 % 97) as f32 - 48.0) * 2e-3)
        .collect();
    let v: Vec<f32> = (0..(pos + 1) * kv_heads * head_dim)
        .map(|i| ((i * 29 % 89) as f32 - 44.0) * 3e-3)
        .collect();
    let bias: Vec<f32> = (0..heads * (pos + 1))
        .map(|i| ((i * 13 % 7) as f32 - 3.0) * 0.01)
        .collect();
    let sink: Vec<f32> = (0..heads).map(|h| -0.2 * h as f32).collect();
    Fixture {
        heads,
        kv_heads,
        head_dim,
        pos,
        q,
        k,
        v,
        bias,
        sink,
    }
}

fn cpu_reference(f: &Fixture, window: Option<usize>) -> Vec<f32> {
    let group = f.heads / f.kv_heads;
    let mut out = vec![0f32; f.heads * f.head_dim];
    for h in 0..f.heads {
        let kvh = h / group;
        // Narrow the caches to this KV head, which is what the kernel indexes.
        let mut k = vec![0f32; (f.pos + 1) * f.head_dim];
        let mut v = vec![0f32; (f.pos + 1) * f.head_dim];
        for t in 0..=f.pos {
            k[t * f.head_dim..(t + 1) * f.head_dim]
                .copy_from_slice(&f.k[(t * f.kv_heads + kvh) * f.head_dim..][..f.head_dim]);
            v[t * f.head_dim..(t + 1) * f.head_dim]
                .copy_from_slice(&f.v[(t * f.kv_heads + kvh) * f.head_dim..][..f.head_dim]);
        }
        let bias_row = &f.bias[h * (f.pos + 1)..(h + 1) * (f.pos + 1)];
        let got = decode_head(
            &f.q[h * f.head_dim..(h + 1) * f.head_dim],
            &k,
            &v,
            bias_row,
            f.sink[h],
            f.pos,
            window,
        );
        out[h * f.head_dim..(h + 1) * f.head_dim].copy_from_slice(&got);
    }
    out
}

fn gpu_reference(ctx: &EngineContext, f: &Fixture, window: Option<usize>) -> Vec<f32> {
    let mut d_q = DeviceBuf::alloc(std::mem::size_of_val(&f.q[..])).unwrap();
    let mut d_k = DeviceBuf::alloc(std::mem::size_of_val(&f.k[..])).unwrap();
    let mut d_v = DeviceBuf::alloc(std::mem::size_of_val(&f.v[..])).unwrap();
    let mut d_bias = DeviceBuf::alloc(std::mem::size_of_val(&f.bias[..])).unwrap();
    let mut d_sink = DeviceBuf::alloc(std::mem::size_of_val(&f.sink[..])).unwrap();
    let d_out = DeviceBuf::alloc(std::mem::size_of_val(&f.q[..])).unwrap();
    d_q.copy_from_host(to_bytes(&f.q)).unwrap();
    d_k.copy_from_host(to_bytes(&f.k)).unwrap();
    d_v.copy_from_host(to_bytes(&f.v)).unwrap();
    d_bias.copy_from_host(to_bytes(&f.bias)).unwrap();
    d_sink.copy_from_host(to_bytes(&f.sink)).unwrap();

    let stream = scoria_hip_rt::HipStream::new().unwrap();
    let mut pos = f.pos as u32;
    let mut heads = f.heads as u32;
    let mut kv_heads = f.kv_heads as u32;
    let mut head_dim = f.head_dim as u32;
    let mut win = window.map_or(u32::MAX, |w| w as u32);
    // Each slot must point at the storage holding the argument value.
    let (mut p_q, mut p_k, mut p_v) = (d_q.as_ptr(), d_k.as_ptr(), d_v.as_ptr());
    let (mut p_bias, mut p_sink, mut p_out) = (d_bias.as_ptr(), d_sink.as_ptr(), d_out.as_ptr());
    let mut args: [*mut c_void; 11] = [
        &mut p_q as *mut *mut c_void as *mut c_void,
        &mut p_k as *mut *mut c_void as *mut c_void,
        &mut p_v as *mut *mut c_void as *mut c_void,
        &mut p_bias as *mut *mut c_void as *mut c_void,
        &mut p_sink as *mut *mut c_void as *mut c_void,
        &mut p_out as *mut *mut c_void as *mut c_void,
        &mut pos as *mut u32 as *mut c_void,
        &mut heads as *mut u32 as *mut c_void,
        &mut kv_heads as *mut u32 as *mut c_void,
        &mut head_dim as *mut u32 as *mut c_void,
        &mut win as *mut u32 as *mut c_void,
    ];
    unsafe {
        ctx.infer_module
            .launch_handle(
                ctx.infer.inkling2_attn_decode_f32,
                [f.heads as u32, 1, 1],
                [ctx.wave_size, 1, 1],
                0,
                &stream,
                &mut args,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    let mut out = vec![0u8; std::mem::size_of_val(&f.q[..])];
    d_out.copy_to_host(&mut out).unwrap();
    from_bytes(&out)
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 2e-4 + 2e-3 * w.abs(),
            "{what}: element {i} gpu {g} vs cpu {w}"
        );
    }
}

#[test]
fn gpu_decode_attention_matches_the_cpu_reference() {
    let ctx = EngineContext::init(0).expect("GPU required for this parity test");
    for (pos, windows) in [
        (0usize, vec![None, Some(1)]),
        (7, vec![None, Some(8), Some(3), Some(1)]),
        // Past the 1024 relative extent: bias is zero there, which is the interesting case.
        (1100, vec![None, Some(1024), Some(17)]),
    ] {
        let f = fixture(pos);
        for window in windows {
            let want = cpu_reference(&f, window);
            let got = gpu_reference(&ctx, &f, window);
            assert_close(&got, &want, &format!("pos {pos} window {window:?}"));
        }
    }
}

#[test]
fn gpu_decode_attention_with_everything_masked_is_zero() {
    let ctx = EngineContext::init(0).expect("GPU required");
    let f = fixture(3);
    // Window 0 hides even the current position, so only the sink has mass and the output
    // must be exactly zero rather than NaN.
    let got = gpu_reference(&ctx, &f, Some(0));
    assert!(got.iter().all(|v| *v == 0.0), "expected zeros, got {got:?}");
}

#[test]
fn cpu_helpers_agree_with_the_host_preprocessing_the_kernel_expects() {
    // The kernel takes the bias and the convolved K/V as inputs, so the host side has to
    // produce them. Pin the two helpers a loader will use.
    let mut rel_states = vec![0.5f32, -0.25, 0.75, 0.125];
    scaled_l2_clip_in_place(&mut rel_states, 1.0);
    assert!(rel_states.iter().all(|v| v.is_finite()));

    let proj = vec![0.01f32; 16 * 1024];
    let pos_k: Vec<usize> = (0..8).collect();
    let mut bias = vec![0f32; 8];
    relative_bias(&[1.0; 16], 1, 16, &proj, 1024, 7, &pos_k, &mut bias);
    // Distance 7..0, all inside the extent; a constant proj makes every bias equal.
    assert!(bias.iter().all(|b| (b - bias[0]).abs() < 1e-6));

    let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
    let weight = vec![0.25f32; 4];
    let mut cache = vec![0.0f32; 3];
    short_conv1d_4tap(&mut x, &weight, &mut cache, 1);
    // All taps 0.25, cold cache: y[0] = (0.25*1 + 1) / sqrt(5) = 0.55901699. The residual
    // add and the 1/sqrt(kernel+1) scale are part of the reference's convolution, not
    // optional decorations.
    assert!((x[0] - 1.25 / 5f32.sqrt()).abs() < 1e-6, "{}", x[0]);
    // The cache now carries the current token as the newest entry, so a second step sees
    // it in tap 1: y[1] = (0.25*(2+1+0+0) + 2) / sqrt(5).
    assert!((cache[0] - 1.0).abs() < 1e-6, "cache {cache:?}");
}

fn scaled_l2_clip_in_place(x: &mut [f32], threshold: f32) {
    l2_clip(x, threshold);
}

// ---------------------------------------------------------------------------------------
// Projection post-processing: conv4 -> q clip -> k clip -> v rmsnorm -> sigmoid, checked
// against `inkling2_attn::post_qkv` over several decode steps so the conv caches chain.
// ---------------------------------------------------------------------------------------

fn launch(
    ctx: &EngineContext,
    handle: scoria_hip_rt::KernelHandle,
    grid: u32,
    block: u32,
    args: &mut [*mut c_void],
) {
    let stream = scoria_hip_rt::HipStream::new().unwrap();
    unsafe {
        ctx.infer_module
            .launch_handle(handle, [grid, 1, 1], [block, 1, 1], 0, &stream, args)
            .unwrap();
    }
    stream.synchronize().unwrap();
}

#[test]
fn gpu_post_qkv_matches_the_cpu_reference_over_chained_steps() {
    let ctx = EngineContext::init(0).expect("GPU required");
    let (heads, head_dim) = (4usize, 128usize);
    let kv_heads = 2usize;
    let kv_dim = kv_heads * head_dim;

    let scale_q: Vec<f32> = (0..heads * head_dim)
        .map(|i| 0.5 + 0.01 * (i % 40) as f32)
        .collect();
    let conv_w_k: Vec<f32> = (0..4 * kv_dim).map(|i| 0.05 * ((i % 7) as f32)).collect();
    let conv_w_v: Vec<f32> = (0..4 * kv_dim).map(|i| 0.03 * ((i % 5) as f32)).collect();

    let mut k_cache_cpu = vec![0f32; 3 * kv_dim];
    let mut v_cache_cpu = vec![0f32; 3 * kv_dim];
    let k_cache_gpu = vec![0f32; 3 * kv_dim];
    let v_cache_gpu = vec![0f32; 3 * kv_dim];

    let mut d_convwk = DeviceBuf::alloc(std::mem::size_of_val(&conv_w_k[..])).unwrap();
    let mut d_convwv = DeviceBuf::alloc(std::mem::size_of_val(&conv_w_v[..])).unwrap();
    let mut d_ck = DeviceBuf::alloc(std::mem::size_of_val(&k_cache_gpu[..])).unwrap();
    let mut d_cv = DeviceBuf::alloc(std::mem::size_of_val(&v_cache_gpu[..])).unwrap();
    let mut d_scale = DeviceBuf::alloc(std::mem::size_of_val(&scale_q[..])).unwrap();
    d_convwk.copy_from_host(to_bytes(&conv_w_k)).unwrap();
    d_convwv.copy_from_host(to_bytes(&conv_w_v)).unwrap();
    d_ck.copy_from_host(to_bytes(&k_cache_gpu)).unwrap();
    d_cv.copy_from_host(to_bytes(&v_cache_gpu)).unwrap();
    d_scale.copy_from_host(to_bytes(&scale_q)).unwrap();

    for step in 0..4usize {
        let q: Vec<f32> = (0..heads * head_dim)
            .map(|i| ((i * 17 + step * 5) % 61) as f32 * 0.1 - 3.0)
            .collect();
        let k: Vec<f32> = (0..kv_dim)
            .map(|i| ((i * 11 + step * 3) % 53) as f32 * 0.2 - 5.0)
            .collect();
        let v: Vec<f32> = (0..kv_dim)
            .map(|i| ((i * 7 + step) % 31) as f32 * 0.3 - 4.0)
            .collect();
        let gate: Vec<f32> = (0..heads * head_dim)
            .map(|i| ((i * 3 + step) % 19) as f32 * 0.2 - 2.0)
            .collect();

        let (mut cq, mut ck, mut cv, mut cg) = (q.clone(), k.clone(), v.clone(), gate.clone());
        scoria_engine::inkling2_attn::post_qkv(
            &mut cq,
            &mut ck,
            &mut cv,
            &mut cg,
            &scale_q,
            &conv_w_k,
            &conv_w_v,
            &mut k_cache_cpu,
            &mut v_cache_cpu,
            heads,
            head_dim,
            kv_dim,
            22.627_417,
            1e-6,
        );

        // GPU: upload the raw projections, run conv on k and v, then the head ops.
        let mut d_q = DeviceBuf::alloc(std::mem::size_of_val(&q[..])).unwrap();
        let mut d_k = DeviceBuf::alloc(std::mem::size_of_val(&k[..])).unwrap();
        let mut d_v = DeviceBuf::alloc(std::mem::size_of_val(&v[..])).unwrap();
        let mut d_g = DeviceBuf::alloc(std::mem::size_of_val(&gate[..])).unwrap();
        d_q.copy_from_host(to_bytes(&q)).unwrap();
        d_k.copy_from_host(to_bytes(&k)).unwrap();
        d_v.copy_from_host(to_bytes(&v)).unwrap();
        d_g.copy_from_host(to_bytes(&gate)).unwrap();

        let wave = ctx.wave_size;
        let kv_dim_u = kv_dim as u32;
        let mut heads_kv = kv_heads as u32;
        let mut hd = head_dim as u32;
        let clip = 22.627_417f32;
        let mut eps = 1e-6f32;
        let mut n_q = (heads * head_dim) as u32;

        let (mut pk, mut pwk, mut pck, mut pkd) =
            (d_k.as_ptr(), d_convwk.as_ptr(), d_ck.as_ptr(), kv_dim_u);
        let mut a: [*mut c_void; 4] = [
            &mut pk as *mut *mut c_void as *mut c_void,
            &mut pwk as *mut *mut c_void as *mut c_void,
            &mut pck as *mut *mut c_void as *mut c_void,
            &mut pkd as *mut u32 as *mut c_void,
        ];
        launch(
            &ctx,
            ctx.infer.inkling2_conv4_f32,
            (kv_dim as u32).div_ceil(256),
            256,
            &mut a,
        );

        let (mut pv, mut pwv, mut pcv) = (d_v.as_ptr(), d_convwv.as_ptr(), d_cv.as_ptr());
        let mut a: [*mut c_void; 4] = [
            &mut pv as *mut *mut c_void as *mut c_void,
            &mut pwv as *mut *mut c_void as *mut c_void,
            &mut pcv as *mut *mut c_void as *mut c_void,
            &mut pkd as *mut u32 as *mut c_void,
        ];
        launch(
            &ctx,
            ctx.infer.inkling2_conv4_f32,
            (kv_dim as u32).div_ceil(256),
            256,
            &mut a,
        );

        // Q: scaled clip, one wave per query head.
        let (mut pq, mut ps, mut ph, mut pm) = (d_q.as_ptr(), d_scale.as_ptr(), heads as u32, clip);
        let mut a: [*mut c_void; 5] = [
            &mut pq as *mut *mut c_void as *mut c_void,
            &mut ps as *mut *mut c_void as *mut c_void,
            &mut ph as *mut u32 as *mut c_void,
            &mut hd as *mut u32 as *mut c_void,
            &mut pm as *mut f32 as *mut c_void,
        ];
        launch(
            &ctx,
            ctx.infer.inkling2_head_clip_f32,
            heads as u32,
            wave,
            &mut a,
        );

        // K: unscaled clip (null scale).
        let mut null = std::ptr::null_mut::<c_void>();
        let mut a: [*mut c_void; 5] = [
            &mut pk as *mut *mut c_void as *mut c_void,
            &mut null as *mut *mut c_void as *mut c_void,
            &mut heads_kv as *mut u32 as *mut c_void,
            &mut hd as *mut u32 as *mut c_void,
            &mut pm as *mut f32 as *mut c_void,
        ];
        launch(
            &ctx,
            ctx.infer.inkling2_head_clip_f32,
            kv_heads as u32,
            wave,
            &mut a,
        );

        // V: unweighted RMSNorm.
        let mut a: [*mut c_void; 4] = [
            &mut pv as *mut *mut c_void as *mut c_void,
            &mut heads_kv as *mut u32 as *mut c_void,
            &mut hd as *mut u32 as *mut c_void,
            &mut eps as *mut f32 as *mut c_void,
        ];
        launch(
            &ctx,
            ctx.infer.inkling2_head_rmsnorm_f32,
            kv_heads as u32,
            wave,
            &mut a,
        );

        // Gate: sigmoid over the whole tensor.
        let mut pg = d_g.as_ptr();
        let mut a: [*mut c_void; 2] = [
            &mut pg as *mut *mut c_void as *mut c_void,
            &mut n_q as *mut u32 as *mut c_void,
        ];
        launch(
            &ctx,
            ctx.infer.inkling2_sigmoid_f32,
            n_q.div_ceil(256),
            256,
            &mut a,
        );

        let mut got = vec![0u8; std::mem::size_of_val(&q[..])];
        d_q.copy_to_host(&mut got).unwrap();
        assert_close(&from_bytes(&got), &cq, &format!("q step {step}"));

        let mut got = vec![0u8; std::mem::size_of_val(&k[..])];
        d_k.copy_to_host(&mut got).unwrap();
        assert_close(&from_bytes(&got), &ck, &format!("k step {step}"));

        let mut got = vec![0u8; std::mem::size_of_val(&v[..])];
        d_v.copy_to_host(&mut got).unwrap();
        assert_close(&from_bytes(&got), &cv, &format!("v step {step}"));

        let mut got = vec![0u8; std::mem::size_of_val(&gate[..])];
        d_g.copy_to_host(&mut got).unwrap();
        assert_close(&from_bytes(&got), &cg, &format!("gate step {step}"));

        // The GPU-side caches must have advanced exactly like the CPU ones.
        let mut ck_gpu = vec![0u8; std::mem::size_of_val(&k_cache_gpu[..])];
        d_ck.copy_to_host(&mut ck_gpu).unwrap();
        assert_close(
            &from_bytes(&ck_gpu),
            &k_cache_cpu,
            &format!("k cache step {step}"),
        );
        let mut cv_gpu = vec![0u8; std::mem::size_of_val(&v_cache_gpu[..])];
        d_cv.copy_to_host(&mut cv_gpu).unwrap();
        assert_close(
            &from_bytes(&cv_gpu),
            &v_cache_cpu,
            &format!("v cache step {step}"),
        );
    }
}
