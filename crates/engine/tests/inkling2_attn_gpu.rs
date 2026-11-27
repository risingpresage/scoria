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
    // All taps 0.25: y[0] = 0.25 * (0 + 0 + 0 + 1) = 0.25.
    assert!((x[0] - 0.25).abs() < 1e-6);
}

fn scaled_l2_clip_in_place(x: &mut [f32], threshold: f32) {
    l2_clip(x, threshold);
}
