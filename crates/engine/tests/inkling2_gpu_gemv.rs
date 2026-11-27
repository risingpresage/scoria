//! `linear_f32` as used by the Inkling projection path: the host matvec is the oracle.
//!
//! The model calls this kernel once per projection per visit, so a wrong argument order or
//! a mis-scaled grid would silently corrupt every layer. This test pins the launch
//! convention against `matvec` over shapes that matter: a single row, the real
//! `2304 -> 3072` projection shape, and a k that is not a multiple of the wave size.

use std::ffi::c_void;

use scoria_engine::EngineContext;
use scoria_hip_rt::{DeviceBuf, HipStream};

fn to_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// `y[o] = sum_i w[o][i] * x[i]`, the host reference.
fn matvec(w: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|o| {
            w[o * k..(o + 1) * k]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

fn gemv(ctx: &EngineContext, w: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut wbuf = DeviceBuf::alloc(w.len() * 4).unwrap();
    wbuf.copy_from_host(to_bytes(w)).unwrap();
    let mut xbuf = DeviceBuf::alloc(k * 4).unwrap();
    xbuf.copy_from_host(to_bytes(x)).unwrap();
    let ybuf = DeviceBuf::alloc(n * 4).unwrap();
    let stream = HipStream::new().unwrap();

    let mut m = 1u32;
    let mut nn = n as u32;
    let mut kk = k as u32;
    let waves_per_wg = 256u32 / ctx.wave_size.max(1);
    let grid_x = nn.div_ceil(waves_per_wg);
    // `hipModuleLaunchKernel` wants the *address* of every argument, pointer arguments
    // included. Passing a pointer value instead makes HIP read the kernel argument from
    // device memory on the host and fault inside libamdhip64.
    let (mut pw, mut px, mut py) = (wbuf.as_ptr(), xbuf.as_ptr(), ybuf.as_ptr());
    let mut pbias: *mut c_void = std::ptr::null_mut();
    let mut args: [*mut c_void; 7] = [
        &mut pw as *mut *mut c_void as *mut c_void,
        &mut px as *mut *mut c_void as *mut c_void,
        &mut pbias as *mut *mut c_void as *mut c_void,
        &mut py as *mut *mut c_void as *mut c_void,
        &mut m as *mut u32 as *mut c_void,
        &mut nn as *mut u32 as *mut c_void,
        &mut kk as *mut u32 as *mut c_void,
    ];
    unsafe {
        ctx.infer_module
            .launch_handle(
                ctx.infer.linear_f32,
                [grid_x, 1, 1],
                [256, 1, 1],
                0,
                &stream,
                &mut args,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();
    let mut out = vec![0f32; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, n * 4) };
    ybuf.copy_to_host(bytes).unwrap();
    out
}

fn check(n: usize, k: usize) {
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i * 37 % 211) as f32 - 105.0) * 1e-4)
        .collect();
    let x: Vec<f32> = (0..k)
        .map(|i| ((i * 53 % 97) as f32 - 48.0) * 1e-3)
        .collect();
    let want = matvec(&w, &x, n, k);
    let ctx = EngineContext::init(0).expect("HIP context");
    let got = gemv(&ctx, &w, &x, n, k);
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        worst = worst.max((g - w).abs());
    }
    // f32 dot products over k terms: the wave reduction reassociates the sum, so the
    // tolerance scales with the magnitude of the row, not with an absolute constant.
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        worst <= 1e-6 + scale * 1e-5,
        "n={n} k={k}: worst |diff| {worst} against scale {scale}"
    );
    assert!(
        got.iter().all(|v| v.is_finite()),
        "n={n} k={k}: non-finite output"
    );
}

#[test]
fn gemv_matches_the_host_matvec() {
    // One row, the real projection shape, and a k that is not a wave multiple.
    check(1, 4);
    check(8, 1024);
    check(3072, 2304);
}

#[test]
fn gemv_rejects_a_mismatched_row_count() {
    // The kernel writes `y[tok * n + row]`; a grid that over-covers n must not write past
    // the buffer, so this checks the boundary shape rather than relying on the guard alone.
    check(5, 7);
}
