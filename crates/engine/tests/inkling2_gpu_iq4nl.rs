//! `linear_iq4nl` against the CPU dequantiser, on **real checkpoint rows**.
//!
//! IQ4_NL is 18 bytes per 32 elements: an f16 scale and 16 bytes of packed nibbles indexing a
//! **non-uniform** 16-level codebook. The low nibble of byte `j` is element `j` and the high
//! nibble is element `j + 16`. Every one of those details is silent if wrong -- finite, wrong
//! -- so the oracle is `dequant_row_iq4nl` reading the same bytes from the GGUF.
//!
//! The codebook is the part a synthetic test would miss: a test that only used codes 0 and 15
//! would pass even if the table were sorted wrongly in the middle.

use std::ffi::c_void;

use scoria_engine::EngineContext;
use scoria_gguf::GgufFile;
use scoria_hip_rt::{DeviceBuf, HipStream};

const GGUF: &str = "/home/noorm/models/inkling/Inkling-2-Mini-UD-Q3_K_XL.gguf";

fn to_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn gemv_iq4nl(ctx: &EngineContext, w: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut wbuf = DeviceBuf::alloc(w.len()).unwrap();
    wbuf.copy_from_host(w).unwrap();
    let mut xbuf = DeviceBuf::alloc(k * 4).unwrap();
    xbuf.copy_from_host(to_bytes(x)).unwrap();
    let ybuf = DeviceBuf::alloc(n * 4).unwrap();
    let stream = HipStream::new().unwrap();

    let mut m = 1u32;
    let mut nn = n as u32;
    let mut kk = k as u32;
    let waves_per_wg = 256u32 / ctx.wave_size.max(1);
    let (mut pw, mut px) = (wbuf.as_ptr(), xbuf.as_ptr());
    let mut pbias: *mut c_void = std::ptr::null_mut();
    let mut py = ybuf.as_ptr();
    let mut args: [*mut c_void; 7] = [
        &mut pw as *mut *mut c_void as *mut c_void,
        &mut px as *mut *mut c_void as *mut c_void,
        &mut pbias as *mut *mut c_void as *mut c_void,
        &mut py as *mut *mut c_void as *mut c_void,
        &mut m as *mut u32 as *mut c_void,
        &mut nn as *mut u32 as *mut c_void,
        &mut kk as *mut u32 as *mut c_void,
    ];
    let grid_x = (n as u32).div_ceil(waves_per_wg);
    unsafe {
        ctx.infer_module
            .launch_handle(
                ctx.infer.linear_iq4nl,
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

fn read_rows(file: &GgufFile, name: &str, rows: usize, k: usize) -> Vec<u8> {
    let info = file.tensor(name).expect("tensor present");
    assert_eq!(info.ggml_type, scoria_gguf::GgmlType::IQ4NL, "{name}");
    let row_bytes = info.ggml_type.tensor_byte_len(k).unwrap();
    let all = file.tensor_bytes(info).unwrap();
    all[..rows * row_bytes].to_vec()
}

fn cpu_reference(w: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let row_bytes = (k / 32) * 18;
    let mut out = vec![0f32; n];
    let mut row = vec![0f32; k];
    for r in 0..n {
        scoria_gguf::dequant::dequant_row_iq4_nl(&w[r * row_bytes..(r + 1) * row_bytes], &mut row);
        out[r] = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
    out
}

fn check(file: &GgufFile, ctx: &EngineContext, name: &str, n: usize, k: usize) {
    let w = read_rows(file, name, n, k);
    let x: Vec<f32> = (0..k)
        .map(|i| ((i * 7919 % 2003) as f32 - 1001.0) * 1e-4)
        .collect();
    let want = cpu_reference(&w, &x, n, k);
    let got = gemv_iq4nl(ctx, &w, &x, n, k);
    let mut worst = 0f32;
    for (g, w) in got.iter().zip(&want) {
        worst = worst.max((g - w).abs());
    }
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(got.iter().all(|v| v.is_finite()), "{name}: non-finite");
    assert!(
        scale > 0.0 && worst <= scale * 5e-5,
        "{name}: worst |diff| {worst} against scale {scale}"
    );
    // Determinism: no dependence on lane or launch.
    assert_eq!(got, gemv_iq4nl(ctx, &w, &x, n, k));
}

#[test]
fn iq4nl_gemv_matches_the_cpu_dequantiser_on_real_rows() {
    let file = GgufFile::open(GGUF).expect("checkpoint");
    let ctx = EngineContext::init(0).expect("HIP context");
    // `blk.20.ffn_down_exps` is IQ4_NL with k = 1152; the first 64 rows are expert 0's.
    check(&file, &ctx, "blk.20.ffn_down_exps.weight", 64, 1152);
    check(&file, &ctx, "blk.23.ffn_down_exps.weight", 32, 1152);
}
