//! `linear_iq3_xxs` against the CPU dequantiser, on **real checkpoint rows**.
//!
//! IQ3_XXS is 98 bytes per 256 elements: an f16 scale, 64 bytes of 4-bit grid indices and 32
//! bytes of `aux` words (four 7-bit sign selectors and a 4-bit sub-block scale). The grid is
//! a 256-entry codebook transcribed into device constant memory, so this test is the only
//! thing standing between a mistyped table entry and a silently wrong model: it compares the
//! kernel against `dequant_row_iq3_xxs` reading the same bytes from the GGUF.

use std::ffi::c_void;

use scoria_engine::EngineContext;
use scoria_gguf::GgufFile;
use scoria_hip_rt::{DeviceBuf, HipStream};

const GGUF: &str = "/home/noorm/models/inkling/Inkling-2-Mini-UD-Q3_K_XL.gguf";

fn to_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn gemv_iq3xxs(ctx: &EngineContext, w: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
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
                ctx.infer.linear_iq3_xxs,
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
    assert_eq!(info.ggml_type, scoria_gguf::GgmlType::IQ3XXS, "{name}");
    let row_bytes = info.ggml_type.tensor_byte_len(k).unwrap();
    let all = file.tensor_bytes(info).unwrap();
    all[..rows * row_bytes].to_vec()
}

fn cpu_reference(w: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let row_bytes = (k / 256) * 98;
    let mut out = vec![0f32; n];
    let mut row = vec![0f32; k];
    for r in 0..n {
        scoria_gguf::dequant::dequant_row_iq3_xxs(&w[r * row_bytes..(r + 1) * row_bytes], &mut row);
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
    let got = gemv_iq3xxs(ctx, &w, &x, n, k);
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
    assert_eq!(got, gemv_iq3xxs(ctx, &w, &x, n, k));
}

#[test]
fn iq3xxs_gemv_matches_the_cpu_dequantiser_on_real_rows() {
    let file = GgufFile::open(GGUF).expect("checkpoint");
    let ctx = EngineContext::init(0).expect("HIP context");
    // `blk.20.ffn_down_exps` is IQ3_XXS with k = 1152; the first 64 rows are expert 0's.
    check(&file, &ctx, "blk.12.ffn_gate_up_exps.weight", 48, 2304);
    check(&file, &ctx, "blk.18.ffn_gate_up_exps.weight", 24, 2304);
}
