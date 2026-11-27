//! `linear_nvfp4` against the CPU dequantiser, on **real checkpoint rows**.
//!
//! NVFP4 is 36 bytes per 64 elements: four unsigned-E4M3 scales (one per 16-element
//! sub-block) and 32 bytes of packed E2M1 nibbles, with the low nibble holding element
//! `s*16 + j` and the high nibble `s*16 + 8 + j`. Every one of those details is a silent
//! error if it is wrong -- the output stays finite and merely wrong -- so the oracle is
//! `scoria_gguf::dequant::dequant_row_nvfp4` reading the same bytes out of the GGUF.
//!
//! Synthetic data would not catch a scale-ordering bug, because a synthetic block can be
//! built so that every sub-block shares one scale. The checkpoint cannot.

use std::ffi::c_void;

use scoria_engine::EngineContext;
use scoria_gguf::GgufFile;
use scoria_hip_rt::{DeviceBuf, HipStream};

const GGUF: &str = "/home/noorm/models/inkling/Inkling-2-Mini-UD-Q3_K_XL.gguf";

fn to_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn gemv_nvfp4(ctx: &EngineContext, w: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
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
    // `kernelParams` entries are the *address* of each argument, pointers included.
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
                ctx.infer.linear_nvfp4,
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

/// Reads `rows` consecutive rows of an NVFP4 tensor starting at `first_row`.
fn read_rows(file: &GgufFile, name: &str, first_row: usize, rows: usize, k: usize) -> Vec<u8> {
    let info = file.tensor(name).expect("tensor present");
    assert_eq!(info.ggml_type, scoria_gguf::GgmlType::NVFP4, "{name}");
    let row_bytes = info.ggml_type.tensor_byte_len(k).unwrap();
    let all = file.tensor_bytes(info).unwrap();
    all[first_row * row_bytes..(first_row + rows) * row_bytes].to_vec()
}

fn cpu_reference(w: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let row_bytes = (k / 64) * 36;
    let mut out = vec![0f32; n];
    let mut row = vec![0f32; k];
    for r in 0..n {
        scoria_gguf::dequant::dequant_row_nvfp4(&w[r * row_bytes..(r + 1) * row_bytes], &mut row);
        out[r] = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
    out
}

#[test]
fn nvfp4_gemv_matches_the_cpu_dequantiser_on_real_rows() {
    let file = GgufFile::open(GGUF).expect("checkpoint");
    let ctx = EngineContext::init(0).expect("HIP context");

    // A layer whose gate_up experts are NVFP4, taking the first 64 rows of expert 0: the
    // full `[2*ff, n_embd]` row width with k = 2304 (36 blocks of 64) and n = 64.
    let name = "blk.5.ffn_gate_up_exps.weight";
    let k = 2304usize;
    let n = 64usize;
    let slab_rows = 2304usize; // rows per expert
    let w = read_rows(&file, name, 0, n, k);
    assert_eq!(w.len(), n * (k / 64) * 36);

    // An input with a distinct value per element, so a sub-block ordering error shows up.
    let x: Vec<f32> = (0..k)
        .map(|i| ((i * 7919 % 2003) as f32 - 1001.0) * 1e-4)
        .collect();
    let want = cpu_reference(&w, &x, n, k);
    let got = gemv_nvfp4(&ctx, &w, &x, n, k);
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        worst = worst.max((g - w).abs());
    }
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(got.iter().all(|v| v.is_finite()), "non-finite output");
    assert!(
        scale > 0.0 && worst <= scale * 5e-5,
        "{name}: worst |diff| {worst} against scale {scale}"
    );

    // The same block decoded twice must agree exactly: no dependence on lane or launch.
    let again = gemv_nvfp4(&ctx, &w, &x, n, k);
    assert_eq!(got, again);
    assert!(slab_rows > n);

    // And a second layer, to cover a different tensor shape being NVFP4 (down: k = 1152).
    let name = "blk.0.ffn_down_exps.weight";
    let k = 1152usize;
    let n = 32usize;
    let w = read_rows(&file, name, 0, n, k);
    let x: Vec<f32> = (0..k)
        .map(|i| ((i * 31 % 17) as f32 - 8.0) * 1e-3)
        .collect();
    let want = cpu_reference(&w, &x, n, k);
    let got = gemv_nvfp4(&ctx, &w, &x, n, k);
    let mut worst = 0.0f32;
    for (g, w) in got.iter().zip(&want) {
        worst = worst.max((g - w).abs());
    }
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        scale > 0.0 && worst <= scale * 5e-5,
        "{name}: worst |diff| {worst} against scale {scale}"
    );
}
