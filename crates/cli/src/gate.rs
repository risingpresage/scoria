use std::ffi::c_void;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use scoria_engine::EngineContext;
use scoria_gguf::QuantCodec;
use scoria_hip_rt::{
    DeviceBuf, HipEvent, HipResult, HipStream, PinnedBuf, device_synchronize, mem_info,
};
use scoria_kernels::names;

use crate::uring;

struct TempFileGuard(PathBuf);

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn measure_ms<F: FnMut() -> HipResult<()>>(stream: &HipStream, mut f: F) -> HipResult<f64> {
    let start = HipEvent::new()?;
    let stop = HipEvent::new()?;
    start.record(stream)?;
    f()?;
    stop.record(stream)?;
    stop.synchronize()?;
    Ok(HipEvent::elapsed_ms(&start, &stop)? as f64)
}

fn median_f64(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

fn write_incompressible_file(path: &Path, bytes: usize) -> Result<(), String> {
    let mut f = fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    // Fill 4 MiB buffer with xorshift32 PRNG so Btrfs zstd compression cannot deflate it.
    let mut chunk = vec![0u8; 4 * 1024 * 1024];
    let mut s: u32 = 0x9E37_79B9;
    let mut remaining = bytes;
    while remaining > 0 {
        for word in chunk.as_chunks_mut::<4>().0 {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            word.copy_from_slice(&s.to_le_bytes());
        }
        let n = remaining.min(chunk.len());
        f.write_all(&chunk[..n])
            .map_err(|e| format!("write {}: {e}", path.display()))?;
        remaining -= n;
    }
    f.sync_all()
        .map_err(|e| format!("fsync {}: {e}", path.display()))?;
    Ok(())
}

pub fn run(quick: bool) -> Result<(), Box<dyn std::error::Error>> {
    let ctx = EngineContext::init(0)?;
    let _target_codecs = [
        QuantCodec::H4F8,
        QuantCodec::Q4W32,
        QuantCodec::Bitcos,
        QuantCodec::QtipW32,
    ];
    let (free_b, total_b) = mem_info()?;
    let module = &ctx.gate_module;
    let stream = HipStream::new()?;

    // 1. ROCm 7.16.0 Stability Soak
    let soak_cycles = if quick { 20 } else { 100 };
    let max_safe = (free_b as f64 * 0.72) as usize;
    let gib = 1024 * 1024 * 1024;
    let sizes_gib: Vec<usize> = [1usize, 4, 8]
        .into_iter()
        .filter(|g| g * gib <= max_safe)
        .collect();
    let soak_fn = module.function(names::SOAK_TOUCH)?;
    let mut soak_passes = 0usize;
    let mut soak_min_ms = f64::INFINITY;
    let mut soak_max_ms = 0.0f64;
    let soak_t0 = Instant::now();
    let probe = [0x1234_5678u32.to_le_bytes(), 0x8765_4321u32.to_le_bytes()].concat();

    for c in 0..soak_cycles {
        let cyc_t0 = Instant::now();
        for &g in &sizes_gib {
            let bytes = g * gib;
            let mut dev = DeviceBuf::alloc(bytes)?;
            dev.copy_from_host(&probe[..4])?;
            let mut ptr = dev.as_ptr();
            let mut n_words = (bytes / 4) as u32;
            let mut magic = (c as u32).wrapping_add(1);
            let mut args: [*mut c_void; 3] = [
                (&mut ptr as *mut *mut c_void) as *mut c_void,
                (&mut n_words as *mut u32) as *mut c_void,
                (&mut magic as *mut u32) as *mut c_void,
            ];
            unsafe {
                soak_fn.launch([1, 1, 1], [32, 1, 1], 0, &stream, &mut args)?;
            }
            stream.synchronize()?;
        }
        let dt = cyc_t0.elapsed().as_secs_f64() * 1000.0;
        soak_min_ms = soak_min_ms.min(dt);
        soak_max_ms = soak_max_ms.max(dt);
        soak_passes += 1;
    }
    let soak_total_ms = soak_t0.elapsed().as_secs_f64() * 1000.0;
    device_synchronize()?;

    // 2. VRAM Streaming & L3 Infinity Cache Preservation
    let stream_bytes: usize = 1024 * 1024 * 1024; // 1 GiB
    let stream_vec4 = (stream_bytes / 16) as u32;
    let hot_bytes: usize = 32 * 1024 * 1024; // 32 MiB (inside 64 MiB L3)
    let hot_vec4 = (hot_bytes / 16) as u32;
    let grid_blocks = 1024u32;
    let block_threads = 256u32;
    let out_bytes = (grid_blocks as usize * (block_threads as usize / ctx.wave_size as usize)) * 16;

    let dev_stream = DeviceBuf::alloc(stream_bytes)?;
    let dev_hot = DeviceBuf::alloc(hot_bytes)?;
    let dev_out = DeviceBuf::alloc(out_bytes)?;

    let f_default = module.function(names::STREAM_BW_DEFAULT)?;
    let f_nt = module.function(names::STREAM_BW_NT)?;
    let f_ht = module.function(names::L3_PROBE_HT)?;

    let launch_ht = |iters: u32| -> HipResult<f64> {
        let mut hot_ptr = dev_hot.as_ptr();
        let mut hot_n = hot_vec4;
        let mut out_ptr = dev_out.as_ptr();
        let mut it = iters;
        let mut args: [*mut c_void; 4] = [
            (&mut hot_ptr as *mut *mut c_void) as *mut c_void,
            (&mut hot_n as *mut u32) as *mut c_void,
            (&mut out_ptr as *mut *mut c_void) as *mut c_void,
            (&mut it as *mut u32) as *mut c_void,
        ];
        measure_ms(&stream, || unsafe {
            f_ht.launch(
                [grid_blocks, 1, 1],
                [block_threads, 1, 1],
                0,
                &stream,
                &mut args,
            )
        })
    };

    let launch_stream = |f: &scoria_hip_rt::HipFunction<'_>| -> HipResult<f64> {
        let mut in_ptr = dev_stream.as_ptr();
        let mut out_ptr = dev_out.as_ptr();
        let mut n = stream_vec4;
        let mut args: [*mut c_void; 3] = [
            (&mut in_ptr as *mut *mut c_void) as *mut c_void,
            (&mut out_ptr as *mut *mut c_void) as *mut c_void,
            (&mut n as *mut u32) as *mut c_void,
        ];
        measure_ms(&stream, || unsafe {
            f.launch(
                [grid_blocks, 1, 1],
                [block_threads, 1, 1],
                0,
                &stream,
                &mut args,
            )
        })
    };

    // Warm L3 with hot buffer
    let _ = launch_ht(8)?;
    let l3_base_ms = launch_ht(16)?;
    let l3_baseline_gbps = (hot_bytes as f64 * 16.0) / (l3_base_ms * 1e-3) / 1e9;

    let mut def_samples = Vec::new();
    for _ in 0..3 {
        let ms = launch_stream(&f_default)?;
        def_samples.push((stream_bytes as f64) / (ms * 1e-3) / 1e9);
    }
    let stream_default_gbps = median_f64(def_samples);
    let l3_after_def_ms = launch_ht(1)?;
    let l3_after_default_gbps = (hot_bytes as f64) / (l3_after_def_ms * 1e-3) / 1e9;

    // Re-warm L3, then stream 1 GiB with TH_LOAD_NT
    let _ = launch_ht(8)?;
    let mut nt_samples = Vec::new();
    for _ in 0..3 {
        let ms = launch_stream(&f_nt)?;
        nt_samples.push((stream_bytes as f64) / (ms * 1e-3) / 1e9);
    }
    let stream_nt_gbps = median_f64(nt_samples);
    let l3_after_nt_ms = launch_ht(1)?;
    let l3_after_nt_gbps = (hot_bytes as f64) / (l3_after_nt_ms * 1e-3) / 1e9;

    drop(dev_stream);
    drop(dev_hot);
    drop(dev_out);

    // 3. Matrix Peak Compute (RDNA 4 WMMA/SWMMAC or CDNA 3 MFMA)
    let compute_blocks = 512u32;
    let compute_threads = 256u32;
    let total_waves = (compute_blocks as f64) * (compute_threads as f64 / ctx.wave_size as f64);
    let wmma_iters = if quick { 1024u32 } else { 4096u32 };
    let dev_wmma_out = DeviceBuf::alloc(256 * 4)?;

    let bench_matrix_f = |kname: &str, k_dim: usize, is_float_seed: bool| -> HipResult<f64> {
        let func = module.function(kname)?;
        let run_once = |it_count: u32| -> HipResult<f64> {
            let mut out_ptr = dev_wmma_out.as_ptr();
            let mut it = it_count;
            let mut fseed = 1.0f32;
            let mut useed = 1u32;
            let mut args: [*mut c_void; 3] = [
                (&mut out_ptr as *mut *mut c_void) as *mut c_void,
                (&mut it as *mut u32) as *mut c_void,
                if is_float_seed {
                    (&mut fseed as *mut f32) as *mut c_void
                } else {
                    (&mut useed as *mut u32) as *mut c_void
                },
            ];
            measure_ms(&stream, || unsafe {
                func.launch(
                    [compute_blocks, 1, 1],
                    [compute_threads, 1, 1],
                    0,
                    &stream,
                    &mut args,
                )
            })
        };
        let _ = run_once(16)?;
        let mut runs = Vec::new();
        for _ in 0..3 {
            let ms = run_once(wmma_iters)?;
            let ops = total_waves * (wmma_iters as f64) * 8.0 * (2.0 * 16.0 * 16.0 * k_dim as f64);
            runs.push(ops / (ms * 1e-3) / 1e12);
        }
        Ok(median_f64(runs))
    };

    let (wmma_f16_tflops, wmma_fp8_tflops, wmma_iu4_tops, swmmac_f16_tflops, swmmac_fp8_tflops) =
        if ctx.arch.is_rdna4() {
            (
                bench_matrix_f(names::WMMA_BENCH_F16, 16, true)?,
                bench_matrix_f(names::WMMA_BENCH_FP8, 16, false)?,
                bench_matrix_f(names::WMMA_BENCH_IU4, 32, false)?,
                bench_matrix_f(names::SWMMAC_BENCH_F16, 32, true)?,
                bench_matrix_f(names::SWMMAC_BENCH_FP8, 32, false)?,
            )
        } else {
            (
                bench_matrix_f(names::MFMA_BENCH_F16, 16, true)?,
                bench_matrix_f(names::MFMA_BENCH_FP8, 32, false)?,
                bench_matrix_f(names::MFMA_BENCH_I8, 32, false)?,
                0.0,
                0.0,
            )
        };
    drop(dev_wmma_out);

    // 4. Register LUT (v_permlane16_var_b32) vs LDS LUT & BITCOS Unpack
    let lut_words: usize = 16 * 1024 * 1024; // 64 MiB packed nibbles (134M nibbles)
    let lut_iters: u32 = if quick { 2 } else { 4 };
    let dev_nib = DeviceBuf::alloc(lut_words * 4)?;
    let dev_signs = DeviceBuf::alloc(lut_words * 4)?;
    let dev_lut_out = DeviceBuf::alloc(out_bytes)?;

    let bench_lut = |kname: &str| -> HipResult<f64> {
        let func = module.function(kname)?;
        let mut nib_ptr = dev_nib.as_ptr();
        let mut out_ptr = dev_lut_out.as_ptr();
        let mut nw = lut_words as u32;
        let mut it = lut_iters;
        let mut args: [*mut c_void; 4] = [
            (&mut nib_ptr as *mut *mut c_void) as *mut c_void,
            (&mut out_ptr as *mut *mut c_void) as *mut c_void,
            (&mut nw as *mut u32) as *mut c_void,
            (&mut it as *mut u32) as *mut c_void,
        ];
        let _ = measure_ms(&stream, || unsafe {
            func.launch(
                [grid_blocks, 1, 1],
                [block_threads, 1, 1],
                0,
                &stream,
                &mut args,
            )
        })?;
        let ms = measure_ms(&stream, || unsafe {
            func.launch(
                [grid_blocks, 1, 1],
                [block_threads, 1, 1],
                0,
                &stream,
                &mut args,
            )
        })?;
        let lookups = (lut_words as f64) * 8.0 * (lut_iters as f64);
        Ok(lookups / (ms * 1e-3) / 1e9)
    };

    let lut_permlane_glups = bench_lut(names::LUT_BENCH_PERMLANE)?;
    let lut_lds_glups = bench_lut(names::LUT_BENCH_LDS)?;

    let bitcos_glups = {
        let func = module.function(names::BITCOS_BENCH)?;
        let mut bm_ptr = dev_nib.as_ptr();
        let mut sg_ptr = dev_signs.as_ptr();
        let mut out_ptr = dev_lut_out.as_ptr();
        let mut nw = lut_words as u32;
        let mut it = lut_iters;
        let mut args: [*mut c_void; 5] = [
            (&mut bm_ptr as *mut *mut c_void) as *mut c_void,
            (&mut sg_ptr as *mut *mut c_void) as *mut c_void,
            (&mut out_ptr as *mut *mut c_void) as *mut c_void,
            (&mut nw as *mut u32) as *mut c_void,
            (&mut it as *mut u32) as *mut c_void,
        ];
        let _ = measure_ms(&stream, || unsafe {
            func.launch(
                [grid_blocks, 1, 1],
                [block_threads, 1, 1],
                0,
                &stream,
                &mut args,
            )
        })?;
        let ms = measure_ms(&stream, || unsafe {
            func.launch(
                [grid_blocks, 1, 1],
                [block_threads, 1, 1],
                0,
                &stream,
                &mut args,
            )
        })?;
        let weights = (lut_words as f64) * 32.0 * (lut_iters as f64);
        weights / (ms * 1e-3) / 1e9
    };

    drop(dev_nib);
    drop(dev_signs);
    drop(dev_lut_out);

    // 5. PCIe Pinned DMA & NVMe io_uring O_DIRECT
    let dma_bytes = 512 * 1024 * 1024;
    let mut pinned = PinnedBuf::alloc(dma_bytes)?;
    let mut dev_dma = DeviceBuf::alloc(dma_bytes)?;
    // warmup
    dev_dma.copy_from_host_async(pinned.as_ptr(), dma_bytes, &stream)?;
    stream.synchronize()?;

    let h2d_ms = measure_ms(&stream, || {
        dev_dma.copy_from_host_async(pinned.as_ptr(), dma_bytes, &stream)
    })?;
    let d2h_ms = measure_ms(&stream, || {
        dev_dma.copy_to_host_async(pinned.as_mut_ptr(), dma_bytes, &stream)
    })?;
    let pcie_h2d_gbps = (dma_bytes as f64) / (h2d_ms * 1e-3) / 1e9;
    let pcie_d2h_gbps = (dma_bytes as f64) / (d2h_ms * 1e-3) / 1e9;
    drop(dev_dma);
    drop(pinned);

    let nvme_bytes = if quick {
        64 * 1024 * 1024
    } else {
        256 * 1024 * 1024
    };
    let chunk_bytes = 256 * 1024;
    let p31_path = PathBuf::from("/home/noorm/projects/scoria/.gate_tmp_p31.bin");
    let _p31_guard = TempFileGuard(p31_path.clone());
    write_incompressible_file(&p31_path, nvme_bytes)?;
    let p31_gbps = uring::bench_seq_read_odirect(&p31_path, nvme_bytes, chunk_bytes, 32)?;

    let nm_dir = Path::new("/mnt/nuclearmissilesilo");
    let mut s980_gbps: Option<f64> = None;
    let mut dual_nvme_gbps: Option<f64> = None;
    if nm_dir.is_dir() {
        let s980_path = nm_dir.join(".gate_tmp_980.bin");
        let _s980_guard = TempFileGuard(s980_path.clone());
        if write_incompressible_file(&s980_path, nvme_bytes).is_ok()
            && let Ok(gbps) = uring::bench_seq_read_odirect(&s980_path, nvme_bytes, chunk_bytes, 32)
        {
            s980_gbps = Some(gbps);
            let t0 = Instant::now();
            let (r1, r2) = std::thread::scope(|s| {
                let h1 = s.spawn(|| {
                    uring::bench_seq_read_odirect(&p31_path, nvme_bytes, chunk_bytes, 32)
                });
                let h2 = s.spawn(|| {
                    uring::bench_seq_read_odirect(&s980_path, nvme_bytes, chunk_bytes, 32)
                });
                (h1.join().unwrap(), h2.join().unwrap())
            });
            let wall = t0.elapsed().as_secs_f64();
            if r1.is_ok() && r2.is_ok() {
                dual_nvme_gbps = Some((2.0 * nvme_bytes as f64) / wall / 1e9);
            }
        }
    }

    let s980_str = s980_gbps
        .map(|v| format!("{v:.3}"))
        .unwrap_or_else(|| "null".to_string());
    let dual_str = dual_nvme_gbps
        .map(|v| format!("{v:.3}"))
        .unwrap_or_else(|| "null".to_string());

    let json = format!(
        r#"{{
  "arch": "gfx1201",
  "quick": {quick},
  "vram_total_bytes": {total_b},
  "vram_free_at_start_bytes": {free_b},
  "soak": {{
    "cycles": {soak_cycles},
    "passes": {soak_passes},
    "sizes_gib": {sizes_gib:?},
    "min_cycle_ms": {soak_min_ms:.2},
    "max_cycle_ms": {soak_max_ms:.2},
    "total_ms": {soak_total_ms:.2}
  }},
  "vram_and_l3": {{
    "stream_default_gbps": {stream_default_gbps:.2},
    "stream_th_nt_gbps": {stream_nt_gbps:.2},
    "l3_baseline_gbps": {l3_baseline_gbps:.2},
    "l3_after_default_gbps": {l3_after_default_gbps:.2},
    "l3_after_th_nt_gbps": {l3_after_nt_gbps:.2}
  }},
  "compute": {{
    "wmma_f16_tflops": {wmma_f16_tflops:.2},
    "wmma_fp8_tflops": {wmma_fp8_tflops:.2},
    "wmma_iu4_tops": {wmma_iu4_tops:.2},
    "swmmac_f16_tflops": {swmmac_f16_tflops:.2},
    "swmmac_fp8_tflops": {swmmac_fp8_tflops:.2},
    "swmmac_f16_ratio": {f16_ratio:.3},
    "swmmac_fp8_ratio": {fp8_ratio:.3}
  }},
  "lut_and_ternary": {{
    "permlane16_var_giga_lookups_s": {lut_permlane_glups:.2},
    "lds_giga_lookups_s": {lut_lds_glups:.2},
    "bitcos_giga_weights_s": {bitcos_glups:.2}
  }},
  "pcie_and_nvme": {{
    "pcie_h2d_gbps": {pcie_h2d_gbps:.2},
    "pcie_d2h_gbps": {pcie_d2h_gbps:.2},
    "nvme_p31_odirect_gbps": {p31_gbps:.3},
    "nvme_980_odirect_gbps": {s980_str},
    "nvme_dual_odirect_gbps": {dual_str}
  }}
}}
"#,
        f16_ratio = swmmac_f16_tflops / wmma_f16_tflops,
        fp8_ratio = swmmac_fp8_tflops / wmma_fp8_tflops,
    );

    let baseline_path = Path::new("harness/baselines/gfx1201_gate.json");
    if let Some(parent) = baseline_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(baseline_path, &json)?;

    println!("\n=== scoria Phase 0 gfx1201 Hardware Gate ===");
    println!(
        "Soak (ROCm 7.16):     {soak_passes}/{soak_cycles} cycles passed (sizes={sizes_gib:?} GiB, {soak_total_ms:.1} ms)"
    );
    println!(
        "VRAM Stream Default:  {stream_default_gbps:.1} GB/s | L3 probe after: {l3_after_default_gbps:.1} GB/s (baseline {l3_baseline_gbps:.1} GB/s)"
    );
    println!(
        "VRAM Stream TH_NT:    {stream_nt_gbps:.1} GB/s | L3 probe after: {l3_after_nt_gbps:.1} GB/s"
    );
    println!(
        "WMMA FP16 / FP8 / IU4:{wmma_f16_tflops:.1} TFLOPS / {wmma_fp8_tflops:.1} TFLOPS / {wmma_iu4_tops:.1} TOPS"
    );
    println!(
        "SWMMAC FP16 / FP8:    {swmmac_f16_tflops:.1} TFLOPS ({:.2}x) / {swmmac_fp8_tflops:.1} TFLOPS ({:.2}x)",
        swmmac_f16_tflops / wmma_f16_tflops,
        swmmac_fp8_tflops / wmma_fp8_tflops
    );
    println!(
        "LUT permlane vs LDS:  {lut_permlane_glups:.1} vs {lut_lds_glups:.1} G-lookups/s | BITCOS: {bitcos_glups:.1} G-weights/s"
    );
    println!("PCIe Pinned H2D/D2H:  {pcie_h2d_gbps:.2} / {pcie_d2h_gbps:.2} GB/s");
    println!(
        "NVMe io_uring O_DIRECT: P31={p31_gbps:.3} GB/s | 980={} GB/s | Dual={} GB/s",
        s980_gbps.map_or("n/a".into(), |v| format!("{v:.3}")),
        dual_nvme_gbps.map_or("n/a".into(), |v| format!("{v:.3}"))
    );
    println!("Saved: {}", baseline_path.display());
    Ok(())
}
