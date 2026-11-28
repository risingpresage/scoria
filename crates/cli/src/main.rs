mod gate;
mod uring;

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process;

use scoria_engine::EngineContext;
use scoria_engine::qwen2::{LayerTap, Qwen2Model, top_k_logits};
use scoria_gguf::{BpeTokenizer, GgufFile, ModelConfig};

fn usage() -> ! {
    eprintln!(
        "Usage:\n  scoria gate [--quick]\n  scoria inspect <model.gguf>\n  scoria tokenize <model.gguf> <prompt> [--no-bos]\n  scoria run <model.gguf> (--prompt <text> | --tokens <csv>) [--max-tokens <N>] [--max-seq <N>] [--no-bos] [--ignore-eos] [--warmup <N>] [--kv-f16] [--dump-logits <path>] [--tap-layer <L>] [--profile]\n  scoria graph-soak <model.gguf> [--replays <N>] [--tokens <csv>]\n  scoria plan <model.gguf> [--vram-mib <N>] [--ctx <N>]\n  scoria strata <model.gguf> [--top-k <N>] [--tokens <N>] [--qd <N>]\n  scoria arch-check <model.gguf> [--dequant-smoke]\n  scoria layer0 <model.gguf> [--tokens <csv>] [--moe] [--router-softmax]\n  scoria logits <model.gguf> [--length <N>] [--tokens <csv>] [--top <N>] [--prompt <text>] [--perp <text>] [--generate <N>] [--threads <N>] [--no-gpu] [--max-visits <N>]"
    );
    process::exit(2);
}

fn cmd_inspect(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let gguf = GgufFile::open(path)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;
    let tok = BpeTokenizer::from_gguf(&gguf)?;
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for t in &gguf.tensors {
        *counts.entry(format!("{:?}", t.ggml_type)).or_insert(0) += 1;
    }
    println!("file: {path}");
    println!(
        "gguf: v{}, tensors={}, kv={}, alignment={}, data_offset={}",
        gguf.header.version,
        gguf.header.tensor_count,
        gguf.header.metadata_kv_count,
        gguf.alignment,
        gguf.data_offset
    );
    println!("config: {cfg:?}");
    println!(
        "tokenizer: vocab={}, bos={:?}, eos={:?}, add_bos={}",
        tok.tokens.len(),
        tok.bos_token_id,
        tok.eos_token_id,
        tok.add_bos_token
    );
    println!("tensor_types: {counts:?}");
    Ok(())
}

fn cmd_tokenize(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.len() < 2 {
        usage();
    }
    let path = &args[0];
    let prompt = &args[1];
    let no_bos = args.iter().any(|a| a == "--no-bos");
    let gguf = GgufFile::open(path)?;
    let tok = BpeTokenizer::from_gguf(&gguf)?;
    let ids = tok.encode(prompt, !no_bos);
    println!("{ids:?}");
    Ok(())
}

fn arg_val<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

fn cmd_run(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        usage();
    }
    let path = &args[0];
    let rest = &args[1..];
    let prompt_arg = arg_val(rest, "--prompt");
    let tokens_arg = arg_val(rest, "--tokens");
    if prompt_arg.is_none() && tokens_arg.is_none() {
        usage();
    }

    let max_tokens: usize = arg_val(rest, "--max-tokens")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(32);
    let max_seq: usize = arg_val(rest, "--max-seq")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(2048);
    let no_bos = rest.iter().any(|a| a == "--no-bos");
    let ignore_eos = rest.iter().any(|a| a == "--ignore-eos");
    let profile = rest.iter().any(|a| a == "--profile");
    let kv_f16 = rest.iter().any(|a| a == "--kv-f16");
    let warmup: usize = arg_val(rest, "--warmup")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(0);
    let dump_logits = arg_val(rest, "--dump-logits").map(PathBuf::from);
    let tap_layer: Option<usize> = arg_val(rest, "--tap-layer")
        .map(|s| s.parse())
        .transpose()?;

    let gguf = GgufFile::open(path)?;
    let tok = BpeTokenizer::from_gguf(&gguf)?;
    let prompt_ids: Vec<u32> = if let Some(csv) = tokens_arg {
        csv.split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?
    } else {
        tok.encode(prompt_arg.unwrap(), !no_bos)
    };

    let ctx = EngineContext::init(0)?;
    let mut model = Qwen2Model::load_with(
        &ctx,
        &gguf,
        max_seq,
        scoria_engine::qwen2::ModelOpts { kv_f16 },
    )?;

    let mut tap_cb = |name: &str, vals: &[f32]| {
        if let Some(target_l) = tap_layer {
            let suffix = format!("-{target_l}");
            if name.ends_with(&suffix) || name == "result_norm" {
                let l2: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
                let preview: Vec<f32> = vals.iter().take(6).copied().collect();
                eprintln!(
                    "[tap] {name}: len={}, l2={l2:.6}, first6={preview:?}",
                    vals.len()
                );
            }
        }
    };

    // Untimed warm-up passes: the first forward in a fresh process pays lazy HSACO
    // binding and clock ramp, which otherwise lands entirely in the prefill timing.
    for _ in 0..warmup {
        model.forward_tokens(&ctx, &prompt_ids, None)?;
        model.reset();
    }

    let tap_opt: Option<LayerTap<'_>> = if tap_layer.is_some() {
        Some(&mut tap_cb)
    } else {
        None
    };

    let res = model.generate_greedy(
        &ctx,
        &prompt_ids,
        max_tokens,
        tok.eos_token_id,
        ignore_eos,
        tap_opt,
    )?;

    if let Some(dump_path) = dump_logits {
        let mut f = fs::File::create(&dump_path)?;
        let raw = unsafe {
            std::slice::from_raw_parts(
                res.prefill_logits.as_ptr() as *const u8,
                res.prefill_logits.len() * std::mem::size_of::<f32>(),
            )
        };
        f.write_all(raw)?;
    }

    let top5 = top_k_logits(&res.prefill_logits, 5);
    let decoded = tok.decode(&res.generated);
    let prefill_tps = (prompt_ids.len() as f64) / (res.prefill_ms * 1e-3);
    let gen_steps_after_prefill = res.generated.len().saturating_sub(1);
    let decode_tps = if res.decode_ms > 0.0 && gen_steps_after_prefill > 0 {
        (gen_steps_after_prefill as f64) / (res.decode_ms * 1e-3)
    } else {
        0.0
    };

    println!("prompt_tokens: {prompt_ids:?}");
    println!("generated_tokens: {:?}", res.generated);
    println!("top5_prefill_logits: {top5:?}");
    println!(
        "timing: prefill_ms={:.2} ({prefill_tps:.2} tok/s), decode_ms={:.2} ({decode_tps:.2} tok/s)",
        res.prefill_ms, res.decode_ms
    );
    println!("decoded: {decoded}");
    if profile {
        let tok0 = res.generated.first().copied().unwrap_or(1);
        let prof = model.profile_decode_step(&ctx, tok0)?;
        let gpu_sum = prof.gemv_other_ms
            + prof.q6k_ms
            + prof.lm_head_ms
            + prof.rmsnorm_ms
            + prof.attn_ms
            + prof.other_ms;
        println!(
            "profile_1step_ms: gpu_sum={gpu_sum:.3} gemv_q4k/other={:.3} q6k_layers={:.3} lm_head={:.3} rmsnorm={:.3} attn={:.3} other={:.3}",
            prof.gemv_other_ms,
            prof.q6k_ms,
            prof.lm_head_ms,
            prof.rmsnorm_ms,
            prof.attn_ms,
            prof.other_ms
        );
    }
    Ok(())
}

/// Replays the captured decode graph many times and asserts VRAM does not drift.
/// ROCm graph instantiate/launch paths have historically leaked; this is the guard
/// the Phase 2B plan requires before `hipGraph` decode can be trusted.
fn cmd_graph_soak(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        usage();
    }
    let path = &args[0];
    let rest = &args[1..];
    let replays: usize = arg_val(rest, "--replays")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1000);
    let tokens: Vec<u32> = arg_val(rest, "--tokens")
        .unwrap_or("785,6722,315,9625,374")
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;

    let gguf = GgufFile::open(path)?;
    let ctx = EngineContext::init(0)?;
    let mut model = Qwen2Model::load(&ctx, &gguf, 4096)?;

    // Prefill so the KV cache is populated and the decode step is representative.
    let logits = model.forward_tokens(&ctx, &tokens, None)?;
    let first = scoria_engine::qwen2::argmax_f32(&logits);
    model
        .seed_decode_token(&ctx, first)
        .expect("seed decode token");
    let (free0, total) = scoria_hip_rt::mem_info()?;
    let graph = model.capture_decode_graph(&ctx)?;

    let mut tokens_out = std::collections::BTreeSet::new();
    for _ in 0..replays {
        match model.decode_graph_next(&graph, 4096)? {
            Some(tok) => {
                tokens_out.insert(tok);
            }
            None => break,
        }
    }
    drop(graph);
    let (free1, _) = scoria_hip_rt::mem_info()?;

    let drift = free0 as i128 - free1 as i128;
    println!(
        "graph_soak: replays={replays} free_before={:.2} MiB free_after={:.2} MiB drift={:.3} MiB total={:.2} MiB distinct_tokens={}",
        free0 as f64 / (1024.0 * 1024.0),
        free1 as f64 / (1024.0 * 1024.0),
        drift as f64 / (1024.0 * 1024.0),
        total as f64 / (1024.0 * 1024.0),
        tokens_out.len()
    );
    if drift > 64 * 1024 * 1024 {
        return Err(format!("decode graph leaked {drift} bytes over {replays} replays").into());
    }
    Ok(())
}

fn cmd_plan(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        usage();
    }
    let path = &args[0];
    let rest = &args[1..];
    let vram_override: Option<u64> = arg_val(rest, "--vram-mib")
        .map(|s| s.parse::<u64>())
        .transpose()?;
    let target_ctx: Option<usize> = arg_val(rest, "--ctx").map(|s| s.parse()).transpose()?;

    let gguf = GgufFile::open(path)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;
    let budget = cfg.kv_budget();
    let weights = fs::metadata(path)?.len();

    println!("model:       {path}");
    println!(
        "arch:        {}, {} layers, {} KV slots{}",
        cfg.architecture,
        cfg.n_layers,
        budget.slots,
        match (cfg.loop_count, cfg.loop_start, cfg.loop_end) {
            (Some(c), Some(s), Some(e)) if c > 1 => format!(" ({c}x loop on {s}..{e})"),
            _ => String::new(),
        }
    );
    println!("context:     {} trained", cfg.context_length);
    if let (Some(total), Some(used)) = (cfg.n_experts, cfg.n_experts_used) {
        println!("experts:     {total}, top-{used}");
    }
    let windowed_layers = cfg
        .sliding_window_pattern
        .as_ref()
        .map(|p| p.iter().filter(|&&v| v != 0).count())
        .unwrap_or(0);
    println!(
        "sliding:     window {} on {} of {} layers",
        budget.window, windowed_layers, cfg.n_layers
    );
    println!(
        "kv slots:    {} full-history + {} windowed",
        budget.full_slots, budget.windowed_slots
    );
    println!(
        "per token:   {} B FP32 / {} B FP16 / {} B FP8  (full-history slots only)",
        budget.full_bytes_per_token * 4,
        budget.full_bytes_per_token * 2,
        budget.full_bytes_per_token
    );

    println!(
        "\n{:>8}  {:>12}  {:>12}  {:>12}",
        "ctx", "FP32", "FP16", "FP8"
    );
    let mut ctxs = vec![32768usize, 131072, 200000, 262144];
    if let Some(c) = target_ctx {
        ctxs.push(c);
    }
    ctxs.sort_unstable();
    ctxs.dedup();
    for ctx in ctxs {
        if ctx > cfg.context_length {
            continue;
        }
        println!(
            "{:>8}  {:>9.2} GiB  {:>9.2} GiB  {:>9.2} GiB",
            ctx,
            budget.total_bytes(ctx, 4) as f64 / (1u64 << 30) as f64,
            budget.total_bytes(ctx, 2) as f64 / (1u64 << 30) as f64,
            budget.total_bytes(ctx, 1) as f64 / (1u64 << 30) as f64
        );
    }

    let (free, total) = scoria_hip_rt::mem_info()?;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    println!("\nweights:     {:.2} GiB", gib(weights));
    println!(
        "vram:        {:.2} GiB free of {:.2} GiB",
        gib(free as u64),
        gib(total as u64)
    );
    let room = match vram_override {
        Some(mib) => mib * 1024 * 1024,
        None => (free as u64).saturating_sub(weights),
    };
    if (free as u64) < weights {
        println!(
            "deficit:     weights need {:.2} GiB more than is free; they must stream from NVMe (Strata)",
            gib(weights - free as u64)
        );
    }
    if room == 0 {
        println!("kv room:     none left after weights");
        return Ok(());
    }
    println!("kv room:     {:.2} GiB", gib(room));
    for (name, bytes) in [("FP32", 4u64), ("FP16", 2), ("FP8", 1)] {
        let max = budget.max_ctx(room, bytes).min(cfg.context_length);
        println!("max ctx {name}: {max:>9} tokens");
    }
    Ok(())
}

/// Strata: measure the paging pattern a MoE model needs when its experts do not fit.
///
/// Per token, only `top_k` of `n_experts` expert slabs are touched, so a model can
/// live on NVMe and stream the routed experts. This reads exactly that pattern
/// (random expert slabs, io_uring O_DIRECT, queue depth 16) and reports the
/// decode ceiling it implies.
fn cmd_strata(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Read;
    use std::os::unix::fs::FileExt;

    unsafe extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    }
    const POSIX_FADV_DONTNEED: i32 = 4;

    if args.is_empty() {
        usage();
    }
    let path = &args[0];
    let rest = &args[1..];
    let top_k: usize = arg_val(rest, "--top-k")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(8);
    let tokens: usize = arg_val(rest, "--tokens")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(16);
    let qd: usize = arg_val(rest, "--qd")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(16);
    let force_buffered = rest.iter().any(|a| a == "--buffered");

    let gguf = GgufFile::open(path)?;
    let cfg = scoria_gguf::ModelConfig::from_gguf(&gguf)?;
    let n_experts = cfg
        .n_experts
        .ok_or("model has no expert tensors; strata applies to MoE checkpoints only")?;
    let file_len = fs::metadata(path)?.len() as usize;

    let mut spans: Vec<u64> = gguf.tensors.iter().map(|t| t.offset).collect();
    spans.sort_unstable();
    let tensor_len = |offset: u64| -> usize {
        let idx = spans.binary_search(&offset).unwrap();
        let end = spans
            .get(idx + 1)
            .copied()
            .unwrap_or((file_len - gguf.data_offset) as u64);
        (end - offset) as usize
    };

    // [ne0, ne1, n_experts] means expert `e` is one contiguous slab.
    let mut slabs: Vec<(u64, usize)> = Vec::new();
    for l in 0..cfg.n_layers {
        for suffix in ["ffn_gate_up_exps.weight", "ffn_down_exps.weight"] {
            let Some(t) = gguf.tensor(&format!("blk.{l}.{suffix}")) else {
                continue;
            };
            let slab = tensor_len(t.offset) / n_experts;
            slabs.push((gguf.data_offset as u64 + t.offset, slab));
        }
    }
    if slabs.is_empty() {
        return Err(format!("no expert tensors found in {path}").into());
    }
    let expert_bytes: usize = slabs.iter().map(|&(_, s)| s).sum::<usize>() * n_experts;
    let per_token_bytes: usize = slabs.iter().map(|&(_, s)| s).sum::<usize>() * top_k;
    let max_slab = slabs.iter().map(|&(_, s)| s).max().unwrap();

    println!(
        "model:        {} ({}, {} layers, {} experts top-{})",
        path, cfg.architecture, cfg.n_layers, n_experts, top_k
    );
    println!(
        "expert set:   {:.2} GiB in {} tensors, slab {:.2}-{:.2} MiB",
        expert_bytes as f64 / (1u64 << 30) as f64,
        slabs.len(),
        slabs.iter().map(|&(_, s)| s).min().unwrap() as f64 / (1024.0 * 1024.0),
        max_slab as f64 / (1024.0 * 1024.0)
    );
    println!(
        "per token:    {} slabs = {:.1} MiB routed ({:.1}% of the expert set)",
        slabs.len() * top_k,
        per_token_bytes as f64 / (1024.0 * 1024.0),
        100.0 * per_token_bytes as f64 / expert_bytes as f64
    );

    let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut requests: Vec<(u64, usize)> = Vec::new();
    for _ in 0..tokens {
        for &(base, slab) in &slabs {
            for _ in 0..top_k {
                let e = next() % n_experts as u64;
                let off = base + e * slab as u64;
                let aligned = off & !4095u64;
                let skip = (off - aligned) as usize;
                let limit = ((file_len as u64).saturating_sub(aligned)) & !4095u64;
                let len = ((slab + skip).next_multiple_of(4096) as u64).min(limit) as usize;
                if len > 0 {
                    requests.push((aligned, len.min(max_slab + 4096)));
                }
            }
        }
    }

    let mut mode = "io_uring O_DIRECT";
    let mut bytes = 0u64;
    let mut secs = 0.0f64;
    let direct = if force_buffered {
        None
    } else {
        crate::uring::DirectFile::open_readonly(std::path::Path::new(path)).ok()
    };

    match direct {
        Some(file) => {
            let ring = crate::uring::IoUring::new((qd * 2) as u32)?;
            let read_len = max_slab.next_multiple_of(4096) + 4096;
            let mut bufs: Vec<crate::uring::AlignedBuf> = Vec::with_capacity(qd);
            for _ in 0..qd {
                bufs.push(crate::uring::AlignedBuf::alloc(4096, read_len)?);
            }
            let t0 = std::time::Instant::now();
            let (mut inflight, mut next_req, mut done) = (0usize, 0usize, 0usize);
            let mut failed = false;
            while done < requests.len() {
                let mut pushed = 0u32;
                while inflight < qd && next_req < requests.len() {
                    let slot = next_req % qd;
                    let (off, len) = requests[next_req];
                    if !ring.push_read(
                        file.fd(),
                        off,
                        bufs[slot].as_mut_ptr(),
                        len as u32,
                        slot as u64,
                    ) {
                        break;
                    }
                    next_req += 1;
                    inflight += 1;
                    pushed += 1;
                }
                if pushed == 0 && inflight == 0 {
                    break;
                }
                ring.submit_and_wait(pushed, 1)?;
                while let Some(cqe) = ring.pop_cqe() {
                    if cqe.res < 0 {
                        failed = true;
                        break;
                    }
                    bytes += cqe.res as u64;
                    done += 1;
                    inflight -= 1;
                }
                if failed {
                    break;
                }
            }
            secs = t0.elapsed().as_secs_f64();
            if failed {
                println!(
                    "direct io:    rejected (EIO) - the model store is on a btrfs subvolume with \
compress=zstd, which does not support O_DIRECT; falling back to buffered reads"
                );
                bytes = 0;
            }
        }
        None => {
            println!("direct io:    unavailable for this path; using buffered reads");
        }
    }

    if bytes == 0 {
        mode = "buffered pread + FADV_DONTNEED";
        let file = fs::File::open(path)?;
        let mut buf = vec![0u8; max_slab + 4096];
        let t0 = std::time::Instant::now();
        for &(off, len) in &requests {
            file.read_exact_at(&mut buf[..len], off)?;
            bytes += len as u64;
            unsafe {
                posix_fadvise(
                    std::os::unix::io::AsRawFd::as_raw_fd(&file),
                    off as i64,
                    len as i64,
                    POSIX_FADV_DONTNEED,
                );
            }
        }
        secs = t0.elapsed().as_secs_f64();
    }
    let _ = std::io::stdin().read(&mut [0u8; 0]);

    let ms_per_token = secs * 1000.0 / tokens as f64;
    println!(
        "mode:         {mode}, qd {qd}, {tokens} tokens, {:.1} MiB read",
        bytes as f64 / (1024.0 * 1024.0)
    );
    println!("throughput:   {:.2} GB/s", bytes as f64 / secs / 1e9);
    println!(
        "paged decode: {:.1} ms/token = {:.0} tok/s ceiling with zero compute",
        ms_per_token,
        1000.0 / ms_per_token
    );
    println!(
        "resident alternative: {:.2} GiB of experts in VRAM removes all of it, which is why \
a hot-expert LRU beats raw disk speed",
        expert_bytes as f64 / (1u64 << 30) as f64
    );
    Ok(())
}

/// Validate a checkpoint against the `inkling2_loop` architecture contract: which
/// tensors exist for every role, which quantisation types the engine cannot dequantise
/// yet, and the 48-visit execution order with its KV slots.
/// Dequantises the first blocks of one real tensor per distinct GGML type in the file and
/// reports the value range. This is the end-to-end check the synthetic block tests cannot
/// give: mmap -> block slicing -> dequant on weights that actually ship.
fn dequant_smoke(gguf: &GgufFile) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::BTreeMap;
    let mut first_by_type: BTreeMap<u32, &str> = BTreeMap::new();
    for t in &gguf.tensors {
        first_by_type
            .entry(t.ggml_type.as_u32())
            .or_insert(t.name.as_str());
    }
    println!("\ndequant smoke (8 blocks per type, from the checkpoint):");
    for (ty, name) in first_by_type {
        let info = gguf.tensor(name).expect("tensor present");
        let Some(block_elems) = info.ggml_type.block_elements() else {
            println!("  {name:38} type {ty:<3} unsupported");
            continue;
        };
        let Some(block_bytes) = info.ggml_type.block_bytes() else {
            println!("  {name:38} type {ty:<3} unsupported");
            continue;
        };
        let bytes = gguf.tensor_bytes(info)?;
        let blocks = (bytes.len() / block_bytes).min(8);
        let mut out = vec![0f32; blocks * block_elems];
        scoria_gguf::dequant::dequant_row(
            info.ggml_type,
            &bytes[..blocks * block_bytes],
            &mut out,
        )?;
        let finite = out.iter().filter(|v| v.is_finite()).count();
        let lo = out.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // Expert tensors are stored pre-divided by a per-expert f32 outer scale; without
        // it the raw weights are four orders of magnitude too large, so report both.
        let scale_name = name
            .strip_suffix(".weight")
            .map_or_else(|| format!("{name}.scale"), |stem| format!("{stem}.scale"));
        let outer = gguf
            .tensor(&scale_name)
            .and_then(|si| gguf.tensor_bytes(si).ok())
            .filter(|b| b.len() >= 4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        let scaled = match outer {
            Some(k) => format!(" outer={k:.3e} -> [{:.4}, {:.4}]", lo * k, hi * k),
            None => String::new(),
        };
        println!(
            "  {name:38} type {ty:<3} {:<7} n={:<6} finite={}/{} raw=[{lo:.4}, {hi:.4}]{scaled}",
            format!("{:?}", info.ggml_type),
            out.len(),
            finite,
            out.len()
        );
    }
    Ok(())
}

fn cmd_arch_check(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use scoria_engine::inkling2;
    if args.is_empty() {
        usage();
    }
    let path = &args[0];
    let gguf = GgufFile::open(path)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;
    if cfg.architecture != inkling2::ARCH {
        return Err(format!(
            "{} is `{}`, not `{}`",
            path,
            cfg.architecture,
            inkling2::ARCH
        )
        .into());
    }
    let params = inkling2::Inkling2Params::from_gguf(&gguf, &cfg);
    let (checks, unknown) = inkling2::check(&gguf, &cfg);

    println!("model:       {path}");
    println!(
        "params:      rel {}x{} clip {:.3} | qk_clip {:.3} | log_scale {:.2}/{} | situ {:.1}/{:.1}",
        params.relative_length,
        params.relative_extent,
        params.qk_clip,
        params.qk_clip,
        params.log_scale_alpha,
        params.log_scale_floor,
        params.situ_beta,
        params.situ_linear_beta
    );
    println!(
        "             attn_res block {} | softcap {:.1} | conv cache {} | expert ff {} scale {:.1}",
        params.attn_res_block,
        params.final_logit_softcap,
        params.shortconv_cache,
        params.expert_ff,
        params.expert_weights_scale
    );
    println!();

    let w = inkling2::ROLES
        .iter()
        .map(|r| r.suffix.len())
        .max()
        .unwrap_or(0);
    println!(
        "{:<w$}  {:>6}  {:>7}  {:<28} status",
        "role",
        "layers",
        "missing",
        "types",
        w = w
    );
    for c in &checks {
        let status = if c.missing > 0 {
            "MISSING".to_string()
        } else if !c.unsupported.is_empty() {
            format!("needs dequant: {}", c.unsupported.join(","))
        } else {
            "ok".to_string()
        };
        println!(
            "{:<w$}  {:>6}  {:>7}  {:<28} {}",
            c.suffix,
            c.present,
            c.missing,
            c.types.join(","),
            status,
            w = w
        );
    }

    let calls = inkling2::schedule(&cfg);
    let budget = cfg.kv_budget();
    println!();
    println!(
        "schedule:    {} visits = {} direct + {} looped + {} tail; {} KV slots",
        calls.len(),
        cfg.loop_start.unwrap_or(0),
        (cfg.loop_end.unwrap_or(0) - cfg.loop_start.unwrap_or(0)) * cfg.loop_count.unwrap_or(1),
        cfg.n_layers - cfg.loop_end.unwrap_or(cfg.n_layers),
        budget.slots
    );
    println!(
        "visit order: 0..{} layer 0..{}, then layer {}..{} x{}, then layer {}..{}",
        cfg.loop_start.unwrap_or(0),
        cfg.loop_start.unwrap_or(0),
        cfg.loop_start.unwrap_or(0),
        cfg.loop_end.unwrap_or(0),
        cfg.loop_count.unwrap_or(1),
        cfg.loop_end.unwrap_or(0),
        cfg.n_layers
    );
    if !unknown.is_empty() {
        println!("unrecognised tensors ({}):", unknown.len());
        for name in &unknown {
            println!("  {name}");
        }
    }
    let unsupported: Vec<String> = checks
        .iter()
        .filter(|c| !c.unsupported.is_empty())
        .map(|c| format!("{}:{}", c.suffix, c.unsupported.join(",")))
        .collect();
    if !unsupported.is_empty() {
        println!("blocking dequant work: {}", unsupported.join(" "));
    }
    let missing: Vec<&str> = checks
        .iter()
        .filter(|c| c.missing > 0)
        .map(|c| c.suffix)
        .collect();
    if !missing.is_empty() {
        println!("missing required roles: {}", missing.join(" "));
    }
    if args.iter().any(|a| a == "--dequant-smoke") {
        dequant_smoke(&gguf)?;
    }
    Ok(())
}

/// Run layer 0 of an `inkling2_loop` checkpoint end to end on the host: embedding ->
/// attention norm -> projections -> conv/clip/RMSNorm/gate -> relative bias -> attention ->
/// output gate -> W_o -> post conv -> residual. Prints diagnostics, because the FFN and the
/// rest of the stack are not wired yet.
/// One token position through all 48 visits of the full stack, then the output head.
fn cmd_logits(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use scoria_engine::inkling2_model::Model;
    if args.is_empty() {
        usage();
    }
    let path = &args[0];
    let rest = &args[1..];
    let tokens: Vec<u32> = arg_val(rest, "--tokens")
        .unwrap_or("785,6722,315,9625,374")
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;
    let top = arg_val(rest, "--top")
        .unwrap_or("5")
        .parse::<usize>()
        .unwrap_or(5);
    let generate = arg_val(rest, "--generate")
        .map(|v| v.parse::<usize>().unwrap_or(0))
        .unwrap_or(0);
    let prompt = arg_val(rest, "--prompt");
    let perp = arg_val(rest, "--perp");
    let threads = arg_val(rest, "--threads")
        .map(|v| v.parse::<usize>().unwrap_or(0))
        .unwrap_or(0);
    let max_visits = arg_val(rest, "--max-visits")
        .map(|v| v.parse::<usize>().unwrap_or(usize::MAX))
        .unwrap_or(usize::MAX);
    // A deterministic synthetic stream, for long-context runs: same ids every run, so two
    // measurements at the same length are comparable.
    let length = arg_val(rest, "--length").map(|v| v.parse::<usize>().unwrap_or(0));

    let (tok, prompt_tokens) = if let Some(text) = perp.or(prompt) {
        let g = GgufFile::open(path)?;
        let t = scoria_gguf::tokenizer::BpeTokenizer::from_gguf(&g)?;
        let ids = t.encode(text, false);
        (Some(t), ids)
    } else {
        (None, tokens.clone())
    };
    // Routing locality and the dequant/arithmetic split: the two numbers that decide
    // whether caching dequantised experts is worth building.
    let route_stats = arg_val(rest, "--route-stats")
        .map(|v| v.parse::<usize>().unwrap_or(0))
        .unwrap_or(0);
    let bench = rest.iter().any(|a| a == "--bench-dequant");

    let tokens = match length {
        Some(n) if n > 0 => (0..n).map(|i| 300 + ((i * 37) % 4000) as u32).collect(),
        _ => prompt_tokens,
    };
    if tokens.is_empty() {
        return Err("no tokens (empty prompt)".into());
    }

    let started = std::time::Instant::now();
    let router = if rest.iter().any(|a| a == "--router-softmax") {
        scoria_engine::inkling2_moe::RouterScore::Softmax
    } else {
        scoria_engine::inkling2_moe::RouterScore::Sigmoid
    };
    let mut model = Model::open_with_router(path, router)?;
    model.max_visits = max_visits;
    if threads > 0 {
        model.set_threads(threads);
    }
    let prefill_chunk = arg_val(rest, "--prefill-chunk")
        .map(|v| v.parse::<usize>().unwrap_or(0))
        .unwrap_or(0);
    if prefill_chunk > 0 {
        let mut model = Model::open_with_router(path, router)?;
        model.max_visits = max_visits;
        // Measured on this machine: expert residency is 2.5x faster for decode and 20-50x
        // slower for the chunked prefill (the host-side expert reads for the layers without a
        // device codec appear to lose the page cache to the VRAM allocation). Prefill keeps
        // its experts on the host.
        // Measured: residency is 2.5x faster for decode and 20-50x slower for the chunked
        // prefill. `--hot-experts` keeps residency on and pre-faults the bytes the card does
        // not hold, to test whether page-cache eviction was the whole problem.
        let hot = rest.iter().any(|a| a == "--hot-experts");
        if !hot {
            model.set_expert_residency(false);
        }
        if threads > 0 {
            model.set_threads(threads);
        }
        if !rest.iter().any(|a| a == "--no-gpu") {
            match model.enable_gpu() {
                Ok(resident) => println!(
                    "projections: GPU (linear_f32), {:.2} GiB resident",
                    resident as f64 / (1u64 << 30) as f64
                ),
                Err(e) => println!("projections: host ({e})"),
            }
        }
        if hot {
            // After `enable_gpu`, so only the tensors the card does not hold are copied.
            let bytes = model.prefetch_host_experts();
            println!(
                "hot experts: {:.2} GiB pre-faulted in host memory",
                bytes as f64 / (1u64 << 30) as f64
            );
        }
        let started = std::time::Instant::now();
        let all = model.prefill(&tokens, prefill_chunk);
        let elapsed = started.elapsed().as_secs_f64();
        let per = elapsed / tokens.len() as f64;
        println!(
            "prefill: {} tokens in chunks of {prefill_chunk}: {:.2}s total, {:.3}s/token, {:.2} tok/s",
            tokens.len(),
            elapsed,
            per,
            1.0 / per
        );
        let p = model.profile;
        let visits = p.visits.max(1) as f64;
        println!(
            "  attention {:.3}s, moe {:.3}s (gpu experts {:.3}s) per visit; {} visits",
            p.attention / visits,
            p.moe / visits,
            p.gpu_experts / visits,
            p.visits
        );
        let last = all.last().unwrap();
        let best = Model::top_k(last, 5);
        println!("  argmax {} ({:+.4})", best[0].0, best[0].1);
        if rest.iter().any(|a| a == "--compare-single") {
            // The single-position path is the oracle: replay the same tokens through it on
            // the same weights and compare logits position by position.
            model.reset();
            let mut single: Vec<Vec<f32>> = Vec::new();
            for &t in &tokens {
                single.push(model.forward(t));
            }
            let mut worst = 0f32;
            let mut worst_at = 0usize;
            let mut argmax_agree = 0usize;
            for (i, (a, b)) in all.iter().zip(&single).enumerate() {
                for (x, y) in a.iter().zip(b) {
                    let d = (x - y).abs();
                    if d > worst {
                        worst = d;
                        worst_at = i;
                    }
                }
                if Model::top_k(a, 1)[0].0 == Model::top_k(b, 1)[0].0 {
                    argmax_agree += 1;
                }
            }
            println!(
                "  vs single-position: max |dlogit| {worst:.6} (position {worst_at}), argmax agrees {argmax_agree}/{}",
                all.len()
            );
            let per_position: Vec<String> = all
                .iter()
                .zip(&single)
                .enumerate()
                .take(14)
                .map(|(i, (a, b))| {
                    let d = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                    format!("{i}:{d:.5}")
                })
                .collect();
            println!("  per position: {}", per_position.join(" "));
        }
        if tokens.len() > 1 {
            // Perplexity from the same logits, so a batched path can be compared with the
            // single-position one on the objective's own number.
            let mut nll = 0f64;
            let mut hits = 0usize;
            let mut scored = 0usize;
            for i in 0..tokens.len() - 1 {
                let target = tokens[i + 1] as usize;
                if target >= all[i].len() {
                    continue;
                }
                let max = all[i].iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let sum: f64 = all[i].iter().map(|v| ((*v as f64) - max).exp()).sum();
                nll -= (all[i][target] as f64 - max) - sum.ln();
                if Model::top_k(&all[i], 1)[0].0 == target {
                    hits += 1;
                }
                scored += 1;
            }
            println!(
                "  perplexity over {scored} tokens: {:.3} (top-1 {hits}/{scored})",
                (nll / scored as f64).exp()
            );
        }
        return Ok(());
    }

    if rest.iter().any(|a| a == "--reset-check") {
        let mut model = Model::open_with_router(path, router)?;
        let first: Vec<Vec<f32>> = tokens.iter().map(|&t| model.forward(t)).collect();
        model.reset();
        let second: Vec<Vec<f32>> = tokens.iter().map(|&t| model.forward(t)).collect();
        let mut worst = 0f32;
        for (a, b) in first.iter().zip(&second) {
            for (x, y) in a.iter().zip(b) {
                worst = worst.max((x - y).abs());
            }
        }
        println!("reset round trip: max |dlogit| {worst:.6}");
        return Ok(());
    }

    if rest.iter().any(|a| a == "--proj-selftest") {
        let mut model = Model::open_with_router(path, router)?;
        if !rest.iter().any(|a| a == "--no-gpu") {
            model.enable_gpu().map_err(|e| format!("gpu: {e}"))?;
        }
        let m = arg_val(rest, "--proj-m")
            .map(|v| v.parse::<usize>().unwrap_or(8))
            .unwrap_or(8);
        for (layer, worst) in model.projection_selftest(m) {
            println!(
                "layer {layer:>2} (m={m}): q {:.5} k {:.5} v {:.5} rel {:.5} gate {:.5} sink {:.5}",
                worst[0], worst[1], worst[2], worst[3], worst[4], worst[5]
            );
        }
        return Ok(());
    }

    if bench {
        // Pure host measurement: no VRAM, so it cannot collide with anything else.
        report_dequant(&model);
        return Ok(());
    }
    let want_gpu = !rest.iter().any(|a| a == "--no-gpu");
    if want_gpu {
        match model.enable_gpu() {
            Ok(resident) => println!(
                "projections: GPU (linear_f32), {:.2} GiB resident",
                resident as f64 / (1u64 << 30) as f64
            ),
            Err(e) => println!("projections: host ({e})"),
        }
    } else {
        println!("projections: host (--no-gpu)");
    }
    if route_stats > 0 {
        report_routing(&mut model, route_stats);
        return Ok(());
    }
    let load = started.elapsed();
    println!(
        "model: {} layers, {} visits, vocab {}",
        model.cfg.n_layers,
        model.visits().len(),
        model.cfg.vocab_size
    );
    println!("load: {:.1}s", load.as_secs_f64());

    let started = std::time::Instant::now();
    let mut last = Vec::new();
    for (step, &tok) in tokens.iter().enumerate() {
        let logits = model.forward(tok);
        let finite = logits.iter().filter(|v| v.is_finite()).count();
        let best = Model::top_k(&logits, top);
        let top1 = best[0];
        println!(
            "  step {step} tok {tok:>6}: |logits|={:.3} finite={finite}/{} argmax={} ({:.4})",
            logits.iter().map(|v| v * v).sum::<f32>().sqrt(),
            logits.len(),
            top1.0,
            top1.1
        );
        if step + 1 == tokens.len() {
            let report = model.last_report.clone().unwrap();
            println!(
                "    final |hidden|={:.3} block snapshots={} visits={}",
                report.hidden_norm,
                report.block_snapshots,
                report.per_visit.len()
            );
            let widths: Vec<String> = report
                .per_visit
                .iter()
                .map(|(l, n)| format!("{l}:{n:.1}"))
                .collect();
            println!("    per-visit |out|: {}", widths.join(" "));
            let attn: Vec<String> = report
                .attention
                .iter()
                .map(|t| {
                    format!(
                        "{}:pk{:.2}@{} sink{:.2} H{:.1}",
                        t.layer, t.peak, t.peak_distance, t.sink, t.entropy
                    )
                })
                .collect();
            println!("    head0 attention: {}", attn.join(" "));
            println!("    top-{top}:");
            for (id, value) in &best {
                println!("      {id:>6}  {value:+.6}");
            }
        }
        last = logits;
    }
    let elapsed = started.elapsed();
    println!(
        "prefill: {:.2}s total, {:.2}s/token",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / tokens.len() as f64
    );
    let p = model.profile;
    let per_token = tokens.len() as f64;
    if p.visits > 0 {
        println!(
            "  last token: attention {:.2}s (gpu {:.2}s) moe {:.2}s (gpu experts {:.2}s) over {} visits (moe {:.0}%)",
            p.attention,
            p.gpu_proj,
            p.moe,
            p.gpu_experts,
            p.visits,
            100.0 * p.moe / (p.attention + p.moe)
        );
        let _ = per_token;
    }

    if let Some(text) = perp {
        // Next-token log-likelihood on held-out natural text. A structurally wrong stack
        // cannot score text well, so mean NLL is the cheapest oracle available when the
        // reference checkpoint is not on disk.
        let all = tok
            .as_ref()
            .map(|t| t.encode(text, false))
            .unwrap_or_default();
        if all.len() < 2 {
            return Err("--perp text tokenizes to too few tokens".into());
        }
        // The per-step loop above already pushed these tokens through the model. Scoring
        // them again without resetting would let the model attend over its own earlier pass,
        // which makes any text look predictable: an earlier version of this reported 1.317
        // for prose whose true perplexity is ~7.6.
        model.reset();
        let mut hits = 0usize;
        let mut nll = 0f64;
        let mut skipped = 0usize;
        let mut shown = Vec::new();
        for i in 0..all.len() - 1 {
            let logits = model.forward(all[i]);
            let target = all[i + 1];
            if (target as usize) >= logits.len() {
                skipped += 1;
                continue;
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = logits.iter().map(|v| ((*v as f64) - max).exp()).sum();
            let logp = (logits[target as usize] as f64 - max) - sum.ln();
            if logp > f32::MIN as f64 {
                nll -= logp;
            }
            if Model::top_k(&logits, 1)[0].0 as u32 == target {
                hits += 1;
            }
            if shown.len() < 24 {
                let want = tok
                    .as_ref()
                    .map(|t| t.decode(&[target]))
                    .unwrap_or_default();
                let got = tok
                    .as_ref()
                    .map(|t| t.decode(&[Model::top_k(&logits, 1)[0].0 as u32]))
                    .unwrap_or_default();
                shown.push(format!(
                    "{i:>3} p={:.4} want {want:?} got {got:?}",
                    logp.exp()
                ));
            }
        }
        let scored = all.len() - 1 - skipped;
        let mean_nll = nll / scored as f64;
        println!("perplexity over {scored} tokens: {:.3}", mean_nll.exp());
        println!("  mean NLL {mean_nll:.4} nats, top-1 agreement {hits}/{scored}");
        for line in shown {
            println!("    {line}");
        }
        return Ok(());
    }

    if generate > 0 {
        let started = std::time::Instant::now();
        let mut produced = Vec::new();
        let mut next = Model::top_k(&last, 1)[0].0 as u32;
        for _ in 0..generate {
            produced.push(next);
            let logits = model.forward(next);
            next = Model::top_k(&logits, 1)[0].0 as u32;
        }
        let decoded = tok
            .as_ref()
            .map(|t| t.decode(&produced))
            .unwrap_or_else(|| format!("{produced:?}"));
        let per = started.elapsed().as_secs_f64() / generate as f64;
        println!("decode: {:.2}s/token", per);
        println!("--- generated ---\n{decoded}\n-----------------");
    }

    // Determinism is not optional: a second run of the same token sequence must reproduce
    // the logits bit for bit, otherwise nothing downstream can be trusted.
    let (windowed, full) = model.kv_bytes();
    let rss_kib = std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<u64>().ok())
        })
        .map(|pages| pages * 4)
        .unwrap_or(0);
    println!(
        "kv: windowed {:.1} MiB + full {:.1} MiB at {} positions ({} B/token for the full slots), rss {:.2} GiB",
        windowed as f64 / (1u64 << 20) as f64,
        full as f64 / (1u64 << 20) as f64,
        tokens.len(),
        if tokens.is_empty() {
            0
        } else {
            full / tokens.len()
        },
        rss_kib as f64 / (1u64 << 20) as f64,
    );

    // Replay on the same model: a second GPU model would need another copy of every resident
    // tensor, and an allocation that silently failed would compare a GPU run against a host
    // run and report "non-deterministic" for the wrong reason.
    // Both checks double the run time, so a long-context measurement skips them; the
    // 5-token probe and the 150-token ring test cover them.
    if tokens.len() > 64 || rest.iter().any(|a| a == "--no-check") {
        println!(
            "determinism: skipped on a long run ({} tokens)",
            tokens.len()
        );
        return Ok(());
    }
    model.reset();
    for &tok in &tokens[..tokens.len() - 1] {
        model.forward(tok);
    }
    let repeat = model.forward(tokens[tokens.len() - 1]);
    let same = repeat == last;
    println!("deterministic (repeat last token): {same}");
    if !same {
        let diff = repeat.iter().zip(&last).filter(|(a, b)| a != b).count();
        println!("  {diff} logits differ");
        return Err("non-deterministic logits".into());
    }

    // The host path is the oracle for the GPU path: same weights, same arithmetic, only the
    // reduction order differs.
    if want_gpu {
        let mut host = Model::open_with_router(path, router)?;
        host.max_visits = max_visits;
        if threads > 0 {
            host.set_threads(threads);
        }
        for &tok in &tokens[..tokens.len() - 1] {
            host.forward(tok);
        }
        let reference = host.forward(tokens[tokens.len() - 1]);
        let mut worst = 0f32;
        for (a, b) in reference.iter().zip(&last) {
            worst = worst.max((a - b).abs());
        }
        let agree = Model::top_k(&reference, 1)[0].0 == Model::top_k(&last, 1)[0].0;
        println!(
            "gpu-vs-host: max |dlogit| {worst:.6}, argmax agrees: {agree}, host argmax {}",
            Model::top_k(&reference, 1)[0].0
        );
    }
    Ok(())
}

fn cmd_layer0(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use scoria_engine::inkling2_layer::LayerZero;
    if args.is_empty() {
        usage();
    }
    let path = &args[0];
    let rest = &args[1..];
    let tokens: Vec<u32> = arg_val(rest, "--tokens")
        .unwrap_or("785,6722,315,9625,374")
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;

    let gguf = GgufFile::open(path)?;
    let cfg = ModelConfig::from_gguf(&gguf)?;
    let mut layer = LayerZero::load(&gguf, &cfg)?;
    let want_moe = rest.iter().any(|a| a == "--moe");
    let score = if rest.iter().any(|a| a == "--router-softmax") {
        scoria_engine::inkling2_moe::RouterScore::Softmax
    } else {
        scoria_engine::inkling2_moe::RouterScore::Sigmoid
    };
    let mut moe = if want_moe {
        Some(scoria_engine::inkling2_moe::Moe::load(&gguf, &cfg, score)?)
    } else {
        None
    };
    println!(
        "layer 0: {} | n_embd {} | {} heads x {} | {} kv heads | qk_clip {} | rel {}x{}",
        cfg.architecture,
        cfg.n_embd,
        cfg.n_heads,
        cfg.head_dim,
        cfg.n_kv_heads,
        gguf.meta_f32("inkling2_loop.attention.qk_clip")
            .unwrap_or(0.0),
        gguf.meta_u32("inkling2_loop.attention.relative_length")
            .unwrap_or(0),
        gguf.meta_u32("inkling2_loop.attention.relative_extent")
            .unwrap_or(0)
    );
    for (step, &tok) in tokens.iter().enumerate() {
        let out = layer.attention_step(tok);
        let finite = out.head0_attn.iter().all(|v| v.is_finite());
        println!(
            "  step {step} tok {tok:>6}: |q|={:.3} |k|={:.3} |v|={:.3} gate_mean={:.3} |attn|={:.3} |post|={:.3} |residual|={:.3} finite={finite}",
            out.q_norm,
            out.k_norm,
            out.v_norm,
            out.gate_mean,
            out.attn_norm,
            out.post_conv_norm,
            out.residual_norm
        );
        if let Some(moe) = moe.as_mut() {
            let ffn = layer.ffn(&gguf, moe);
            let route = moe.last_route.clone().unwrap();
            println!(
                "    ffn: |w|={:.3} |norm_in|={:.3} experts {:?} weights {:?} |moe_out|={:.3} |layer_out|={:.3}",
                ffn.weight_norm,
                ffn.input_norm,
                route.indices,
                route
                    .weights
                    .iter()
                    .map(|w| (w * 1000.0).round() / 1000.0)
                    .collect::<Vec<_>>(),
                ffn.moe_norm,
                ffn.layer_norm
            );
        }
        if step == 0 {
            println!(
                "    sink[0..4]={:?}\n    bias(at distance 0)[0..4]={:?}",
                &out.sink[..4],
                &out.bias_at_zero[..4]
            );
            println!(
                "    q head0[0..4]={:?}\n    attn head0[0..4]={:?}",
                &out.head0_q[..4],
                &out.head0_attn[..4]
            );
        }
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let res = match args.first().map(|s| s.as_str()) {
        Some("gate") => {
            let quick = args.iter().any(|a| a == "--quick");
            gate::run(quick)
        }
        Some("inspect") if args.len() >= 2 => cmd_inspect(&args[1]),
        Some("tokenize") => cmd_tokenize(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        Some("graph-soak") => cmd_graph_soak(&args[1..]),
        Some("plan") => cmd_plan(&args[1..]),
        Some("strata") => cmd_strata(&args[1..]),
        Some("arch-check") => cmd_arch_check(&args[1..]),
        Some("layer0") => cmd_layer0(&args[1..]),
        Some("logits") => cmd_logits(&args[1..]),
        _ => usage(),
    };
    if let Err(e) = res {
        eprintln!("scoria error: {e}");
        process::exit(1);
    }
}

/// How concentrated the routing is: distinct experts per layer, the share of accesses taken
/// by the top 8 experts, and the hit rate an LRU of a given size would have achieved.
fn report_routing(model: &mut scoria_engine::inkling2_model::Model, tokens: usize) {
    let layers = model.cfg.n_layers;
    let mut accesses: Vec<Vec<usize>> = vec![Vec::new(); layers];
    for step in 0..tokens {
        let tok = 300 + ((step * 37) % 4000) as u32;
        model.forward(tok);
        for (layer, list) in accesses.iter_mut().enumerate() {
            if let Some(route) = model.route_of(layer) {
                list.extend_from_slice(&route.indices);
            }
        }
    }
    let total: usize = accesses.iter().map(|a| a.len()).sum();
    println!("routing over {tokens} tokens: {total} expert accesses");

    // Concentration: what fraction of accesses go to each layer's 8 most used experts.
    let mut all_top = 0usize;
    for (layer, list) in accesses.iter().enumerate() {
        let mut counts = [0usize; 128];
        for &e in list {
            counts[e] += 1;
        }
        let mut sorted: Vec<usize> = counts.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let top8: usize = sorted.iter().take(8).sum();
        let distinct = counts.iter().filter(|c| **c > 0).count();
        all_top += top8;
        if layer % 6 == 0 || layer == layers - 1 {
            println!(
                "  layer {layer:>2}: distinct {distinct:>3}/128, top-8 share {:.1}%",
                100.0 * top8 as f64 / list.len().max(1) as f64
            );
        }
    }
    println!(
        "  all layers: top-8 share {:.1}%",
        100.0 * all_top as f64 / total.max(1) as f64
    );

    // Per-layer LRU: the only shape with a realistic memory budget. Each cached expert is
    // ~32 MB in f32 (21 MB gate/up + 11 MB down), so `cap` slots per layer costs
    // `24 * cap * 32` MiB.
    println!("  per-layer LRU (a cached expert is ~32 MiB f32):");
    for cap in [4usize, 8, 16, 32] {
        let mut hits = 0usize;
        let mut counts = 0usize;
        for list in &accesses {
            let mut lru: Vec<usize> = Vec::with_capacity(cap);
            for &e in list {
                counts += 1;
                if let Some(pos) = lru.iter().position(|k| *k == e) {
                    hits += 1;
                    let k = lru.remove(pos);
                    lru.push(k);
                } else {
                    if lru.len() == cap {
                        lru.remove(0);
                    }
                    lru.push(e);
                }
            }
        }
        println!(
            "    {cap:>2}/layer ({} MiB total): hit rate {:.1}%",
            layers * cap * 32,
            100.0 * hits as f64 / counts.max(1) as f64
        );
    }

    // And a global LRU over (layer, expert), for comparison: it thrashes because 24 layers
    // compete for one budget.
    for cap in [64usize, 256, 512] {
        let mut lru: Vec<(usize, usize)> = Vec::with_capacity(cap);
        let mut hits = 0usize;
        for (layer, list) in accesses.iter().enumerate() {
            for &e in list {
                let key = (layer, e);
                if let Some(pos) = lru.iter().position(|k| *k == key) {
                    hits += 1;
                    let k = lru.remove(pos);
                    lru.push(k);
                } else {
                    if lru.len() == cap {
                        lru.remove(0);
                    }
                    lru.push(key);
                }
            }
        }
        println!(
            "  global LRU {cap:>3} ({} MiB): hit rate {:.1}%",
            cap * 32,
            100.0 * hits as f64 / total.max(1) as f64
        );
    }
}

/// Dequantisation versus arithmetic in one expert GEMV. If dequantisation dominates, a
/// cache of dequantised experts is worth its memory; if not, it is not.
fn report_dequant(model: &scoria_engine::inkling2_model::Model) {
    for (layer, down) in [(5usize, false), (5usize, true), (0usize, false)] {
        let (dequant, dot) = model.bench_dequant(layer, down, 0, 2);
        let total = dequant + dot;
        println!(
            "  layer {layer} {}: dequant {:.3}s, dot {:.3}s -> dequant is {:.0}% of a GEMV",
            if down { "down" } else { "gate_up" },
            dequant,
            dot,
            100.0 * dequant / total.max(1e-9)
        );
    }
}
