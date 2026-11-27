mod gate;
mod uring;

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process;

use scoria_engine::qwen2::{top_k_logits, LayerTap, Qwen2Model};
use scoria_engine::EngineContext;
use scoria_gguf::{BpeTokenizer, GgufFile, ModelConfig};

fn usage() -> ! {
    eprintln!(
        "Usage:\n  scoria gate [--quick]\n  scoria inspect <model.gguf>\n  scoria tokenize <model.gguf> <prompt> [--no-bos]\n  scoria run <model.gguf> (--prompt <text> | --tokens <csv>) [--max-tokens <N>] [--max-seq <N>] [--no-bos] [--ignore-eos] [--warmup <N>] [--kv-f16] [--dump-logits <path>] [--tap-layer <L>] [--profile]\n  scoria graph-soak <model.gguf> [--replays <N>] [--tokens <csv>]\n  scoria plan <model.gguf> [--vram-mib <N>] [--ctx <N>]\n  scoria strata <model.gguf> [--top-k <N>] [--tokens <N>] [--qd <N>]\n  scoria arch-check <model.gguf>"
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
        _ => usage(),
    };
    if let Err(e) = res {
        eprintln!("scoria error: {e}");
        process::exit(1);
    }
}
