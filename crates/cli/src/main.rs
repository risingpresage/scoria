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
        "Usage:\n  scoria gate [--quick]\n  scoria inspect <model.gguf>\n  scoria tokenize <model.gguf> <prompt> [--no-bos]\n  scoria run <model.gguf> (--prompt <text> | --tokens <csv>) [--max-tokens <N>] [--max-seq <N>] [--no-bos] [--ignore-eos] [--dump-logits <path>] [--tap-layer <L>]"
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
    let mut model = Qwen2Model::load(&ctx, &gguf, max_seq)?;

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
        _ => usage(),
    };
    if let Err(e) = res {
        eprintln!("scoria error: {e}");
        process::exit(1);
    }
}
