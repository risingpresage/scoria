#!/usr/bin/env python3
"""Control-Candidate-Control (C-A-C) 3-trial median benchmark for scoria."""
import argparse
import datetime
import json
import os
import re
import statistics
import subprocess

DEFAULT_MODELS = [
    (
        "qwen2.5-0.5b-q4_k_m",
        "/home/noorm/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    ),
    (
        "deepseek-r1-distill-qwen-7b-q4_k_m",
        "/home/noorm/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "<｜User｜>Solve 2x + 6 = 14 for x.<｜Assistant｜><think>\n",
    ),
]


def build_128_prompt_tokens(model: str, base_prompt: str) -> list[int]:
    out = subprocess.check_output(
        ["./target/release/scoria", "tokenize", model, base_prompt], text=True
    ).strip()
    ids = json.loads(out)
    while len(ids) < 128:
        ids.extend(ids)
    return ids[:128]


def run_once(model: str, tokens: list[int], max_tokens: int) -> dict:
    csv = ",".join(str(x) for x in tokens)
    out = subprocess.check_output(
        [
            "./target/release/scoria",
            "run",
            model,
            "--tokens",
            csv,
            "--max-tokens",
            str(max_tokens),
            "--ignore-eos",
        ],
        text=True,
    )
    m = re.search(
        r"prefill_ms=([\d.]+) \(([\d.]+) tok/s\), decode_ms=([\d.]+) \(([\d.]+) tok/s\)",
        out,
    )
    if not m:
        raise RuntimeError(f"failed to parse timing from scoria run output:\n{out}")
    return {
        "prefill_ms": float(m.group(1)),
        "prefill_tok_s": float(m.group(2)),
        "decode_ms": float(m.group(3)),
        "decode_tok_s": float(m.group(4)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="Experiment tag, e.g. phase1_ctrl or step3_gemv")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--ledger", default="harness/ledger.jsonl")
    ap.add_argument("--only-7b", action="store_true")
    args = ap.parse_args()

    models = DEFAULT_MODELS[1:] if args.only_7b else DEFAULT_MODELS
    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tag": args.tag,
        "trials": args.trials,
        "decode_tokens": args.decode_tokens,
        "results": {},
    }

    for name, path, prompt in models:
        short_toks = json.loads(
            subprocess.check_output(
                ["./target/release/scoria", "tokenize", path, prompt], text=True
            ).strip()
        )
        long_toks = build_128_prompt_tokens(path, prompt)

        short_trials = [run_once(path, short_toks, args.decode_tokens) for _ in range(args.trials)]
        long_trials = [run_once(path, long_toks, 1) for _ in range(args.trials)]

        short_dec = statistics.median(t["decode_tok_s"] for t in short_trials)
        short_pre = statistics.median(t["prefill_tok_s"] for t in short_trials)
        long_pre = statistics.median(t["prefill_tok_s"] for t in long_trials)

        summary["results"][name] = {
            "short_prompt_tokens": len(short_toks),
            "short_prefill_tok_s_median": round(short_pre, 2),
            "short_decode_tok_s_median": round(short_dec, 2),
            "prefill_128_tok_s_median": round(long_pre, 2),
            "short_trials": short_trials,
            "long_trials": long_trials,
        }
        print(
            f"[{args.tag}] {name}: decode={short_dec:.2f} tok/s | "
            f"prefill({len(short_toks)})={short_pre:.2f} tok/s | "
            f"prefill(128)={long_pre:.2f} tok/s"
        )

    os.makedirs(os.path.dirname(args.ledger), exist_ok=True)
    with open(args.ledger, "a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
