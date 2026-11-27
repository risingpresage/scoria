#!/usr/bin/env python3
import argparse
import ctypes
import json
import math
import os
import struct
import subprocess
import sys

LLAMA_DIR = "/home/noorm/llama.cpp/llama-b11050"


class LlamaModelParams(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("load_mode", ctypes.c_int32),
        ("lazy_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("load_mtp", ctypes.c_bool),
    ]


class LlamaContextParams(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_outputs_max_per_seq", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("moe_cache_size", ctypes.c_size_t),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.c_void_p),
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", ctypes.c_void_p),
    ]


class LlamaBatch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(ctypes.c_int32)),
        ("embd", ctypes.c_void_p),
        ("pos", ctypes.POINTER(ctypes.c_int32)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]


def load_lib():
    # Preload ggml dependencies in order
    ctypes.CDLL(os.path.join(LLAMA_DIR, "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
    ggml = ctypes.CDLL(os.path.join(LLAMA_DIR, "libggml.so"), mode=ctypes.RTLD_GLOBAL)
    ggml.ggml_backend_load_all_from_path.restype = None
    ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
    ggml.ggml_backend_load_all_from_path(LLAMA_DIR.encode("utf-8"))
    lib = ctypes.CDLL(os.path.join(LLAMA_DIR, "libllama.so"), mode=ctypes.RTLD_GLOBAL)

    lib.llama_backend_init.restype = None
    lib.llama_backend_init.argtypes = []

    lib.llama_model_default_params.restype = LlamaModelParams
    lib.llama_model_default_params.argtypes = []

    lib.llama_context_default_params.restype = LlamaContextParams
    lib.llama_context_default_params.argtypes = []

    lib.llama_model_load_from_file.restype = ctypes.c_void_p
    lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, LlamaModelParams]

    lib.llama_model_free.restype = None
    lib.llama_model_free.argtypes = [ctypes.c_void_p]

    lib.llama_model_get_vocab.restype = ctypes.c_void_p
    lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]

    lib.llama_vocab_n_tokens.restype = ctypes.c_int32
    lib.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]

    lib.llama_init_from_model.restype = ctypes.c_void_p
    lib.llama_init_from_model.argtypes = [ctypes.c_void_p, LlamaContextParams]

    lib.llama_free.restype = None
    lib.llama_free.argtypes = [ctypes.c_void_p]

    lib.llama_batch_get_one.restype = LlamaBatch
    lib.llama_batch_get_one.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]

    lib.llama_decode.restype = ctypes.c_int32
    lib.llama_decode.argtypes = [ctypes.c_void_p, LlamaBatch]

    lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
    lib.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]

    return lib


def run_oracle(model_path: str, tokens: list[int], max_tokens: int, dump_logits: str | None):
    lib = load_lib()
    lib.llama_backend_init()

    mparams = lib.llama_model_default_params()
    mparams.n_gpu_layers = 0  # Deterministic CPU reference
    model = lib.llama_model_load_from_file(model_path.encode("utf-8"), mparams)
    if not model:
        raise RuntimeError(f"llama_model_load_from_file failed for {model_path}")

    vocab = lib.llama_model_get_vocab(model)
    n_vocab = int(lib.llama_vocab_n_tokens(vocab))

    cparams = lib.llama_context_default_params()
    cparams.n_ctx = 2048
    cparams.n_batch = max(512, len(tokens))
    cparams.n_threads = 8
    cparams.n_threads_batch = 8
    cparams.no_perf = True
    ctx = lib.llama_init_from_model(model, cparams)
    if not ctx:
        lib.llama_model_free(model)
        raise RuntimeError("llama_init_from_model failed")

    arr = (ctypes.c_int32 * len(tokens))(*tokens)
    batch = lib.llama_batch_get_one(arr, len(tokens))
    rc = lib.llama_decode(ctx, batch)
    if rc != 0:
        raise RuntimeError(f"prefill llama_decode failed: {rc}")

    logits_ptr = lib.llama_get_logits_ith(ctx, len(tokens) - 1)
    prefill_logits = [logits_ptr[i] for i in range(n_vocab)]
    if dump_logits:
        with open(dump_logits, "wb") as f:
            f.write(struct.pack(f"<{n_vocab}f", *prefill_logits))

    top5 = sorted(enumerate(prefill_logits), key=lambda kv: kv[1], reverse=True)[:5]
    generated = []
    if max_tokens > 0:
        nxt = top5[0][0]
        generated.append(nxt)
        for _ in range(1, max_tokens):
            step_arr = (ctypes.c_int32 * 1)(nxt)
            step_batch = lib.llama_batch_get_one(step_arr, 1)
            rc = lib.llama_decode(ctx, step_batch)
            if rc != 0:
                raise RuntimeError(f"decode llama_decode failed: {rc}")
            lp = lib.llama_get_logits_ith(ctx, 0)
            best_i = 0
            best_v = -1e30
            for i in range(n_vocab):
                v = lp[i]
                if v > best_v:
                    best_v = v
                    best_i = i
            nxt = best_i
            generated.append(nxt)

    lib.llama_free(ctx)
    lib.llama_model_free(model)
    return {"n_vocab": n_vocab, "top5": top5, "generated": generated, "prefill_logits": prefill_logits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--dump-logits", default=None)
    args = ap.parse_args()

    tokens = [int(x.strip()) for x in args.tokens.split(",") if x.strip()]
    res = run_oracle(args.model, tokens, args.max_tokens, args.dump_logits)
    out = {
        "n_vocab": res["n_vocab"],
        "top5": res["top5"],
        "generated": res["generated"],
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
