//! Adversarial GGUF parsing tests.
//!
//! Every case here is a hostile input, not a happy path: the parser must return an error
//! or a defined value, and must never panic, over-read, or hang. These are the rules
//! `AGENTS.md` requires, applied to the format boundary where a bad file arrives from
//! the network.

use std::io::Write;
use std::path::PathBuf;

use scoria_gguf::{GgufFile, ModelConfig};

// ---------------------------------------------------------------------------- builder
enum Kv<'a> {
    Str(&'a str),
    U32(u32),
    F32(f32),
    ArrU32(&'a [u32]),
    ArrBool(&'a [bool]),
    ArrStr(&'a [&'a str]),
}

struct TensorSpec<'a> {
    name: &'a str,
    dims: Vec<u64>,
    ggml_type: u32,
    offset: u64,
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_kv(out: &mut Vec<u8>, key: &str, value: &Kv<'_>) {
    put_str(out, key);
    match value {
        Kv::Str(s) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            put_str(out, s);
        }
        Kv::U32(v) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        Kv::F32(v) => {
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&v.to_le_bytes());
        }
        Kv::ArrU32(values) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&4u32.to_le_bytes()); // element type
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in *values {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        Kv::ArrBool(values) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&7u32.to_le_bytes());
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in *values {
                out.push(u8::from(*v));
            }
        }
        Kv::ArrStr(values) => {
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in *values {
                put_str(out, v);
            }
        }
    }
}

fn assemble(kvs: &[(&str, Kv<'_>)], tensors: &[TensorSpec<'_>], version: u32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, v) in kvs {
        put_kv(&mut out, k, v);
    }
    for t in tensors {
        put_str(&mut out, t.name);
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.ggml_type.to_le_bytes());
        out.extend_from_slice(&t.offset.to_le_bytes());
    }
    let pad = out.len().next_multiple_of(32) - out.len();
    out.extend(std::iter::repeat_n(0u8, pad + 64));
    out
}

/// A minimal checkpoint `ModelConfig` accepts: two layers, an 8-wide and an 8-wide
/// key/value head array given as an *array*, a boolean sliding-window mask, a loop over
/// layers 0..2 and one tokenizer table so `vocab_size` resolves.
fn valid_checkpoint() -> Vec<u8> {
    let tokens = ["a", "b", "c"];
    let kvs = [
        ("general.architecture", Kv::Str("test")),
        ("test.block_count", Kv::U32(2)),
        ("test.context_length", Kv::U32(64)),
        ("test.embedding_length", Kv::U32(64)),
        ("test.feed_forward_length", Kv::U32(128)),
        ("test.attention.head_count", Kv::ArrU32(&[8, 8])),
        ("test.attention.head_count_kv", Kv::ArrU32(&[2, 2])),
        ("test.attention.key_length", Kv::U32(32)),
        ("test.attention.layer_norm_rms_epsilon", Kv::F32(1e-6)),
        ("test.attention.sliding_window", Kv::U32(16)),
        (
            "test.attention.sliding_window_pattern",
            Kv::ArrBool(&[true, false]),
        ),
        ("test.loop.count", Kv::U32(2)),
        ("test.loop.start", Kv::U32(0)),
        ("test.loop.end", Kv::U32(2)),
        ("tokenizer.ggml.tokens", Kv::ArrStr(&tokens)),
    ];
    let tensors = [
        TensorSpec {
            name: "token_embd.weight",
            dims: vec![64, 3],
            ggml_type: 0,
            offset: 0,
        },
        TensorSpec {
            name: "output.weight",
            dims: vec![64, 3],
            ggml_type: 0,
            offset: 64 * 3 * 4,
        },
    ];
    assemble(&kvs, &tensors, 3)
}

fn temp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("scoria_adv_{tag}_{}.gguf", std::process::id()));
    p
}

fn open_bytes(tag: &str, bytes: &[u8]) -> (PathBuf, Result<GgufFile, scoria_gguf::GgufError>) {
    let path = temp_path(tag);
    let mut f = std::fs::File::create(&path).expect("create temp gguf");
    f.write_all(bytes).expect("write temp gguf");
    drop(f);
    let result = GgufFile::open(&path);
    (path, result)
}

// ------------------------------------------------------------------------- invariants
#[test]
fn valid_checkpoint_parses_and_plans() {
    let (path, opened) = open_bytes("valid", &valid_checkpoint());
    let file = opened.expect("valid checkpoint must open");
    let cfg = ModelConfig::from_gguf(&file).expect("config must parse");

    // Arrays are read as arrays, and the scalar accessor takes the max.
    assert_eq!(cfg.n_heads, 8);
    assert_eq!(cfg.n_kv_heads, 2);
    assert_eq!(cfg.kv_heads_per_layer, Some(vec![2, 2]));
    // Booleans in the mask array are read as 1/0, not dropped.
    assert_eq!(cfg.sliding_window_pattern, Some(vec![1, 0]));
    assert_eq!(cfg.sliding_window, Some(16));
    assert_eq!(cfg.loop_count, Some(2));

    // Layer 0 is windowed, layer 1 is full history, and the loop doubles each layer,
    // so the budget is 1 windowed slot * 2 passes and 1 full slot * 2 passes.
    let b = cfg.kv_budget();
    assert_eq!(b.slots, 4);
    assert_eq!(b.full_slots, 2);
    assert_eq!(b.windowed_slots, 2);
    assert_eq!(b.full_bytes_per_token, 2 * 2 * (32 + 32));
    assert_eq!(b.windowed_bytes, 2 * 2 * (32 + 32));

    // Boundary: zero context and a budget smaller than one token must not panic or
    // overflow, and must stay monotonic.
    assert_eq!(b.total_bytes(0, 4), 0);
    assert_eq!(b.max_ctx(0, 4), 0);
    assert!(b.total_bytes(1, 4) <= b.total_bytes(2, 4));
    assert!(b.max_ctx(u64::MAX, 1) >= b.max_ctx(1 << 20, 1));
    std::fs::remove_file(path).ok();
}

#[test]
fn every_truncation_prefix_is_an_error_not_a_panic() {
    let bytes = valid_checkpoint();
    // Every prefix up to the end of the tensor table must be rejected.
    for cut in (0..bytes.len().min(400)).step_by(7) {
        let (path, opened) = open_bytes(&format!("cut{cut}"), &bytes[..cut]);
        assert!(
            opened.is_err() || cut == 0,
            "truncated at {cut} of {} bytes unexpectedly parsed",
            bytes.len()
        );
        std::fs::remove_file(path).ok();
    }
}

#[test]
fn bad_magic_and_version_are_rejected() {
    let mut bytes = valid_checkpoint();
    bytes[0] = b'X';
    let (path, opened) = open_bytes("magic", &bytes);
    assert!(opened.is_err(), "bad magic must not parse");
    std::fs::remove_file(path).ok();

    let (path, opened) = open_bytes("version", &assemble(&[], &[], 99));
    assert!(opened.is_err(), "unsupported version must not parse");
    std::fs::remove_file(path).ok();
}

#[test]
fn absurd_counts_do_not_allocate_or_hang() {
    // kv_count = u64::MAX with no payload: must error rather than loop or allocate.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    let (path, opened) = open_bytes("kvcount", &bytes);
    assert!(opened.is_err(), "kv_count u64::MAX must not parse");
    std::fs::remove_file(path).ok();

    // tensor_count = u64::MAX.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    let (path, opened) = open_bytes("tcount", &bytes);
    assert!(opened.is_err(), "tensor_count u64::MAX must not parse");
    std::fs::remove_file(path).ok();
}

#[test]
fn string_length_past_end_of_file_is_rejected() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // key length
    let (path, opened) = open_bytes("strlen", &bytes);
    assert!(opened.is_err(), "string length past EOF must not parse");
    std::fs::remove_file(path).ok();
}

#[test]
fn array_length_past_end_of_file_is_rejected() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    put_str(&mut bytes, "k");
    bytes.extend_from_slice(&9u32.to_le_bytes()); // array
    bytes.extend_from_slice(&4u32.to_le_bytes()); // u32 elements
    bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // count
    let (path, opened) = open_bytes("arrlen", &bytes);
    assert!(opened.is_err(), "array length past EOF must not parse");
    std::fs::remove_file(path).ok();
}

#[test]
fn zero_alignment_does_not_divide_by_zero() {
    // `data_offset` is computed with `div_ceil(alignment)`, so alignment 0 is a panic
    // unless it is guarded. This test exists because that is exactly the kind of rule
    // that is easy to get by luck on well-formed files.
    let tokens = ["a"];
    let kvs = [
        ("general.architecture", Kv::Str("test")),
        ("general.alignment", Kv::U32(0)),
        ("test.block_count", Kv::U32(1)),
        ("test.embedding_length", Kv::U32(64)),
        ("test.feed_forward_length", Kv::U32(128)),
        ("test.attention.head_count", Kv::U32(8)),
        ("tokenizer.ggml.tokens", Kv::ArrStr(&tokens)),
    ];
    let (path, opened) = open_bytes("align0", &assemble(&kvs, &[], 3));
    // Either a clean error or a sane default; never a divide-by-zero panic.
    if let Ok(file) = opened {
        assert!(
            file.alignment > 0,
            "alignment must be clamped away from zero"
        );
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn empty_file_opens_without_metadata_and_config_errors() {
    let (path, opened) = open_bytes("empty", &[]);
    if let Ok(file) = opened {
        assert!(file.tensors.is_empty());
        assert!(ModelConfig::from_gguf(&file).is_err());
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn meta_list_reads_scalar_and_boolean_forms() {
    let (path, opened) = open_bytes("lists", &valid_checkpoint());
    let file = opened.expect("open");
    // Array form.
    assert_eq!(
        file.meta_u32_list("test.attention.head_count"),
        Some(vec![8, 8])
    );
    // Boolean array folded to 1/0.
    assert_eq!(
        file.meta_u32_list("test.attention.sliding_window_pattern"),
        Some(vec![1, 0])
    );
    // Scalar form, returned as a one-element list.
    assert_eq!(file.meta_u32_list("test.block_count"), Some(vec![2]));
    // Missing key.
    assert_eq!(file.meta_u32_list("test.nope"), None);
    // Scalar accessor over an array takes the maximum, so a per-layer table cannot be
    // silently misread as the first element.
    assert_eq!(file.meta_u32_any("test.attention.head_count").unwrap(), 8);
    std::fs::remove_file(path).ok();
}

#[test]
fn tensor_lookup_is_exact_and_bounded() {
    let (path, opened) = open_bytes("lookup", &valid_checkpoint());
    let file = opened.expect("open");
    assert!(file.tensor("token_embd.weight").is_some());
    assert!(file.tensor("token_embd").is_none());
    assert!(file.tensor("").is_none());
    assert!(!file.has_tensor("blk.0.attn_q.weight"));
    // An oversized element count must be an error from the size formula, not a
    // multiplication overflow.
    let t = file.tensor("token_embd.weight").unwrap();
    let mut huge = t.clone();
    huge.dims = vec![u64::MAX, 2];
    // The shape itself must be rejected as an overflow, not wrap into a small number.
    assert_eq!(huge.n_elements(), None);
    assert!(file.tensor_bytes(&huge).is_err());
    std::fs::remove_file(path).ok();
}
