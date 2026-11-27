pub mod dequant;
pub mod gguf;
pub mod tokenizer;

use std::fmt;

pub use gguf::{GgmlType, GgufFile, GgufRawLayout, GgufValue, ModelConfig, TensorInfo};
pub use tokenizer::BpeTokenizer;

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantCodec {
    F32,
    F16,
    BF16,
    Q5_0,
    Q8_0,
    Q4K,
    Q6K,
    H4F8,
    Q4W32,
    Bitcos,
    QtipW32,
    Unknown(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightTileSpec {
    pub codec: QuantCodec,
    pub tile_m: u32,
    pub tile_k: u32,
    pub align_bytes: usize,
    pub preswizzled_wave32: bool,
}

pub trait WeightLayout {
    fn spec(&self) -> WeightTileSpec;
    fn packed_bytes(&self, rows: usize, cols: usize) -> usize;
    fn metadata_bytes(&self, rows: usize, cols: usize) -> usize;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    TooShort,
    BadMagic([u8; 4]),
    UnsupportedVersion(u32),
    Io(String),
    Malformed(String),
    MissingMetadata(String),
    UnsupportedQuant(u32),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::TooShort => write!(f, "GGUF header truncated"),
            GgufError::BadMagic(m) => write!(f, "invalid GGUF magic: {:?}", m),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version: {v}"),
            GgufError::Io(msg) => write!(f, "GGUF I/O error: {msg}"),
            GgufError::Malformed(msg) => write!(f, "malformed GGUF: {msg}"),
            GgufError::MissingMetadata(key) => write!(f, "missing GGUF metadata key: {key}"),
            GgufError::UnsupportedQuant(id) => write!(f, "unsupported GGML quant type id: {id}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl GgufHeader {
    pub fn parse(bytes: &[u8]) -> Result<Self, GgufError> {
        if bytes.len() < 24 {
            return Err(GgufError::TooShort);
        }
        let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let metadata_kv_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        Ok(Self {
            version,
            tensor_count,
            metadata_kv_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_gguf_v3_header() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&42u64.to_le_bytes());
        buf.extend_from_slice(&17u64.to_le_bytes());
        let hdr = GgufHeader::parse(&buf).expect("parse GGUF v3");
        assert_eq!(hdr.version, 3);
        assert_eq!(hdr.tensor_count, 42);
        assert_eq!(hdr.metadata_kv_count, 17);
    }

    #[test]
    fn reject_bad_magic() {
        let buf = [0u8; 24];
        assert_eq!(GgufHeader::parse(&buf), Err(GgufError::BadMagic([0; 4])));
    }
}
