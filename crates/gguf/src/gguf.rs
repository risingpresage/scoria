use std::collections::HashMap;
use std::ffi::{CString, c_int, c_void};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr;

use crate::{GgufError, GgufHeader, QuantCodec, WeightLayout, WeightTileSpec};

const O_RDONLY: c_int = 0;
const PROT_READ: c_int = 1;
const MAP_PRIVATE: c_int = 2;
const MAP_FAILED: *mut c_void = (!0usize) as *mut c_void;

#[repr(C)]
struct Stat {
    #[cfg(target_arch = "x86_64")]
    _pad0: [u8; 48],
    #[cfg(target_arch = "x86_64")]
    st_size: i64,
    #[cfg(target_arch = "x86_64")]
    _pad1: [u8; 88],
    #[cfg(not(target_arch = "x86_64"))]
    _pad0: [u8; 48],
    #[cfg(not(target_arch = "x86_64"))]
    st_size: i64,
    #[cfg(not(target_arch = "x86_64"))]
    _pad1: [u8; 72],
}

unsafe extern "C" {
    fn open(pathname: *const i8, flags: c_int, ...) -> c_int;
    fn fstat(fd: c_int, statbuf: *mut Stat) -> c_int;
    fn close(fd: c_int) -> c_int;
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
}

pub struct MmapFile {
    ptr: *mut c_void,
    len: usize,
}

unsafe impl Send for MmapFile {}
unsafe impl Sync for MmapFile {}

impl MmapFile {
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let c_path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| GgufError::Io("path contains NUL byte".to_string()))?;
        let fd = unsafe { open(c_path.as_ptr(), O_RDONLY) };
        if fd < 0 {
            return Err(GgufError::Io(format!(
                "open({}) failed: {}",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }

        let mut st = unsafe { std::mem::zeroed::<Stat>() };
        let rc = unsafe { fstat(fd, &mut st) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            unsafe {
                close(fd);
            }
            return Err(GgufError::Io(format!(
                "fstat({}) failed: {err}",
                path.display()
            )));
        }

        if st.st_size < 0 {
            unsafe {
                close(fd);
            }
            return Err(GgufError::Io("negative file size".to_string()));
        }
        let len = st.st_size as usize;
        if len == 0 {
            unsafe {
                close(fd);
            }
            return Ok(Self {
                ptr: ptr::null_mut(),
                len: 0,
            });
        }

        let mapped = unsafe { mmap(ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, fd, 0) };
        let err = std::io::Error::last_os_error();
        unsafe {
            close(fd);
        }
        if mapped == MAP_FAILED {
            return Err(GgufError::Io(format!(
                "mmap({}) failed: {err}",
                path.display()
            )));
        }
        Ok(Self { ptr: mapped, len })
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 || self.ptr.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
        }
    }
}

impl Drop for MmapFile {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe {
                munmap(self.ptr, self.len);
            }
            self.ptr = ptr::null_mut();
            self.len = 0;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    BF16,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ4NL,
    IQ3S,
    IQ2S,
    NVFP4,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            30 => Self::BF16,
            40 => Self::NVFP4,
            other => Self::Unknown(other),
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::IQ2XXS => 16,
            Self::IQ2XS => 17,
            Self::IQ3XXS => 18,
            Self::IQ4NL => 20,
            Self::IQ3S => 21,
            Self::IQ2S => 22,
            Self::NVFP4 => 40,
            Self::BF16 => 30,
            Self::Unknown(v) => v,
        }
    }

    pub fn block_elements(self) -> Option<usize> {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => Some(1),
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => Some(32),
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::IQ2XS
            | Self::IQ2S
            | Self::IQ3XXS
            | Self::IQ3S => Some(256),
            Self::IQ4NL => Some(32),
            Self::NVFP4 => Some(64),
            Self::IQ2XXS | Self::Unknown(_) => None,
        }
    }

    pub fn block_bytes(self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::BF16 => Some(2),
            Self::Q4_0 => Some(18),
            Self::Q4_1 => Some(20),
            Self::Q5_0 => Some(22),
            Self::Q5_1 => Some(24),
            Self::Q8_0 => Some(34),
            Self::Q8_1 => Some(36),
            Self::Q2K => Some(84),
            Self::Q3K => Some(110),
            Self::Q4K => Some(144),
            Self::Q5K => Some(176),
            Self::Q6K => Some(210),
            Self::Q8K => Some(292),
            Self::IQ2XS => Some(74),
            Self::IQ2S => Some(82),
            Self::IQ3XXS => Some(98),
            Self::IQ3S => Some(110),
            Self::IQ4NL => Some(18),
            Self::NVFP4 => Some(36),
            Self::IQ2XXS | Self::Unknown(_) => None,
        }
    }

    pub fn tensor_byte_len(self, n_elements: usize) -> Result<usize, GgufError> {
        let be = self
            .block_elements()
            .ok_or_else(|| GgufError::UnsupportedQuant(self.as_u32()))?;
        let bb = self
            .block_bytes()
            .ok_or_else(|| GgufError::UnsupportedQuant(self.as_u32()))?;
        if !n_elements.is_multiple_of(be) {
            return Err(GgufError::Malformed(format!(
                "element count {n_elements} not divisible by block size {be} for {:?}",
                self
            )));
        }
        Ok((n_elements / be) * bb)
    }

    pub fn to_codec(self) -> QuantCodec {
        match self {
            Self::F32 => QuantCodec::F32,
            Self::F16 => QuantCodec::F16,
            Self::BF16 => QuantCodec::BF16,
            Self::Q5_0 => QuantCodec::Q5_0,
            Self::Q8_0 => QuantCodec::Q8_0,
            Self::Q4K => QuantCodec::Q4K,
            Self::Q6K => QuantCodec::Q6K,
            other => QuantCodec::Unknown(other.as_u32()),
        }
    }
}

pub struct GgufRawLayout(pub QuantCodec);

impl WeightLayout for GgufRawLayout {
    fn spec(&self) -> WeightTileSpec {
        let tile_k = match self.0 {
            QuantCodec::Q4K | QuantCodec::Q6K => 256,
            QuantCodec::Q5_0 | QuantCodec::Q8_0 => 32,
            _ => 1,
        };
        WeightTileSpec {
            codec: self.0,
            tile_m: 1,
            tile_k,
            align_bytes: 32,
            preswizzled_wave32: false,
        }
    }

    fn packed_bytes(&self, rows: usize, cols: usize) -> usize {
        let elems = rows * cols;
        match self.0 {
            QuantCodec::F32 => elems * 4,
            QuantCodec::F16 | QuantCodec::BF16 => elems * 2,
            QuantCodec::Q5_0 => (elems / 32) * 22,
            QuantCodec::Q8_0 => (elems / 32) * 34,
            QuantCodec::Q4K => (elems / 256) * 144,
            QuantCodec::Q6K => (elems / 256) * 210,
            _ => 0,
        }
    }

    fn metadata_bytes(&self, _rows: usize, _cols: usize) -> usize {
        0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            Self::U32(v) => Some(v),
            Self::I32(v) if v >= 0 => Some(v as u32),
            Self::U64(v) => u32::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U64(v) => Some(v),
            Self::U32(v) => Some(v as u64),
            Self::I64(v) if v >= 0 => Some(v as u64),
            Self::I32(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match *self {
            Self::I32(v) => Some(v),
            Self::U32(v) => i32::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            Self::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

impl TensorInfo {
    /// Total element count, or `None` if the shape overflows `usize`.
    ///
    /// A GGUF header is attacker-controlled input; multiplying its dimensions
    /// unchecked is a panic waiting for a malformed file.
    pub fn n_elements(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?))
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, pos }
    }

    fn read_exact(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| GgufError::Malformed("cursor overflow".to_string()))?;
        if end > self.bytes.len() {
            return Err(GgufError::Malformed(format!(
                "unexpected EOF at offset {} (needed {n} bytes, len {})",
                self.pos,
                self.bytes.len()
            )));
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.read_exact(2)?.try_into().unwrap()))
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.read_exact(2)?.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()? as usize;
        let raw = self.read_exact(len)?;
        Ok(String::from_utf8_lossy(raw).into_owned())
    }

    fn read_value(&mut self, val_type: u32) -> Result<GgufValue, GgufError> {
        match val_type {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_i8()?)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_i16()?)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_i32()?)),
            6 => Ok(GgufValue::F32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                let mut items = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    items.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array(items))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            other => Err(GgufError::Malformed(format!(
                "unknown GGUF metadata value type {other}"
            ))),
        }
    }
}

pub struct GgufFile {
    mmap: MmapFile,
    pub header: GgufHeader,
    pub alignment: usize,
    pub data_offset: usize,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let mmap = MmapFile::open(path.as_ref())?;
        let bytes = mmap.as_slice();
        let header = GgufHeader::parse(bytes)?;
        let mut cur = Cursor::new(bytes, 24);

        // A declared count is a claim, not a fact: a 24-byte file can say `u64::MAX`
        // entries and the allocator will abort. Bound every pre-allocation by what the
        // file could physically contain.
        let kv_cap = (header.metadata_kv_count as usize).min(bytes.len() / 8 + 1);
        let mut metadata = HashMap::with_capacity(kv_cap);
        for _ in 0..header.metadata_kv_count {
            let key = cur.read_string()?;
            let val_type = cur.read_u32()?;
            let val = cur.read_value(val_type)?;
            metadata.insert(key, val);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .map(|a| a.max(1) as usize)
            .unwrap_or(32);

        let n_tensors = header.tensor_count as usize;
        let tensor_cap = n_tensors.min(bytes.len() / 16 + 1);
        let mut tensors = Vec::with_capacity(tensor_cap);
        let mut tensor_index = HashMap::with_capacity(tensor_cap);
        for i in 0..n_tensors {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()? as usize;
            if n_dims > 8 {
                return Err(GgufError::Malformed(format!(
                    "tensor `{name}` has {n_dims} dims"
                )));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cur.read_u64()?);
            }
            let ggml_type = GgmlType::from_u32(cur.read_u32()?);
            let offset = cur.read_u64()?;
            tensor_index.insert(name.clone(), i);
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        let data_offset = cur.pos.div_ceil(alignment) * alignment;
        if data_offset > bytes.len() {
            return Err(GgufError::Malformed(format!(
                "data_offset {data_offset} exceeds file length {}",
                bytes.len()
            )));
        }

        Ok(Self {
            mmap,
            header,
            alignment,
            data_offset,
            metadata,
            tensors,
            tensor_index,
        })
    }

    pub fn meta(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    pub fn meta_str(&self, key: &str) -> Result<&str, GgufError> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| GgufError::MissingMetadata(key.to_string()))
    }

    pub fn meta_u32(&self, key: &str) -> Result<u32, GgufError> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| GgufError::MissingMetadata(key.to_string()))
    }

    /// Accepts either a scalar or an array, taking the largest value from an array.
    ///
    /// Some architectures (e.g. `inkling2_loop`) store per-layer head counts and
    /// sliding-window masks as arrays with one entry per block, so a scalar-only
    /// accessor rejects a perfectly valid checkpoint.
    pub fn meta_u32_any(&self, key: &str) -> Result<u32, GgufError> {
        let value = self
            .metadata
            .get(key)
            .ok_or_else(|| GgufError::MissingMetadata(key.to_string()))?;
        if let Some(v) = value.as_u32() {
            return Ok(v);
        }
        value
            .as_array()
            .and_then(|a| a.iter().filter_map(|v| v.as_u32()).max())
            .ok_or_else(|| GgufError::MissingMetadata(key.to_string()))
    }

    /// Reads an array of integers, or a scalar as a one-element list. Boolean
    /// arrays are read as 1/0, which is how sliding-window masks are stored.
    pub fn meta_u32_list(&self, key: &str) -> Option<Vec<u32>> {
        let value = self.metadata.get(key)?;
        let one = |v: &GgufValue| -> Option<u32> {
            v.as_u32().or(match v {
                GgufValue::Bool(b) => Some(u32::from(*b)),
                _ => None,
            })
        };
        if let Some(v) = one(value) {
            return Some(vec![v]);
        }
        value.as_array().map(|a| a.iter().filter_map(one).collect())
    }

    pub fn meta_f32(&self, key: &str) -> Result<f32, GgufError> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_f32())
            .ok_or_else(|| GgufError::MissingMetadata(key.to_string()))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&idx| &self.tensors[idx])
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_index.contains_key(name)
    }

    pub fn tensor_bytes(&self, info: &TensorInfo) -> Result<&[u8], GgufError> {
        let n_elems = info.n_elements().ok_or_else(|| {
            GgufError::Malformed(format!("tensor `{}` shape overflows usize", info.name))
        })?;
        let byte_len = info.ggml_type.tensor_byte_len(n_elems)?;
        let start = self
            .data_offset
            .checked_add(info.offset as usize)
            .ok_or_else(|| GgufError::Malformed("tensor offset overflow".to_string()))?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| GgufError::Malformed("tensor end overflow".to_string()))?;
        let all = self.mmap.as_slice();
        if end > all.len() {
            return Err(GgufError::Malformed(format!(
                "tensor `{}` [{start}..{end}] exceeds file length {}",
                info.name,
                all.len()
            )));
        }
        Ok(&all[start..end])
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub n_layers: usize,
    pub context_length: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
    pub tied_embeddings: bool,
    /// Per-block KV head counts when the checkpoint stores one value per layer.
    pub kv_heads_per_layer: Option<Vec<usize>>,
    /// Sliding-window size, when the architecture has both full and windowed layers.
    pub sliding_window: Option<usize>,
    /// One entry per block: non-zero marks a sliding-window layer.
    pub sliding_window_pattern: Option<Vec<u8>>,
    pub n_experts: Option<usize>,
    pub n_experts_used: Option<usize>,
    /// A block range executed `loop_count` times (looped transformers).
    pub loop_count: Option<usize>,
    pub loop_start: Option<usize>,
    pub loop_end: Option<usize>,
}

/// KV cache cost for one model, honouring per-layer sliding windows and looped
/// blocks. Sliding-window layers only ever hold `window` positions, which is the
/// difference between a 200k context fitting and not fitting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBudget {
    /// Logical layers, before loop expansion.
    pub layers: usize,
    /// KV slots after loop expansion: each loop pass gets its own cache.
    pub slots: usize,
    pub full_slots: usize,
    pub windowed_slots: usize,
    pub window: usize,
    /// Bytes per position for one full-history slot, at 1 byte per element.
    pub full_bytes_per_token: u64,
    /// Constant bytes held by all windowed slots, at 1 byte per element.
    pub windowed_bytes: u64,
    pub head_dim: usize,
}

impl KvBudget {
    /// Total cache bytes at `ctx` positions with `elem_bytes` per element
    /// (4 = FP32, 2 = FP16, 1 = FP8/INT8).
    pub fn total_bytes(&self, ctx: usize, elem_bytes: u64) -> u64 {
        let window = self.window.min(ctx) as u64;
        (self.full_bytes_per_token * ctx as u64 + self.windowed_bytes * window) * elem_bytes
    }

    /// Largest context whose cache fits in `budget_bytes`.
    pub fn max_ctx(&self, budget_bytes: u64, elem_bytes: u64) -> usize {
        let per_token = self.full_bytes_per_token * elem_bytes;
        if per_token == 0 {
            return usize::MAX;
        }
        let window = self.window.max(1) as u64;
        // Beyond the window the cost is per_token * ctx + a constant.
        let constant = self.windowed_bytes * window * elem_bytes;
        if budget_bytes > constant {
            let ctx = (budget_bytes - constant) / per_token;
            if ctx >= window {
                return ctx as usize;
            }
        }
        // Below the window, every slot grows with ctx.
        let denom = per_token + self.windowed_bytes * elem_bytes;
        (budget_bytes / denom.max(1)) as usize
    }
}

impl ModelConfig {
    /// Summarises the KV cache this checkpoint needs.
    pub fn kv_budget(&self) -> KvBudget {
        let loop_extra = match (self.loop_count, self.loop_start, self.loop_end) {
            (Some(count), Some(start), Some(end)) if end > start && count > 1 => {
                (end - start) * (count - 1)
            }
            _ => 0,
        };
        let slots = self.n_layers + loop_extra;
        let window = self.sliding_window.unwrap_or(self.context_length);
        let is_windowed = |layer: usize| -> bool {
            match (&self.sliding_window_pattern, self.sliding_window) {
                (Some(pattern), Some(_)) => pattern.get(layer).copied().unwrap_or(0) != 0,
                _ => false,
            }
        };
        let kv_heads_at = |layer: usize| -> usize {
            self.kv_heads_per_layer
                .as_ref()
                .and_then(|v| v.get(layer).copied())
                .unwrap_or(self.n_kv_heads)
        };

        let mut full_slots = 0usize;
        let mut windowed_slots = 0usize;
        let mut full_bytes_per_token = 0u64;
        let mut windowed_bytes = 0u64;
        for layer in 0..self.n_layers {
            // A looped layer is executed `passes` times; budget one cache per pass.
            let passes = if loop_extra > 0
                && self.loop_start.is_some_and(|s| layer >= s)
                && self.loop_end.is_some_and(|e| layer < e)
            {
                self.loop_count.unwrap_or(1)
            } else {
                1
            };
            let per_slot = (kv_heads_at(layer) * self.head_dim * 2) as u64;
            if is_windowed(layer) {
                windowed_slots += passes;
                windowed_bytes += per_slot * passes as u64;
            } else {
                full_slots += passes;
                full_bytes_per_token += per_slot * passes as u64;
            }
        }

        KvBudget {
            layers: self.n_layers,
            slots,
            full_slots,
            windowed_slots,
            window,
            full_bytes_per_token,
            windowed_bytes,
            head_dim: self.head_dim,
        }
    }
}

impl ModelConfig {
    pub fn from_gguf(file: &GgufFile) -> Result<Self, GgufError> {
        let arch = file.meta_str("general.architecture")?.to_string();
        let n_layers = file.meta_u32(&format!("{arch}.block_count"))? as usize;
        let context_length = file
            .meta_u32(&format!("{arch}.context_length"))
            .unwrap_or(32768) as usize;
        let n_embd = file.meta_u32(&format!("{arch}.embedding_length"))? as usize;
        let n_ff = file.meta_u32(&format!("{arch}.feed_forward_length"))? as usize;
        let n_heads = file.meta_u32_any(&format!("{arch}.attention.head_count"))? as usize;
        let n_kv_heads = file
            .meta_u32_any(&format!("{arch}.attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;
        let kv_heads_per_layer = file
            .meta_u32_list(&format!("{arch}.attention.head_count_kv"))
            .filter(|v| v.len() > 1)
            .map(|v| v.into_iter().map(|x| x as usize).collect::<Vec<_>>());
        let sliding_window = file
            .meta_u32(&format!("{arch}.attention.sliding_window"))
            .ok()
            .map(|v| v as usize);
        let sliding_window_pattern = file
            .meta_u32_list(&format!("{arch}.attention.sliding_window_pattern"))
            .filter(|v| v.len() > 1)
            .map(|v| v.into_iter().map(|x| (x != 0) as u8).collect::<Vec<_>>());
        let n_experts = file
            .meta_u32(&format!("{arch}.expert_count"))
            .ok()
            .map(|v| v as usize);
        let n_experts_used = file
            .meta_u32(&format!("{arch}.expert_used_count"))
            .ok()
            .map(|v| v as usize);
        let loop_count = file
            .meta_u32(&format!("{arch}.loop.count"))
            .ok()
            .map(|v| v as usize);
        let loop_start = file
            .meta_u32(&format!("{arch}.loop.start"))
            .ok()
            .map(|v| v as usize);
        let loop_end = file
            .meta_u32(&format!("{arch}.loop.end"))
            .ok()
            .map(|v| v as usize);
        let rope_freq_base = file
            .meta_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(10000.0);
        let rms_eps = file
            .meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let vocab_size = file
            .meta("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .ok_or_else(|| GgufError::MissingMetadata("tokenizer.ggml.tokens".to_string()))?;
        let head_dim = file
            .meta_u32(&format!("{arch}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or_else(|_| n_embd / n_heads);
        let tied_embeddings = !file.has_tensor("output.weight");

        Ok(Self {
            architecture: arch,
            n_layers,
            context_length,
            n_embd,
            n_ff,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            rope_freq_base,
            rms_eps,
            tied_embeddings,
            kv_heads_per_layer,
            sliding_window,
            sliding_window_pattern,
            n_experts,
            n_experts_used,
            loop_count,
            loop_start,
            loop_end,
        })
    }
}

#[cfg(test)]
mod kv_budget_tests {
    use super::*;

    fn cfg() -> ModelConfig {
        ModelConfig {
            architecture: "test".to_string(),
            n_layers: 24,
            context_length: 1 << 20,
            n_embd: 2304,
            n_ff: 1152,
            n_heads: 24,
            n_kv_heads: 8,
            head_dim: 128,
            vocab_size: 200058,
            rope_freq_base: 10000.0,
            rms_eps: 1e-6,
            tied_embeddings: false,
            kv_heads_per_layer: None,
            sliding_window: None,
            sliding_window_pattern: None,
            n_experts: Some(128),
            n_experts_used: Some(8),
            loop_count: None,
            loop_start: None,
            loop_end: None,
        }
    }

    #[test]
    fn uniform_full_attention_counts_every_layer() {
        let b = cfg().kv_budget();
        assert_eq!(b.full_slots, 24);
        assert_eq!(b.windowed_slots, 0);
        // 24 layers x 8 kv heads x (128 K + 128 V)
        assert_eq!(b.full_bytes_per_token, 24 * 8 * 256);
    }

    #[test]
    fn inkling_pattern_keeps_only_four_layers_full() {
        // layer 5, 11, 17, 23 are full-history; 6..18 runs three times.
        let mut c = cfg();
        c.sliding_window = Some(1024);
        c.sliding_window_pattern = Some(
            (0..24)
                .map(|l| u8::from(!matches!(l, 5 | 11 | 17 | 23)))
                .collect(),
        );
        c.loop_count = Some(3);
        c.loop_start = Some(6);
        c.loop_end = Some(18);
        let b = c.kv_budget();
        // 48 slots: 24 layers + 12 looped layers x 2 extra passes.
        assert_eq!(b.slots, 48);
        // Layers 11 and 17 are full-history *and* inside the loop, so they get three
        // caches each; layers 5 and 23 get one. 1 + 3 + 3 + 1 = 8.
        assert_eq!(b.full_slots, 8);
        assert_eq!(b.windowed_slots, 40);
        assert_eq!(b.full_bytes_per_token, 8 * 8 * 256);
        // Windowed slots stop at 1024 positions, so they are a constant.
        assert_eq!(b.windowed_bytes, 40 * 8 * 256);
        assert_eq!(b.total_bytes(1024 * 1024, 1), b.total_bytes(1 << 20, 1));
        assert_eq!(
            b.total_bytes(200_000, 1),
            b.full_bytes_per_token * 200_000 + b.windowed_bytes * 1024
        );
    }

    #[test]
    fn max_ctx_inverts_total_bytes() {
        let mut c = cfg();
        c.sliding_window = Some(1024);
        c.sliding_window_pattern = Some(vec![1; 24]);
        let b = c.kv_budget();
        let room = 3 * (1u64 << 30);
        for elem in [4u64, 2, 1] {
            let ctx = b.max_ctx(room, elem).min(c.context_length);
            assert!(b.total_bytes(ctx, elem) <= room);
            // The next token over budget means the inverse is tight.
            assert!(b.total_bytes(ctx + 1, elem) > room || ctx == c.context_length);
        }
    }
}
