/// GGUF binary format parser.
///
/// Parses GGUF v3 files (used by llama.cpp) into structured metadata and
/// an index of tensor names → offsets/sizes/types. The data region is
/// mmap'd or read lazily — tensor contents are loaded on demand.
///
/// Format (little-endian):
///   [4B magic "GGUF"] [u32 version] [u64 n_tensors] [u64 n_kv]
///   [KV pairs: u64 key_len + key + u32 val_type + value]
///   [Tensor infos: u64 name_len + name + u32 n_dims + u64 ne[] + u32 ggml_type + u64 offset]
///   [Padding to 32 bytes]
///   [Tensor data region]
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

// ── GGUF Types ──────────────────────────────────────────────────────

/// GGML quantization / data types (subset relevant to MoE models).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(clippy::enum_variant_names)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            16 => GgmlType::I8,
            17 => GgmlType::I16,
            18 => GgmlType::I32,
            19 => GgmlType::I64,
            20 => GgmlType::F64,
            _ => GgmlType::Unknown(v),
        }
    }

    /// Size of one element in bytes (0 for quantized types that need custom decoding).
    pub fn element_size(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F64 => 8,
            GgmlType::F16 => 2,
            GgmlType::I8 => 1,
            GgmlType::I16 => 2,
            GgmlType::I32 => 4,
            GgmlType::I64 => 8,
            _ => 0, // quantized — use block_size()
        }
    }

    /// Block size for quantized types (bytes per block).
    /// Q4_K uses 256 elements per block; see ggml quants.
    pub fn block_size(self) -> usize {
        match self {
            GgmlType::Q4_0 => 20,  // 32 elements
            GgmlType::Q4_1 => 20,  // 32 elements
            GgmlType::Q5_0 => 22,  // 32 elements
            GgmlType::Q5_1 => 22,  // 32 elements
            GgmlType::Q8_0 => 34,  // 32 elements
            GgmlType::Q8_1 => 34,  // 32 elements
            GgmlType::Q4_K => 144, // 256 elements (Q4_K superblock)
            GgmlType::Q5_K => 176, // 256 elements
            GgmlType::Q6_K => 210, // 256 elements
            GgmlType::Q8_K => 292, // 256 elements
            _ => 0,
        }
    }

    pub fn is_quantized(self) -> bool {
        self.element_size() == 0 && self.block_size() > 0
    }
}

/// GGUF KV value (a single metadata entry).
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Array { element_type: u32, values: Vec<GgufValue> },
}

// ── Parsed metadata ─────────────────────────────────────────────────

pub type GgufMetadata = HashMap<String, GgufValue>;

/// Info about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub shape: Vec<u64>,
    /// GGML type (quantized or raw).
    pub ggml_type: GgmlType,
    /// Byte offset of tensor data in the file, relative to data start.
    pub offset: u64,
    /// Size in bytes of the raw data on disk.
    pub size_bytes: u64,
}

/// Describes the tensor data region in the file.
#[derive(Debug)]
struct DataRegion {
    /// Absolute byte offset in the file where tensor data starts.
    file_offset: u64,
}

/// Parsed GGUF file (header + metadata + tensor index).
pub struct GgufFile {
    pub metadata: GgufMetadata,
    pub tensors: Vec<TensorInfo>,
    data_region: DataRegion,
    file_len: u64,
}

impl GgufFile {
    /// Open and parse a GGUF v3 file.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut f = fs::File::open(path)?;
        let file_len = f.metadata()?.len();

        // Header
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a GGUF file"));
        }
        let version = read_u32(&mut f)?;
        if version != 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version: {version}"),
            ));
        }
        let n_tensors = read_u64(&mut f)?;
        let n_kv = read_u64(&mut f)?;

        // KV pairs
        let metadata = read_kv_pairs(&mut f, n_kv)?;

        // Tensor infos
        let tensors = read_tensor_infos(&mut f, n_tensors)?;

        // Compute data offset (next 32-byte aligned position)
        let pos = f.stream_position()?;
        let align = 32;
        let data_file_offset =
            if pos % align == 0 { pos } else { (pos + align - 1) / align * align };

        Ok(GgufFile {
            metadata,
            tensors,
            data_region: DataRegion { file_offset: data_file_offset },
            file_len,
        })
    }

    /// Find a tensor by exact name.
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Absolute file offset where tensor data region starts.
    pub fn data_file_offset(&self) -> u64 {
        self.data_region.file_offset
    }

    /// Read raw tensor data as f32 from file (dequantized or directly read).
    pub fn read_tensor_f32(&self, path: &Path, info: &TensorInfo) -> io::Result<Vec<f32>> {
        let mut f = fs::File::open(path)?;
        let abs_offset = self.data_region.file_offset + info.offset;
        f.seek(SeekFrom::Start(abs_offset))?;

        let expected_f32 = shape_product(&info.shape) as usize;
        let mut out = Vec::with_capacity(expected_f32);

        match info.ggml_type {
            GgmlType::F32 => {
                // Direct read
                let mut buf = vec![0u8; expected_f32 * 4];
                f.read_exact(&mut buf)?;
                out.reserve(expected_f32);
                for chunk in buf.chunks_exact(4) {
                    out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            GgmlType::F16 => {
                let mut buf = vec![0u8; expected_f32 * 2];
                f.read_exact(&mut buf)?;
                for chunk in buf.chunks_exact(2) {
                    out.push(f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
                }
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!(
                        "tensor type {:?} not supported for direct f32 read — use dequantize",
                        info.ggml_type
                    ),
                ));
            }
        }

        Ok(out)
    }

    /// Get a metadata value as u32.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            GgufValue::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    /// Get a metadata value as f32.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            GgufValue::Float32(v) => Some(*v),
            _ => None,
        }
    }

    /// Get a metadata value as String.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

// ── Readers ───────────────────────────────────────────────────────────

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_i32<R: Read>(r: &mut R) -> io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_f64<R: Read>(r: &mut R) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_kv_pairs<R: Read>(r: &mut R, n: u64) -> io::Result<GgufMetadata> {
    let mut map = HashMap::new();
    for _ in 0..n {
        let key = read_string(r)?;
        let val_type = read_u32(r)?;
        let val = read_value(r, val_type)?;
        map.insert(key, val);
    }
    Ok(map)
}

fn read_value<R: Read>(r: &mut R, val_type: u32) -> io::Result<GgufValue> {
    Ok(match val_type {
        0 => GgufValue::Uint8(read_u8(r)?),
        1 => GgufValue::Int8(read_u8(r)? as i8),
        4 => GgufValue::Uint32(read_u32(r)?),
        5 => GgufValue::Int32(read_i32(r)?),
        6 => GgufValue::Float32(read_f32(r)?),
        7 => GgufValue::Bool(read_u8(r)? != 0),
        8 => GgufValue::String(read_string(r)?),
        9 => {
            let elem_type = read_u32(r)?;
            let len = read_u64(r)?;
            let mut values = Vec::with_capacity(len as usize);
            for _ in 0..len {
                values.push(read_value(r, elem_type)?);
            }
            GgufValue::Array { element_type: elem_type, values }
        }
        10 => GgufValue::Uint64(read_u64(r)?),
        11 => GgufValue::Int64(read_u64(r)? as i64),
        12 => GgufValue::Float64(read_f64(r)?),
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported KV value type: {other}"),
            ));
        }
    })
}

fn read_tensor_infos<R: Read>(r: &mut R, n: u64) -> io::Result<Vec<TensorInfo>> {
    let mut tensors = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let name = read_string(r)?;
        let n_dims = read_u32(r)?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(read_u64(r)?);
        }
        let ggml_type = GgmlType::from_u32(read_u32(r)?);
        let offset = read_u64(r)?;

        let size_bytes = if ggml_type.element_size() > 0 {
            let n_elems = shape_product(&shape);
            n_elems * ggml_type.element_size() as u64
        } else if ggml_type.block_size() > 0 {
            let n_elems = shape_product(&shape);
            let block_elements = if matches!(
                ggml_type,
                GgmlType::Q4_0
                    | GgmlType::Q4_1
                    | GgmlType::Q5_0
                    | GgmlType::Q5_1
                    | GgmlType::Q8_0
                    | GgmlType::Q8_1
            ) {
                32u64
            } else {
                256u64
            };
            let n_blocks = (n_elems + block_elements - 1) / block_elements;
            n_blocks * ggml_type.block_size() as u64
        } else {
            0
        };

        tensors.push(TensorInfo { name, n_dims, shape, ggml_type, offset, size_bytes });
    }
    Ok(tensors)
}

fn shape_product(shape: &[u64]) -> u64 {
    shape.iter().product()
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            (-1.0_f32).powi(sign as i32) * 0.0
        } else {
            (-1.0_f32).powi(sign as i32) * 2.0_f32.powi(-14) * (mant as f32 / 1024.0)
        }
    } else if exp == 31 {
        if mant == 0 { (-1.0_f32).powi(sign as i32) * f32::INFINITY } else { f32::NAN }
    } else {
        (-1.0_f32).powi(sign as i32) * 2.0_f32.powi(exp as i32 - 15) * (1.0 + mant as f32 / 1024.0)
    }
}

impl fmt::Display for GgufValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufValue::String(s) => write!(f, "\"{s}\""),
            GgufValue::Uint32(v) => write!(f, "{v}"),
            GgufValue::Int32(v) => write!(f, "{v}"),
            GgufValue::Float32(v) => write!(f, "{v}"),
            GgufValue::Bool(v) => write!(f, "{v}"),
            GgufValue::Uint64(v) => write!(f, "{v}"),
            GgufValue::Int64(v) => write!(f, "{v}"),
            GgufValue::Float64(v) => write!(f, "{v}"),
            GgufValue::Array { values, .. } => write!(f, "[{} items]", values.len()),
            other => write!(f, "{other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_path() -> Option<PathBuf> {
        let p = PathBuf::from(
            "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b",
        );
        if p.exists() { Some(p) } else { None }
    }

    #[test]
    fn parse_granite_moe_gguf() {
        let path = model_path().expect("Granite MoE GGUF not found");
        let gguf = GgufFile::open(&path).expect("failed to parse GGUF");

        assert_eq!(gguf.tensors.len(), 322);
        assert_eq!(gguf.metadata.len(), 42);

        assert_eq!(gguf.get_string("general.architecture").unwrap(), "granitemoe");
        assert_eq!(gguf.get_u32("granitemoe.block_count").unwrap(), 32);
        assert_eq!(gguf.get_u32("granitemoe.embedding_length").unwrap(), 1536);
        assert_eq!(gguf.get_u32("granitemoe.expert_count").unwrap(), 40);
        assert_eq!(gguf.get_u32("granitemoe.expert_used_count").unwrap(), 8);
    }

    #[test]
    fn find_gate_inp_tensor() {
        let path = model_path().expect("Granite MoE GGUF not found");
        let gguf = GgufFile::open(&path).unwrap();

        let t = gguf.find_tensor("blk.0.ffn_gate_inp.weight").expect("gate_inp tensor not found");
        assert_eq!(t.n_dims, 2);
        assert_eq!(t.shape, vec![1536, 40]);
        assert!(matches!(t.ggml_type, GgmlType::F32));
    }

    #[test]
    fn read_gate_inp_f32() {
        let path = model_path().expect("Granite MoE GGUF not found");
        let gguf = GgufFile::open(&path).unwrap();

        let info = gguf.find_tensor("blk.0.ffn_gate_inp.weight").unwrap();
        let data = gguf.read_tensor_f32(&path, info).unwrap();

        assert_eq!(data.len(), 1536 * 40);
        assert!(data.iter().any(|&v| v.abs() > 0.0), "data should be non-zero");
    }
}
