/// Dequantization of GGML quantized formats to f32.
///
/// Implements the exact dequantization algorithms from ggml-quants.c.
/// Supports Q8_0, Q4_K, Q6_K — the three formats used in Granite MoE.
use std::io::{self, Read, Seek, SeekFrom};

use crate::native_moe::gguf::GgmlType;

pub fn dequantize_tensor(
    path: &std::path::Path,
    file_offset: u64,
    ggml_type: GgmlType,
    shape: &[u64],
) -> io::Result<Vec<f32>> {
    let n_elems: usize = shape.iter().map(|&x| x as usize).product();
    let mut f = std::fs::File::open(path)?;
    f.seek(SeekFrom::Start(file_offset))?;

    match ggml_type {
        GgmlType::F32 => {
            let mut out = vec![0.0f32; n_elems];
            let bytes = bytemuck::cast_slice_mut(&mut out);
            f.read_exact(bytes)?;
            Ok(out)
        }
        GgmlType::F16 => {
            let mut raw = vec![0u16; n_elems];
            let bytes = bytemuck::cast_slice_mut(&mut raw);
            f.read_exact(bytes)?;
            Ok(raw.iter().map(|&h| f16_to_f32(h)).collect())
        }
        GgmlType::Q8_0 => dequant_q8_0(&mut f, n_elems),
        GgmlType::Q4_K => dequant_q4_k(&mut f, n_elems),
        GgmlType::Q6_K => dequant_q6_k(&mut f, n_elems),
        other => Err(io::Error::new(io::ErrorKind::Unsupported,
            format!("dequantize not implemented for {other:?}"))),
    }
}

// ── Q8_0: 32 elements/block, 34 bytes ───────────────────────────────

fn dequant_q8_0<R: Read>(r: &mut R, n_elems: usize) -> io::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n_elems);
    let n_blocks = n_elems / 32;
    for _ in 0..n_blocks {
        let d = read_f16(r)?;
        for _ in 0..32 {
            let q = read_u8(r)? as i8;
            out.push(d * q as f32);
        }
    }
    Ok(out)
}

// ── Q4_K: 256 elements/superblock, 144 bytes ────────────────────────
//
// Block layout (ggml-common.h):
//   ggml_half d;      // super-block scale for scales
//   ggml_half dmin;   // super-block scale for mins
//   uint8_t scales[12]; // 6-bit quantized scale/min pairs
//   uint8_t qs[128];    // 4-bit quants (2 per byte)
//
// Dequant (ggml-quants.c:1529):
//   d = f16_to_f32(d); min = f16_to_f32(dmin);
//   for each sub-block j (0..7, each 32 elements):
//     get_scale_min_k4(j, scales, &sc, &m)
//     d1 = d * sc; m1 = min * m;
//     for 32 elements: value = d1 * nibble - m1

fn dequant_q4_k<R: Read>(r: &mut R, n_elems: usize) -> io::Result<Vec<f32>> {
    let n_blocks = n_elems / 256;
    let mut out = Vec::with_capacity(n_elems);

    for _ in 0..n_blocks {
        let d = read_f16(r)?;
        let dmin = read_f16(r)?;

        let mut scales = [0u8; 12];
        r.read_exact(&mut scales)?;

        let mut qs = [0u8; 128];
        r.read_exact(&mut qs)?;

        // Process 4 groups of 2 sub-blocks each (64 elements per group)
        // Matches ggml-quants.c: dequantize_row_q4_K exactly
        let mut is = 0usize;
        let mut qp = 0usize;
        for _ in (0..256).step_by(64) {
            let (sc0, m0) = get_scale_min_k4(is, &scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;
            let (sc1, m1_s) = get_scale_min_k4(is + 1, &scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1_s as f32;
            for j in 0..32 {
                out.push(d1 * (qs[qp + j] & 0xF) as f32 - m1);
            }
            for j in 0..32 {
                out.push(d2 * (qs[qp + j] >> 4) as f32 - m2);
            }
            qp += 32;
            is += 2;
        }
    }
    Ok(out)
}

/// Unpacks 6-bit scale and min for Q4_K sub-block j.
/// From ggml-quants.c:get_scale_min_k4.
fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
        (d, m)
    }
}

// ── Q6_K: 256 elements/superblock, 210 bytes ────────────────────────

fn dequant_q6_k<R: Read>(r: &mut R, n_elems: usize) -> io::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n_elems);
    let n_blocks = n_elems / 256;

    for _ in 0..n_blocks {
        let mut ql = [0u8; 128];
        r.read_exact(&mut ql)?;
        let mut qh = [0u8; 64];
        r.read_exact(&mut qh)?;
        let mut scales = [0.0f32; 16];
        for s in &mut scales {
            *s = read_f16(r)?;
        }
        let d = read_f16(r)?;

        for group in 0..16 {
            let scale = d * scales[group];
            for j in 0..16 {
                let idx = group * 16 + j;
                let lo = (ql[idx / 2] >> (4 * (idx % 2))) & 0xF;
                let hi = (qh[idx / 4] >> (2 * (idx % 4))) & 0x3;
                let q = ((hi as i32) << 4 | lo as i32) - 32;
                out.push(scale * q as f32);
            }
        }
    }
    Ok(out)
}

fn read_f16<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(f16_to_f32(u16::from_le_bytes(b)))
}

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1F) as i32;
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
        (-1.0_f32).powi(sign as i32) * 2.0_f32.powi(exp - 15) * (1.0 + mant as f32 / 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_moe::gguf::GgufFile;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(
            "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"
        )
    }

    #[test]
    fn dequantize_token_embd_q8_0() {
        let gguf = GgufFile::open(&model_path()).unwrap();
        let info = gguf.find_tensor("token_embd.weight").unwrap();
        assert!(matches!(info.ggml_type, GgmlType::Q8_0));

        let abs_offset = gguf.data_file_offset() + info.offset;
        let data = dequantize_tensor(&model_path(), abs_offset, info.ggml_type, &info.shape).unwrap();

        assert_eq!(data.len(), 49155 * 1536);
        assert!(data.iter().any(|&v| v.abs() > 0.01), "all near-zero — dequant likely wrong");
    }

    #[test]
    fn dequantize_gate_inp_f32_is_direct_read() {
        let gguf = GgufFile::open(&model_path()).unwrap();
        let info = gguf.find_tensor("blk.0.ffn_gate_inp.weight").unwrap();
        assert!(matches!(info.ggml_type, GgmlType::F32));

        let abs_offset = gguf.data_file_offset() + info.offset;
        let data = dequantize_tensor(&model_path(), abs_offset, info.ggml_type, &info.shape).unwrap();

        assert_eq!(data.len(), 1536 * 40);
        assert!(data.iter().any(|&v| v.abs() > 0.001));
    }

    #[test]
    fn dequantize_expert_tensor_q4_k() {
        let gguf = GgufFile::open(&model_path()).unwrap();
        let info = gguf.find_tensor("blk.0.ffn_gate_exps.weight").unwrap();
        assert!(matches!(info.ggml_type, GgmlType::Q4_K));

        let abs_offset = gguf.data_file_offset() + info.offset;
        let data = dequantize_tensor(&model_path(), abs_offset, info.ggml_type, &info.shape).unwrap();

        assert_eq!(data.len(), 1536 * 512 * 40);
        assert!(data.iter().any(|&v| v.abs() > 0.01), "all near-zero — dequant likely wrong");
    }

    #[test]
    fn dequantize_down_exps_q6_k() {
        let gguf = GgufFile::open(&model_path()).unwrap();
        let info = gguf.find_tensor("blk.0.ffn_down_exps.weight").unwrap();
        assert!(matches!(info.ggml_type, GgmlType::Q6_K));

        let abs_offset = gguf.data_file_offset() + info.offset;
        let data = dequantize_tensor(&model_path(), abs_offset, info.ggml_type, &info.shape).unwrap();

        assert_eq!(data.len(), 512 * 1536 * 40);
        assert!(data.iter().any(|&v| v.abs() > 0.01), "all near-zero — dequant likely wrong");
    }

    #[test]
    fn f16_half_conversion() {
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 0.001);
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 0.001);
    }
}
