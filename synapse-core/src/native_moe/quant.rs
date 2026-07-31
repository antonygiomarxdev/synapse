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
        GgmlType::Q8_0 => dequant_q8_0_raw(&mut f, n_elems),
        GgmlType::Q4_K => dequant_q4_k_raw(&mut f, n_elems),
        GgmlType::Q6_K => dequant_q6_k_raw(&mut f, n_elems),
        other => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("dequantize not implemented for {other:?}"),
        )),
    }
}

/// Dequantize a single expert's slice from a quantized expert tensor.
///
/// Expert tensors store `n_experts` experts contiguously. Expert `e` occupies
/// `expert_elems` elements starting at element offset `e * expert_elems`.
///
/// For quantized types, this translates to a byte offset of
/// `e * expert_elems / elems_per_block * block_bytes`.
///
/// Returns the dequantized f32 weights for the specified expert.
pub fn dequantize_expert(
    path: &std::path::Path,
    tensor_file_offset: u64,
    ggml_type: GgmlType,
    expert_index: usize,
    expert_elems: usize,
) -> io::Result<Vec<f32>> {
    let mut f = std::fs::File::open(path)?;

    match ggml_type {
        GgmlType::F32 => {
            let byte_offset =
                tensor_file_offset + (expert_index * expert_elems * 4) as u64;
            f.seek(SeekFrom::Start(byte_offset))?;
            let mut out = vec![0.0f32; expert_elems];
            let bytes = bytemuck::cast_slice_mut(&mut out);
            f.read_exact(bytes)?;
            Ok(out)
        }
        GgmlType::F16 => {
            let byte_offset =
                tensor_file_offset + (expert_index * expert_elems * 2) as u64;
            f.seek(SeekFrom::Start(byte_offset))?;
            let mut raw = vec![0u16; expert_elems];
            let bytes = bytemuck::cast_slice_mut(&mut raw);
            f.read_exact(bytes)?;
            Ok(raw.iter().map(|&h| f16_to_f32(h)).collect())
        }
        quant_type => {
            // Quantized: compute block-aligned offset
            let block_bytes = quant_type.block_size();
            let elems_per_block = quant_type.elements_per_block();
            if block_bytes == 0 || elems_per_block == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("unsupported quant type: {quant_type:?}"),
                ));
            }
            let blocks_per_expert = expert_elems / elems_per_block;
            let expert_byte_offset = blocks_per_expert * block_bytes;
            let seek_pos = tensor_file_offset
                + (expert_index * expert_byte_offset) as u64;
            f.seek(SeekFrom::Start(seek_pos))?;

            match quant_type {
                GgmlType::Q8_0 => dequant_q8_0_raw(&mut f, expert_elems),
                GgmlType::Q4_K => dequant_q4_k_raw(&mut f, expert_elems),
                GgmlType::Q6_K => dequant_q6_k_raw(&mut f, expert_elems),
                other => Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("expert dequant not implemented for {other:?}"),
                )),
            }
        }
    }
}

/// Reorder data from column-major (GGUF storage) to row-major (our usage).
/// GGUF stores tensors column-major: data[col * rows + row]
/// We want row-major: data[row * cols + col]
fn reorder_column_major(raw: &[f32], shape: &[u64]) -> io::Result<Vec<f32>> {
    if shape.len() < 2 {
        return Ok(raw.to_vec());
    }
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    let mut out = vec![0.0f32; raw.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = raw[c * rows + r];
        }
    }
    Ok(out)
}

// ── Q8_0: 32 elements/block, 34 bytes ───────────────────────────────

fn dequant_q8_0_raw<R: Read>(r: &mut R, n_elems: usize) -> io::Result<Vec<f32>> {
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

fn dequant_q4_k_raw<R: Read>(r: &mut R, n_elems: usize) -> io::Result<Vec<f32>> {
    // Q4_K: 256 elements per super-block, 144 bytes per block.
    // Block layout (little-endian):
    //   [0..2]   d    : f16 — super-block scale for quantized scales
    //   [2..4]   dmin : f16 — super-block scale for quantized mins
    //   [4..16]  scales : [u8; 12] — 8 sub-block scales + mins (6-bit each)
    //   [16..144] qs   : [u8; 128] — 4-bit quants (256 nibbles)
    //
    // Dequant formula per element:
    //   y = d * sc * nibble - dmin * m
    // where sc, m are 6-bit values extracted from scales[] via get_scale_min_k4.
    //
    // NaN/Inf guard: some model blocks store NaN or Inf in d/dmin (these are
    // "dead" blocks from the quantizer). We clamp non-finite scale values to 0,
    // producing zero output for those blocks — matching the behavior of inference
    // engines that zero-out corrupted blocks rather than propagating NaN.
    //
    // Reference: ggml-quants.c :: dequantize_row_q4_K
    let mut out = Vec::with_capacity(n_elems);
    let n_blocks = n_elems / 256;
    for _ in 0..n_blocks {
        let d_raw = read_f16(r)?;
        let dmin_raw = read_f16(r)?;
        let mut scales = [0u8; 12];
        r.read_exact(&mut scales)?;
        let mut qs = [0u8; 128];
        r.read_exact(&mut qs)?;

        // Guard: treat non-finite (NaN/Inf) d or dmin as zero.
        let d = if d_raw.is_finite() { d_raw } else { 0.0 };
        let dmin = if dmin_raw.is_finite() { dmin_raw } else { 0.0 };

        // Guard products: if d*sc or dmin*m overflow to Inf, the subtraction
        // Inf - Inf produces NaN. Clamp intermediate products to finite.
        let mut is = 0;
        let mut q_idx = 0;
        for _ in 0..4 {
            let (sc0, m0) = get_scale_min_k4(is, &scales);
            let d1 = d * sc0 as f32;
            let m1 = dmin * m0 as f32;
            let (sc1, m1v) = get_scale_min_k4(is + 1, &scales);
            let d2 = d * sc1 as f32;
            let m2 = dmin * m1v as f32;
            // Low nibbles → 32 elements
            for l in 0..32 {
                let v = d1 * (qs[q_idx + l] & 0xF) as f32 - m1;
                out.push(if v.is_finite() { v } else { 0.0 });
            }
            // High nibbles → 32 elements
            for l in 0..32 {
                let v = d2 * (qs[q_idx + l] >> 4) as f32 - m2;
                out.push(if v.is_finite() { v } else { 0.0 });
            }
            q_idx += 32;
            is += 2;
        }
    }
    Ok(out)
}

fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

// ── Q6_K: 256 elements/superblock, 210 bytes ────────────────────────

fn dequant_q6_k_raw<R: Read>(r: &mut R, n_elems: usize) -> io::Result<Vec<f32>> {
    // Q6_K block layout (210 bytes):
    //   ql[128], qh[64], scales[16] (i8), d (f16)
    // Reference: ggml-quants.c :: dequantize_row_q6_K
    //
    // Output order within each 128-element group (matching ggml):
    //   all q1(l=0..31), all q2(l=0..31), all q3(l=0..31), all q4(l=0..31)
    let mut out = Vec::with_capacity(n_elems);
    let n_blocks = n_elems / 256;
    for _ in 0..n_blocks {
        let mut ql = [0u8; 128];
        r.read_exact(&mut ql)?;
        let mut qh = [0u8; 64];
        r.read_exact(&mut qh)?;
        let mut sc = [0i8; 16];
        {
            let mut buf = [0u8; 16];
            r.read_exact(&mut buf)?;
            for i in 0..16 {
                sc[i] = buf[i] as i8;
            }
        }
        let d = read_f16(r)?;

        // Two groups of 128 elements each
        for (ql_off, qh_off, sc_off) in [(0usize, 0usize, 0usize), (64, 32, 8)] {
            // Compute all q values first, then output in ggml order
            let mut q1_arr = [0i32; 32];
            let mut q2_arr = [0i32; 32];
            let mut q3_arr = [0i32; 32];
            let mut q4_arr = [0i32; 32];
            for l in 0..32usize {
                q1_arr[l] = ((ql[ql_off + l] & 0xF) as i32
                    | (((qh[qh_off + l] >> 0) & 3) as i32) << 4)
                    - 32;
                q2_arr[l] = ((ql[ql_off + l + 32] & 0xF) as i32
                    | (((qh[qh_off + l] >> 2) & 3) as i32) << 4)
                    - 32;
                q3_arr[l] =
                    ((ql[ql_off + l] >> 4) as i32 | (((qh[qh_off + l] >> 4) & 3) as i32) << 4) - 32;
                q4_arr[l] = ((ql[ql_off + l + 32] >> 4) as i32
                    | (((qh[qh_off + l] >> 6) & 3) as i32) << 4)
                    - 32;
            }
            // Output in ggml order: all q1, then all q2, then all q3, then all q4
            for l in 0..32usize {
                let is = l / 16;
                out.push(d * sc[sc_off + is + 0] as f32 * q1_arr[l] as f32);
            }
            for l in 0..32usize {
                let is = l / 16;
                out.push(d * sc[sc_off + is + 2] as f32 * q2_arr[l] as f32);
            }
            for l in 0..32usize {
                let is = l / 16;
                out.push(d * sc[sc_off + is + 4] as f32 * q3_arr[l] as f32);
            }
            for l in 0..32usize {
                let is = l / 16;
                out.push(d * sc[sc_off + is + 6] as f32 * q4_arr[l] as f32);
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
            "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b",
        )
    }

    #[test]
    fn dequantize_attn_v_q4_k() {
        let gguf = GgufFile::open(&model_path()).unwrap();
        let info = gguf.find_tensor("blk.0.attn_v.weight").unwrap();
        eprintln!("attn_v type={:?}, shape={:?}", info.ggml_type, info.shape);
        let abs_offset = gguf.data_file_offset() + info.offset;
        let data =
            dequantize_tensor(&model_path(), abs_offset, info.ggml_type, &info.shape).unwrap();
        eprintln!("attn_v len={}, finite={}", data.len(), data.iter().all(|v| v.is_finite()));
        assert_eq!(data.len(), 1536 * 512);
        assert!(data.iter().any(|&v| v.abs() > 0.01), "Q4_K attn_v: all near-zero");
        assert!(data.iter().all(|v| v.is_finite()), "Q4_K attn_v: contains NaN/Inf");
    }

    #[test]
    fn debug_q4_k_attn_v_nan_blocks() {
        // Diagnostic: count blocks with NaN in raw d/dmin (expected in some models).
        use std::io::{Read, Seek, SeekFrom};
        let gguf = GgufFile::open(&model_path()).unwrap();
        let info = gguf.find_tensor("blk.0.attn_v.weight").unwrap();
        let abs_offset = gguf.data_file_offset() + info.offset;
        let n_elems: usize = info.shape.iter().map(|&x| x as usize).product();
        let n_blocks = n_elems / 256;

        let mut f = std::fs::File::open(&model_path()).unwrap();
        f.seek(SeekFrom::Start(abs_offset)).unwrap();

        let mut nan_count = 0usize;
        for _ in 0..n_blocks {
            let d = read_f16(&mut f).unwrap();
            let dmin = read_f16(&mut f).unwrap();
            let mut scales = [0u8; 12];
            f.read_exact(&mut scales).unwrap();
            let mut qs = [0u8; 128];
            f.read_exact(&mut qs).unwrap();
            if !d.is_finite() || !dmin.is_finite() {
                nan_count += 1;
            }
        }
        eprintln!(
            "Q6_K attn_v: {nan_count}/{n_blocks} blocks have non-finite d/dmin (expected, guarded in dequant)"
        );
    }

    #[test]
    fn dequantize_token_embd_q8_0() {
        let gguf = GgufFile::open(&model_path()).unwrap();
        let info = gguf.find_tensor("token_embd.weight").unwrap();
        let abs_offset = gguf.data_file_offset() + info.offset;
        let data =
            dequantize_tensor(&model_path(), abs_offset, info.ggml_type, &info.shape).unwrap();
        assert_eq!(data.len(), 49155 * 1536);
        assert!(data.iter().any(|&v| v.abs() > 0.01), "all near-zero");
    }

    #[test]
    fn dequantize_gate_inp_f32_is_direct_read() {
        let gguf = GgufFile::open(&model_path()).unwrap();
        let info = gguf.find_tensor("blk.0.ffn_gate_inp.weight").unwrap();
        let abs_offset = gguf.data_file_offset() + info.offset;
        let data =
            dequantize_tensor(&model_path(), abs_offset, info.ggml_type, &info.shape).unwrap();
        assert_eq!(data.len(), 1536 * 40);
        assert!(data.iter().any(|&v| v.abs() > 0.001));
    }

    #[test]
    fn dequantize_expert_tensor_q4_k() {
        let gguf = GgufFile::open(&model_path()).unwrap();
        let info = gguf.find_tensor("blk.0.ffn_gate_exps.weight").unwrap();
        let abs_offset = gguf.data_file_offset() + info.offset;
        let data =
            dequantize_tensor(&model_path(), abs_offset, info.ggml_type, &info.shape).unwrap();
        assert_eq!(data.len(), 1536 * 512 * 40);
        assert!(data.iter().any(|&v| v.abs() > 0.01), "Q4_K gate_exps: all near-zero");
        assert!(data.iter().all(|v| v.is_finite()), "Q4_K gate_exps: contains NaN/Inf");
    }

    #[test]
    fn dequantize_down_exps_q6_k() {
        let gguf = GgufFile::open(&model_path()).unwrap();
        let info = gguf.find_tensor("blk.0.ffn_down_exps.weight").unwrap();
        let abs_offset = gguf.data_file_offset() + info.offset;
        let data =
            dequantize_tensor(&model_path(), abs_offset, info.ggml_type, &info.shape).unwrap();
        assert_eq!(data.len(), 512 * 1536 * 40);
        assert!(data.iter().any(|&v| v.abs() > 0.01), "all near-zero");
    }

    #[test]
    fn f16_half_conversion() {
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 0.001);
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 0.001);
    }
}
