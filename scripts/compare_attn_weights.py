#!/usr/bin/env python3
"""Compare Q4_K dequantization for attention weights between Python and Rust."""
import struct
import numpy as np

MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

def f16_to_f32(h):
    """Convert half-precision float to single-precision."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    
    if exp == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        else:
            val = 2.0**(-14) * (mant / 1024.0)
            return -val if sign else val
    elif exp == 31:
        if mant == 0:
            return float('-inf') if sign else float('inf')
        else:
            return float('nan')
    else:
        val = 2.0**(exp - 15) * (1.0 + mant / 1024.0)
        return -val if sign else val

def get_scale_min_k4(j, scales):
    """Extract scale and min from Q4_K scales array."""
    if j < 4:
        d = scales[j] & 63
        m = scales[j + 4] & 63
    else:
        d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return d, m

def dequant_q4_k_block(block_bytes):
    """Dequantize one Q4_K superblock (144 bytes -> 256 elements)."""
    # Block layout:
    # [0..2]   d    : f16
    # [2..4]   dmin : f16
    # [4..16]  scales : [u8; 12]
    # [16..144] qs   : [u8; 128]
    
    d_raw = struct.unpack('<e', bytes(block_bytes[0:2]))[0]
    dmin_raw = struct.unpack('<e', bytes(block_bytes[2:4]))[0]
    scales = block_bytes[4:16]
    qs = block_bytes[16:144]
    
    # Guard non-finite values
    d = d_raw if np.isfinite(d_raw) else 0.0
    dmin = dmin_raw if np.isfinite(dmin_raw) else 0.0
    
    out = []
    is_idx = 0
    q_idx = 0
    
    for _ in range(4):
        sc0, m0 = get_scale_min_k4(is_idx, scales)
        d1 = d * sc0
        m1 = dmin * m0
        
        sc1, m1v = get_scale_min_k4(is_idx + 1, scales)
        d2 = d * sc1
        m2 = dmin * m1v
        
        # Low nibbles (32 elements)
        for l in range(32):
            v = d1 * (qs[q_idx + l] & 0xF) - m1
            out.append(v if np.isfinite(v) else 0.0)
        
        # High nibbles (32 elements)
        for l in range(32):
            v = d2 * (qs[q_idx + l] >> 4) - m2
            out.append(v if np.isfinite(v) else 0.0)
        
        q_idx += 32
        is_idx += 2
    
    return out

def dequant_q4_k_tensor(data, n_elems):
    """Dequantize Q4_K tensor from raw bytes."""
    n_blocks = n_elems // 256
    out = []
    
    for i in range(n_blocks):
        block_start = i * 144
        block_bytes = data[block_start:block_start + 144]
        out.extend(dequant_q4_k_block(block_bytes))
    
    return np.array(out[:n_elems], dtype=np.float32)

def main():
    from gguf import GGUFReader
    
    reader = GGUFReader(MODEL_PATH)
    
    # Find attn_q tensor for layer 0
    attn_q = None
    for t in reader.tensors:
        if t.name == 'blk.0.attn_q.weight':
            attn_q = t
            break
    
    if attn_q is None:
        print("ERROR: attn_q tensor not found")
        return
    
    print(f"Tensor: {attn_q.name}")
    print(f"Shape: {attn_q.shape}")
    print(f"Type: {attn_q.tensor_type}")
    print(f"Data shape: {attn_q.data.shape}")
    
    # Get raw bytes
    raw_data = attn_q.data.tobytes()
    print(f"Raw data length: {len(raw_data)} bytes")
    
    # Dequantize
    n_elems = 1536 * 1536
    dequantized = dequant_q4_k_tensor(raw_data, n_elems)
    
    print(f"\nDequantized shape: {dequantized.shape}")
    print(f"Dequantized first 8: {dequantized[:8]}")
    print(f"Dequantized norm: {np.linalg.norm(dequantized):.6f}")
    
    # Compare with Rust values
    rust_first_8 = [0.00014030933, -0.007876813, -0.002532065, -0.007876813, 
                    0.021519303, 0.00014030933, 0.00014030933, 0.00014030933]
    rust_norm = 23.188932
    
    print(f"\n=== Comparison with Rust ===")
    print(f"Python first 8: {dequantized[:8]}")
    print(f"Rust first 8:   {rust_first_8}")
    
    # Check if values match
    for i in range(8):
        diff = abs(dequantized[i] - rust_first_8[i])
        print(f"  [{i}] Python={dequantized[i]:.10f} Rust={rust_first_8[i]:.10f} diff={diff:.2e}")
    
    print(f"\nPython norm: {np.linalg.norm(dequantized):.6f}")
    print(f"Rust norm:   {rust_norm:.6f}")
    print(f"Norm diff:   {abs(np.linalg.norm(dequantized) - rust_norm):.6f}")

if __name__ == '__main__':
    main()
