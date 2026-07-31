#!/usr/bin/env python3
"""Compare Q6_K dequantization block by block with ggml reference."""
import struct
import numpy as np
from gguf import GGUFReader

MODEL = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

def f16_to_f32(h):
    s=(h>>15)&1; e=(h>>10)&0x1F; m=h&0x3FF
    if e==0: return (-0.0 if s else 0.0) if m==0 else ((-1 if s else 1)*2.**-14*(m/1024.))
    if e==31: return (float('-inf') if s else float('inf')) if m==0 else float('nan')
    return (-1 if s else 1)*2.**(e-15)*(1.+m/1024.)

def dq_q6k_block(block_bytes):
    """Dequantize one Q6_K block (210 bytes -> 256 elements)."""
    ql = block_bytes[0:128]
    qh = block_bytes[128:192]
    sc = [int(x) if x < 128 else int(x) - 256 for x in block_bytes[192:208]]
    d = f16_to_f32(struct.unpack('<H', bytes(block_bytes[208:210]))[0])
    
    out = []
    for (qlo, qho, so) in [(0, 0, 0), (64, 32, 8)]:
        for l in range(32):
            iv = l // 16
            q1 = ((ql[qlo + l] & 0xF) | ((qh[qho + l] & 3) << 4)) - 32
            q2 = ((ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) - 32
            out.append(d * sc[so + iv + 0] * q1)
            out.append(d * sc[so + iv + 2] * q2)
            out.append(d * sc[so + iv + 4] * q3)
            out.append(d * sc[so + iv + 6] * q4)
    return out

def main():
    reader = GGUFReader(MODEL)
    v_tensor = None
    for t in reader.tensors:
        if t.name == 'blk.0.attn_v.weight':
            v_tensor = t
            break
    
    raw = v_tensor.data.tobytes()
    n_blocks = len(raw) // 210
    
    # Load attn_norm from llama.cpp
    attn_norm = np.fromfile('/tmp/llama_attn_norm.bin', dtype=np.float32)
    
    # Load V projection reference from llama.cpp
    v_ref = np.fromfile('/tmp/llama_v_proj.bin', dtype=np.float32)
    
    # Dequantize block by block and compute V projection incrementally
    v_proj = np.zeros(512, dtype=np.float32)
    
    # Each block has 256 elements. The tensor has shape [1536, 512] column-major.
    # Element index = block_idx * 256 + elem_idx
    # Column (j) = element_index // 1536
    # Row (i) = element_index % 1536
    
    for b in range(min(n_blocks, 10)):  # Check first 10 blocks
        block_start = b * 210
        block_bytes = raw[block_start:block_start + 210]
        elems = dq_q6k_block(block_bytes)
        
        # Check first block against known values
        if b == 0:
            print(f"Block 0 first 8: {elems[:8]}")
            # Known Rust values: [-0., 0.00296307, 0.00048435, -0.00777805, -0., 0.00074077, 0.01549911, -0.01495779]
            expected = [-0., 0.00296307, 0.00048435, -0.00777805, -0., 0.00074077, 0.01549911, -0.01495779]
            match = all(abs(elems[i] - expected[i]) < 1e-6 for i in range(8))
            print(f"Block 0 matches Rust: {match}")
        
        # Compute V projection contribution from this block
        for k in range(256):
            elem_idx = b * 256 + k
            j = elem_idx // 1536  # column
            i = elem_idx % 1536   # row
            if j < 512:
                v_proj[j] += elems[k] * attn_norm[i]
        
        if b < 3:
            print(f"Block {b}: d={f16_to_f32(struct.unpack('<H', bytes(block_bytes[208:210]))[0]):.6f}")
            print(f"  V proj after block {b}: first 5 = {v_proj[:5]}")
    
    print(f"\nFinal V proj (first 10 blocks): norm={np.linalg.norm(v_proj):.4f}")
    print(f"V proj ref: norm={np.linalg.norm(v_ref):.4f}")
    print(f"V proj ref[0..5] = {v_ref[:5]}")
    print(f"V proj our[0..5] = {v_proj[:5]}")
    
    # The V projection is far from complete (only 10 blocks = 2560 elements out of 786432)
    # But we can see if the pattern is correct
    
    # Let me compute the FULL V projection
    print("\nComputing full V projection (this may take a moment)...")
    v_proj_full = np.zeros(512, dtype=np.float32)
    for b in range(n_blocks):
        block_start = b * 210
        block_bytes = raw[block_start:block_start + 210]
        elems = dq_q6k_block(block_bytes)
        for k in range(256):
            elem_idx = b * 256 + k
            j = elem_idx // 1536
            i = elem_idx % 1536
            if j < 512:
                v_proj_full[j] += elems[k] * attn_norm[i]
    
    print(f"Full V proj: norm={np.linalg.norm(v_proj_full):.4f}")
    print(f"Full V proj[0..5] = {v_proj_full[:5]}")
    print(f"V proj ref[0..5] = {v_ref[:5]}")
    
    corr = np.corrcoef(v_ref, v_proj_full)[0, 1]
    print(f"Correlation: {corr:.6f}")

if __name__ == '__main__':
    main()
