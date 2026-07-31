#!/usr/bin/env python3
"""Test Q6_K dequantization precision by comparing with ggml's implementation."""
import struct
import numpy as np

MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

def f16_to_f32(h):
    """Convert f16 to f32 - exact same logic as ggml."""
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

def dequant_q6_k_ours(data, n):
    """Our Q6_K dequantization - matches Rust implementation."""
    out = np.empty(n, dtype=np.float32)
    idx = 0
    n_blocks = n // 256
    for b in range(n_blocks):
        off = b * 210
        ql = data[off:off+128]
        qh = data[off+128:off+192]
        sc_raw = data[off+192:off+208]
        sc = [int(x) if x < 128 else int(x) - 256 for x in sc_raw]
        d = f16_to_f32(struct.unpack('<H', bytes(data[off+208:off+210]))[0])
        
        for (ql_off, qh_off, sc_off) in [(0, 0, 0), (64, 32, 8)]:
            for l in range(32):
                is_val = l // 16
                q1 = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) - 32
                q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) - 32
                out[idx] = d * sc[sc_off + is_val + 0] * q1; idx += 1
                out[idx] = d * sc[sc_off + is_val + 2] * q2; idx += 1
                out[idx] = d * sc[sc_off + is_val + 4] * q3; idx += 1
                out[idx] = d * sc[sc_off + is_val + 6] * q4; idx += 1
    return out

def dequant_q6_k_reference(data, n):
    """Reference Q6_K dequantization - exact ggml-quants.c implementation."""
    out = np.empty(n, dtype=np.float32)
    idx = 0
    n_blocks = n // 256
    for b in range(n_blocks):
        off = b * 210
        ql = data[off:off+128]
        qh = data[off+128:off+192]
        sc_raw = data[off+192:off+208]
        sc = [int(x) if x < 128 else int(x) - 256 for x in sc_raw]
        d = f16_to_f32(struct.unpack('<H', bytes(data[off+208:off+210]))[0])
        
        # ggml-quants.c: two groups, each with 32 iterations
        for n_group in range(2):  # QK_K/128 = 2
            for l in range(32):
                is_val = l // 16
                q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
                out[idx] = d * sc[is_val + 0] * q1; idx += 1
                out[idx] = d * sc[is_val + 2] * q2; idx += 1
                out[idx] = d * sc[is_val + 4] * q3; idx += 1
                out[idx] = d * sc[is_val + 6] * q4; idx += 1
            # Advance pointers (like ggml does)
            ql = ql[64:]
            qh = qh[32:]
            sc = sc[8:]
    return out

def main():
    from gguf import GGUFReader
    
    reader = GGUFReader(MODEL_PATH)
    
    # Test on attn_v (Q6_K) for layer 0
    for tensor_name in ['blk.0.attn_v.weight', 'blk.0.ffn_down_exps.weight']:
        tensor = None
        for t in reader.tensors:
            if t.name == tensor_name:
                tensor = t
                break
        
        raw = tensor.data.tobytes()
        n_elems = int(np.prod(tensor.shape))
        
        print(f"\n=== {tensor_name} ===")
        print(f"  type={tensor.tensor_type}, shape={list(tensor.shape)}, n_elems={n_elems}")
        
        # Dequantize both ways
        ours = dequant_q6_k_ours(raw, n_elems)
        ref = dequant_q6_k_reference(raw, n_elems)
        
        # Compare
        match = np.allclose(ours, ref, atol=1e-10)
        max_diff = np.max(np.abs(ours - ref))
        print(f"  Match: {match}")
        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Ours first 8: {ours[:8]}")
        print(f"  Ref first 8:  {ref[:8]}")
        print(f"  Ours norm: {np.linalg.norm(ours):.6f}")
        print(f"  Ref norm:  {np.linalg.norm(ref):.6f}")

if __name__ == '__main__':
    main()
