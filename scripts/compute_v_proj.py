#!/usr/bin/env python3
"""Compute V projection manually and compare with llama.cpp."""
import struct
import numpy as np
from gguf import GGUFReader

MODEL = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

def f16_to_f32(h):
    s=(h>>15)&1; e=(h>>10)&0x1F; m=h&0x3FF
    if e==0: return (-0.0 if s else 0.0) if m==0 else ((-1 if s else 1)*2.**-14*(m/1024.))
    if e==31: return (float('-inf') if s else float('inf')) if m==0 else float('nan')
    return (-1 if s else 1)*2.**(e-15)*(1.+m/1024.)

def dq_q6k(d, n):
    o=np.empty(n,dtype=np.float32); idx=0
    for b in range(n//256):
        o2=b*210; ql=d[o2:o2+128]; qh=d[o2+128:o2+192]
        sc=[int(x) if x<128 else int(x)-256 for x in d[o2+192:o2+208]]
        dd=f16_to_f32(struct.unpack('<H',bytes(d[o2+208:o2+210]))[0])
        for (qlo,qho,so) in [(0,0,0),(64,32,8)]:
            for l in range(32):
                iv=l//16
                q1=((ql[qlo+l]&0xF)|((qh[qho+l]&3)<<4))-32
                q2=((ql[qlo+l+32]&0xF)|(((qh[qho+l]>>2)&3)<<4))-32
                q3=((ql[qlo+l]>>4)|(((qh[qho+l]>>4)&3)<<4))-32
                q4=((ql[qlo+l+32]>>4)|(((qh[qho+l]>>6)&3)<<4))-32
                o[idx]=dd*sc[so+iv+0]*q1; idx+=1
                o[idx]=dd*sc[so+iv+2]*q2; idx+=1
                o[idx]=dd*sc[so+iv+4]*q3; idx+=1
                o[idx]=dd*sc[so+iv+6]*q4; idx+=1
    return o

def main():
    reader = GGUFReader(MODEL)
    
    # Load V weight
    v_tensor = None
    for t in reader.tensors:
        if t.name == 'blk.0.attn_v.weight':
            v_tensor = t
            break
    
    raw = v_tensor.data.tobytes()
    n = int(np.prod(v_tensor.shape))
    w = dq_q6k(raw, n)
    
    print(f"V weight shape: {list(v_tensor.shape)}")
    print(f"V weight dequantized: {n} elements")
    print(f"V weight first 8: {w[:8]}")
    print(f"V weight norm: {np.linalg.norm(w):.4f}")
    
    # Reshape to [ne0=1536, ne1=512] column-major
    # w[i + j*1536] = tensor[i, j]
    W = w.reshape(512, 1536).T  # W[i, j] = w[i + j*1536]
    print(f"W shape: {W.shape}")
    print(f"W[:, 0] first 5: {W[:5, 0]}")
    
    # Load attn_norm output from llama.cpp dump
    # We'll use the values from the llama.cpp debug output
    # attn_norm: [3.298642, 3.072093, 0.123175, 2.024743, -0.718084] norm=83.6520
    
    # But we need the FULL 1536-element hidden state, not just 5 elements
    # Let me load it from the Rust test output
    
    # Actually, let me compute the V projection using the first 5 elements
    # and see if the ratio matches
    
    # For now, let me check if the dequantized values are correct
    # by comparing with the Rust output
    
    # Rust attn_q first 8: [0.00014030933, -0.007876813, -0.002532065, -0.007876813, 0.021519303, 0.00014030933, 0.00014030933, 0.00014030933]
    # This matches our dequantization
    
    # Let me check if the V weight values are correct
    # by comparing with a known reference
    
    # Actually, let me just check if the data layout is correct
    # by comparing the first few values of the first column
    
    print(f"\nFirst column (j=0) first 10: {W[:10, 0]}")
    print(f"Second column (j=1) first 10: {W[:10, 1]}")
    
    # Check if the columns are different (they should be)
    print(f"Columns 0 and 1 identical? {np.allclose(W[:, 0], W[:, 1])}")

if __name__ == '__main__':
    main()
