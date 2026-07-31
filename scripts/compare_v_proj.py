#!/usr/bin/env python3
"""Compare V projection: manual computation vs llama.cpp."""
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
    # Load attn_norm from llama.cpp dump
    attn_norm = np.fromfile('/tmp/llama_attn_norm.bin', dtype=np.float32)
    print(f"attn_norm: n={len(attn_norm)} norm={np.linalg.norm(attn_norm):.4f}")
    print(f"attn_norm[0..5] = {attn_norm[:5]}")
    
    # Load V projection from llama.cpp dump
    v_proj_ref = np.fromfile('/tmp/llama_v_proj.bin', dtype=np.float32)
    print(f"v_proj_ref: n={len(v_proj_ref)} norm={np.linalg.norm(v_proj_ref):.4f}")
    print(f"v_proj_ref[0..5] = {v_proj_ref[:5]}")
    
    # Load V weight from GGUF
    reader = GGUFReader(MODEL)
    v_tensor = None
    for t in reader.tensors:
        if t.name == 'blk.0.attn_v.weight':
            v_tensor = t
            break
    
    raw = v_tensor.data.tobytes()
    n = int(np.prod(v_tensor.shape))
    w = dq_q6k(raw, n)
    
    # Reshape to [ne0=1536, ne1=512] column-major
    # w[i + j*1536] = tensor[i, j]
    # V projection: v[j] = sum_i tensor[i,j] * hidden[i] = sum_i w[i + j*1536] * hidden[i]
    
    # Method 1: explicit loop (same as Rust)
    v_proj_ours = np.zeros(512, dtype=np.float32)
    for j in range(512):
        for i in range(1536):
            v_proj_ours[j] += w[i + j * 1536] * attn_norm[i]
    
    print(f"\nv_proj_ours[0..5] = {v_proj_ours[:5]}")
    print(f"v_proj_ours norm = {np.linalg.norm(v_proj_ours):.4f}")
    
    # Compare
    print(f"\n=== Comparison ===")
    print(f"ref[0..5] = {v_proj_ref[:5]}")
    print(f"our[0..5] = {v_proj_ours[:5]}")
    
    max_diff = np.max(np.abs(v_proj_ref - v_proj_ours))
    corr = np.corrcoef(v_proj_ref, v_proj_ours)[0, 1]
    print(f"max_diff = {max_diff:.6f}")
    print(f"corr = {corr:.6f}")
    
    # Check per-element
    for i in range(5):
        print(f"  [{i}] ref={v_proj_ref[i]:.6f} our={v_proj_ours[i]:.6f} diff={abs(v_proj_ref[i]-v_proj_ours[i]):.6f}")

if __name__ == '__main__':
    main()
