#!/usr/bin/env python3
"""Use ggml's dequantization via ctypes to compare with our implementation."""
import ctypes
import struct
import numpy as np

MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

# Load ggml library
ggml = ctypes.CDLL("/home/ksante/.local/lib/python3.12/site-packages/lib/libggml-base.so")

# Q6_K type = 14
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q8_0 = 8
GGML_TYPE_F32 = 0

# Get type size
ggml.ggml_type_size.argtypes = [ctypes.c_int]
ggml.ggml_type_size.restype = ctypes.c_size_t

# Check type sizes
for tname, tid in [("Q4_K", 12), ("Q6_K", 14), ("Q8_0", 8)]:
    sz = ggml.ggml_type_size(tid)
    print(f"{tname} (type={tid}): block_size={sz} bytes")

# Our dequantization (same as before)
def f16_to_f32(h):
    s=(h>>15)&1; e=(h>>10)&0x1F; m=h&0x3FF
    if e==0: return (-0.0 if s else 0.0) if m==0 else ((-1 if s else 1)*2.**-14*(m/1024.))
    if e==31: return (float('-inf') if s else float('inf')) if m==0 else float('nan')
    return (-1 if s else 1)*2.**(e-15)*(1.+m/1024.)

def get_scale_min_k4(j, s):
    if j<4: return s[j]&63, s[j+4]&63
    return ((s[j+4]&0xF)|((s[j-4]>>6)<<4)), ((s[j+4]>>4)|((s[j]>>6)<<4))

def dq_q4k(d, n):
    o=np.empty(n,dtype=np.float32); idx=0
    for b in range(n//256):
        o2=b*144; dr=f16_to_f32(struct.unpack('<H',bytes(d[o2:o2+2]))[0])
        dm=f16_to_f32(struct.unpack('<H',bytes(d[o2+2:o2+4]))[0])
        dr=dr if np.isfinite(dr) else 0.; dm=dm if np.isfinite(dm) else 0.
        sc=d[o2+4:o2+16]; qs=d[o2+16:o2+144]; ii=qi=0
        for _ in range(4):
            s0,m0=get_scale_min_k4(ii,sc); s1,m1=get_scale_min_k4(ii+1,sc)
            d1,dm1=dr*s0,dm*m0; d2,dm2=dr*s1,dm*m1
            for l in range(32): v=d1*(qs[qi+l]&0xF)-dm1; o[idx]=v if np.isfinite(v) else 0.; idx+=1
            for l in range(32): v=d2*(qs[qi+l]>>4)-dm2; o[idx]=v if np.isfinite(v) else 0.; idx+=1
            qi+=32; ii+=2
    return o

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
    from gguf import GGUFReader
    reader = GGUFReader(MODEL_PATH)
    
    # Test on attn_q (Q4_K) layer 0
    for tname in ['blk.0.attn_q.weight', 'blk.0.attn_v.weight']:
        tensor = None
        for t in reader.tensors:
            if t.name == tname:
                tensor = t
                break
        
        raw = tensor.data.tobytes()
        n_elems = int(np.prod(tensor.shape))
        ttype = tensor.tensor_type
        
        print(f"\n=== {tname} ===")
        print(f"  type={ttype}, shape={list(tensor.shape)}, n_elems={n_elems}")
        
        if ttype == 12:
            ours = dq_q4k(raw, n_elems)
        elif ttype == 14:
            ours = dq_q6k(raw, n_elems)
        else:
            print(f"  Skipping type {ttype}")
            continue
        
        print(f"  Ours first 8: {ours[:8]}")
        print(f"  Ours norm: {np.linalg.norm(ours):.6f}")
        print(f"  Ours mean: {np.mean(ours):.6f}")
        print(f"  Ours std: {np.std(ours):.6f}")
        
        # Check for any NaN/Inf
        nan_count = np.sum(np.isnan(ours))
        inf_count = np.sum(np.isinf(ours))
        print(f"  NaN count: {nan_count}, Inf count: {inf_count}")

if __name__ == '__main__':
    main()
