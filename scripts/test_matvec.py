#!/usr/bin/env python3
"""Test if mat_vec_transposed produces same results as numpy for large matrices."""
import struct
import numpy as np

MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

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

def mul_ggml(w, x, ne0, ne1):
    """GGML mul_mat: y[j] = sum_i w[i+j*ne0]*x[i]"""
    return (w.reshape(ne1, ne0) @ x).astype(np.float32)

def main():
    from gguf import GGUFReader
    reader = GGUFReader(MODEL_PATH)
    
    # Load attn_q for layer 0
    tensor = None
    for t in reader.tensors:
        if t.name == 'blk.0.attn_q.weight':
            tensor = t
            break
    
    raw = tensor.data.tobytes()
    n_elems = int(np.prod(tensor.shape))
    w = dq_q4k(raw, n_elems)
    
    ne0, ne1 = 1536, 1536
    
    # Create a test input
    np.random.seed(42)
    x = np.random.randn(ne0).astype(np.float32)
    
    # Method 1: Our mul_ggml (reshape + matmul)
    y1 = mul_ggml(w, x, ne0, ne1)
    
    # Method 2: Explicit loop (like Rust)
    y2 = np.zeros(ne1, dtype=np.float32)
    for j in range(ne1):
        acc = 0.0
        for i in range(ne0):
            acc += w[i + j * ne0] * x[i]
        y2[j] = acc
    
    # Method 3: numpy dot on reshaped
    W = w.reshape(ne1, ne0)  # W[j, i] = w[j*ne0+i]
    y3 = W @ x
    
    print(f"y1 (reshape+matmul) first 5: {y1[:5]}")
    print(f"y2 (explicit loop) first 5:  {y2[:5]}")
    print(f"y3 (reshape+dot) first 5:    {y3[:5]}")
    
    print(f"\ny1 vs y2 max diff: {np.max(np.abs(y1-y2)):.2e}")
    print(f"y1 vs y3 max diff: {np.max(np.abs(y1-y3)):.2e}")
    print(f"y2 vs y3 max diff: {np.max(np.abs(y2-y3)):.2e}")
    
    # Now test with actual hidden state from embedding
    embd = None
    for t in reader.tensors:
        if t.name == 'token_embd.weight':
            embd = t
            break
    
    raw_embd = embd.data.tobytes()
    n_embd = int(np.prod(embd.shape))
    
    # Q8_0 dequant
    embd_data = np.empty(n_embd, dtype=np.float32)
    for b in range(n_embd // 32):
        off = b * 34
        d = struct.unpack('<e', bytes(raw_embd[off:off+2]))[0]
        for j in range(32):
            embd_data[b*32+j] = d * struct.unpack('b', bytes([raw_embd[off+2+j]]))[0]
    
    # Get token 49 embedding
    tok49 = embd_data[49*1536:(49+1)*1536] * 12.0
    
    # Multiply by attn_q weight
    y_tok = mul_ggml(w, tok49, ne0, ne1)
    
    print(f"\nToken 49 @ attn_q:")
    print(f"  y[0..5] = {y_tok[:5]}")
    print(f"  norm = {np.linalg.norm(y_tok):.4f}")

if __name__ == '__main__':
    main()
