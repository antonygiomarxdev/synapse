#!/usr/bin/env python3
"""Use ggml's dequantization via ctypes to compare element-by-element."""
import ctypes
import struct
import numpy as np

LIBGGML = "/home/ksante/.local/lib/python3.12/site-packages/lib/libggml-base.so.0"
MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

# Load ggml
ggml = ctypes.CDLL(LIBGGML)

# Get type traits
# ggml_get_type_traits(type) returns struct with to_float function
ggml.ggml_get_type_traits.argtypes = [ctypes.c_int]
ggml.ggml_get_type_traits.restype = ctypes.c_void_p

# struct ggml_type_traits has to_float at offset (depends on build)
# Let's try to find it
class ggml_type_traits(ctypes.Structure):
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ctypes.c_void_p),  # function pointer
        ("from_float", ctypes.c_void_p),
        ("from_float_reference", ctypes.c_void_p),
    ]

# Our dequantization
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
    
    # Get ggml type traits for Q4_K and Q6_K
    for tid, tname in [(12, "Q4_K"), (14, "Q6_K")]:
        ptr = ggml.ggml_get_type_traits(tid)
        if ptr:
            tt = ggml_type_traits.from_address(ptr)
            print(f"{tname}: blck_size={tt.blck_size}, type_size={tt.type_size}, "
                  f"is_quantized={tt.is_quantized}, to_float={tt.to_float:#x}")
    
    # Test on attn_q (Q4_K)
    tensor = None
    for t in reader.tensors:
        if t.name == 'blk.0.attn_q.weight':
            tensor = t
            break
    
    raw = tensor.data.tobytes()
    n_elems = int(np.prod(tensor.shape))
    
    # Our dequantization
    ours = dq_q4k(raw, n_elems)
    
    # ggml dequantization via ctypes
    ptr = ggml.ggml_get_type_traits(12)  # Q4_K
    tt = ggml_type_traits.from_address(ptr)
    
    # Create ctypes function for to_float
    to_float_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
    to_float = to_float_type(tt.to_float)
    
    # Allocate output buffer
    out = (ctypes.c_float * n_elems)()
    
    # Call ggml's dequantization
    raw_ptr = ctypes.create_string_buffer(raw)
    to_float(raw_ptr, out, n_elems)
    
    # Convert to numpy
    ggml_vals = np.array(out[:min(100, n_elems)])
    ours_vals = ours[:min(100, n_elems)]
    
    print(f"\n=== blk.0.attn_q.weight (first 100 elements) ===")
    print(f"Ours first 8: {ours_vals[:8]}")
    print(f"ggml first 8: {ggml_vals[:8]}")
    
    max_diff = np.max(np.abs(ours_vals - ggml_vals))
    print(f"Max diff: {max_diff:.2e}")
    
    if max_diff > 1e-6:
        # Find first difference
        diffs = np.abs(ours_vals - ggml_vals)
        first_diff = np.argmax(diffs > 1e-6)
        print(f"First diff at index {first_diff}: ours={ours_vals[first_diff]:.10f} ggml={ggml_vals[first_diff]:.10f}")

if __name__ == '__main__':
    main()
