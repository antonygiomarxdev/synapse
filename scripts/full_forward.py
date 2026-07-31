#!/usr/bin/env python3
"""Full forward pass in Python, 32 layers, single token. Compare with Rust + llama-cpp."""
import struct, json, sys
import numpy as np

MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

D_MODEL = 1536; N_HEADS = 24; N_KV_HEADS = 8; HEAD_DIM = 64
N_REP = N_HEADS // N_KV_HEADS; N_EXPERTS = 40; N_EXPERTS_ACTIVE = 8
D_FF = 512; VOCAB_SIZE = 49155; ROPE_THETA = 1e7
EMBEDDING_SCALE = 12.0; RESIDUAL_SCALE = 0.22; LOGIT_SCALE = 6.0; ATTENTION_SCALE = 0.015625
N_LAYERS = 32

# ── Dequant ──
def f16_to_f32(h):
    s=(h>>15)&1; e=(h>>10)&0x1F; m=h&0x3FF
    if e==0: return (-0.0 if s else 0.0) if m==0 else ((-1 if s else 1)*2.0**(-14)*(m/1024.0))
    if e==31: return (float('-inf') if s else float('inf')) if m==0 else float('nan')
    return (-1 if s else 1)*2.0**(e-15)*(1.0+m/1024.0)

def get_scale_min_k4(j, s):
    if j<4: return s[j]&63, s[j+4]&63
    return ((s[j+4]&0xF)|((s[j-4]>>6)<<4)), ((s[j+4]>>4)|((s[j]>>6)<<4))

def dequant_q4_k(data, n):
    out=np.empty(n,dtype=np.float32); idx=0
    for b in range(n//256):
        o=b*144; d=f16_to_f32(struct.unpack('<H',bytes(data[o:o+2]))[0])
        dm=f16_to_f32(struct.unpack('<H',bytes(data[o+2:o+4]))[0])
        d=d if np.isfinite(d) else 0.0; dm=dm if np.isfinite(dm) else 0.0
        sc=data[o+4:o+16]; qs=data[o+16:o+144]; ii=0; qi=0
        for _ in range(4):
            s0,m0=get_scale_min_k4(ii,sc); s1,m1=get_scale_min_k4(ii+1,sc)
            d1,m0v=d*s0,dm*m0; d2,m1v=d*s1,dm*m1
            for l in range(32): v=d1*(qs[qi+l]&0xF)-m0v; out[idx]=v if np.isfinite(v) else 0.0; idx+=1
            for l in range(32): v=d2*(qs[qi+l]>>4)-m1v; out[idx]=v if np.isfinite(v) else 0.0; idx+=1
            qi+=32; ii+=2
    return out

def dequant_q6_k(data, n):
    out=np.empty(n,dtype=np.float32); idx=0
    for b in range(n//256):
        o=b*210; ql=data[o:o+128]; qh=data[o+128:o+192]
        sc=[int(x) if x<128 else int(x)-256 for x in data[o+192:o+208]]
        d=f16_to_f32(struct.unpack('<H',bytes(data[o+208:o+210]))[0])
        for (qlo,qho,so) in [(0,0,0),(64,32,8)]:
            for l in range(32):
                iv=l//16
                q1=((ql[qlo+l]&0xF)|((qh[qho+l]&3)<<4))-32
                q2=((ql[qlo+l+32]&0xF)|(((qh[qho+l]>>2)&3)<<4))-32
                q3=((ql[qlo+l]>>4)|(((qh[qho+l]>>4)&3)<<4))-32
                q4=((ql[qlo+l+32]>>4)|(((qh[qho+l]>>6)&3)<<4))-32
                out[idx]=d*sc[so+iv+0]*q1; idx+=1
                out[idx]=d*sc[so+iv+2]*q2; idx+=1
                out[idx]=d*sc[so+iv+4]*q3; idx+=1
                out[idx]=d*sc[so+iv+6]*q4; idx+=1
    return out

def dequant_q8_0(data, n):
    out=np.empty(n,dtype=np.float32)
    for b in range(n//32):
        o=b*34; d=struct.unpack('<e',bytes(data[o:o+2]))[0]
        for j in range(32): out[b*32+j]=d*struct.unpack('b',bytes([data[o+2+j]]))[0]
    return out

def load_tensor(reader, name):
    for t in reader.tensors:
        if t.name == name:
            raw=t.data.tobytes(); n=int(np.prod(t.shape))
            if t.tensor_type==0: return np.frombuffer(raw,dtype=np.float32)[:n].copy()
            if t.tensor_type==8: return dequant_q8_0(raw,n)
            if t.tensor_type==12: return dequant_q4_k(raw,n)
            if t.tensor_type==14: return dequant_q6_k(raw,n)
    return None

def rms_norm(x, w, eps=1e-6):
    return x / np.sqrt(np.mean(x**2) + eps) * w

def silu(x): return x / (1.0 + np.exp(-x))
def softmax(x): e=np.exp(x-np.max(x)); return e/np.sum(e)

def mul_mat(w, x, ne0, ne1):
    """GGML column-major: y[j] = sum_i w[i + j*ne0] * x[i]"""
    W = w.reshape(ne1, ne0).T
    return (W.T @ x).astype(np.float32)

def main():
    from gguf import GGUFReader
    print("Loading all tensors (this takes a moment)...")
    reader = GGUFReader(MODEL_PATH)
    
    embd = load_tensor(reader, 'token_embd.weight')
    out_norm = load_tensor(reader, 'output_norm.weight')
    
    # Load all layer weights
    layers = []
    for li in range(N_LAYERS):
        sys.stdout.write(f"\r  Loading layer {li}/{N_LAYERS-1}...")
        sys.stdout.flush()
        L = {}
        L['an'] = load_tensor(reader, f'blk.{li}.attn_norm.weight')
        L['q'] = load_tensor(reader, f'blk.{li}.attn_q.weight')
        L['k'] = load_tensor(reader, f'blk.{li}.attn_k.weight')
        L['v'] = load_tensor(reader, f'blk.{li}.attn_v.weight')
        L['wo'] = load_tensor(reader, f'blk.{li}.attn_output.weight')
        L['fn'] = load_tensor(reader, f'blk.{li}.ffn_norm.weight')
        gi = load_tensor(reader, f'blk.{li}.ffn_gate_inp.weight')
        ge = load_tensor(reader, f'blk.{li}.ffn_gate_exps.weight')
        ue = load_tensor(reader, f'blk.{li}.ffn_up_exps.weight')
        de = load_tensor(reader, f'blk.{li}.ffn_down_exps.weight')
        L['gi'] = gi.reshape(N_EXPERTS, D_MODEL)
        L['ge'] = ge.reshape(N_EXPERTS, D_FF, D_MODEL)
        L['ue'] = ue.reshape(N_EXPERTS, D_FF, D_MODEL)
        L['de'] = de.reshape(N_EXPERTS, D_MODEL, D_FF)
        layers.append(L)
    print("\nAll loaded.")
    
    # Forward pass
    token = 49
    hidden = embd[token*D_MODEL:(token+1)*D_MODEL] * EMBEDDING_SCALE
    print(f"[EMBD] hidden[0..5] = {hidden[:5]}")
    print(f"[EMBD] norm = {np.linalg.norm(hidden):.4f}")
    
    for li in range(N_LAYERS):
        L = layers[li]
        residual = hidden.copy()
        
        # Attention norm
        hidden = rms_norm(hidden, L['an'])
        
        # Q, K, V projections
        q = mul_mat(L['q'], hidden, D_MODEL, D_MODEL)
        k = mul_mat(L['k'], hidden, D_MODEL, N_KV_HEADS * HEAD_DIM)
        v = mul_mat(L['v'], hidden, D_MODEL, N_KV_HEADS * HEAD_DIM)
        
        # RoPE at pos=0 (identity)
        
        # Single token attention: V grouped by head
        attn_out = np.zeros(D_MODEL, dtype=np.float32)
        for h in range(N_HEADS):
            kv_h = h // N_REP
            attn_out[h*HEAD_DIM:(h+1)*HEAD_DIM] = v[kv_h*HEAD_DIM:(kv_h+1)*HEAD_DIM]
        
        # Output projection
        attn_out = mul_mat(L['wo'], attn_out, D_MODEL, D_MODEL)
        
        # Residual
        hidden = residual + RESIDUAL_SCALE * attn_out
        
        # FFN norm
        residual2 = hidden.copy()
        hidden = rms_norm(hidden, L['fn'])
        
        # Routing
        scores = L['gi'] @ hidden
        probs = softmax(scores)
        top_idx = np.argsort(probs)[-N_EXPERTS_ACTIVE:][::-1]
        norm_scores = probs[top_idx] / np.sum(probs[top_idx])
        
        # Expert FFN
        ffn_out = np.zeros(D_MODEL, dtype=np.float32)
        for ki, e in enumerate(top_idx):
            s = norm_scores[ki]
            gate_out = L['ge'][e] @ hidden
            up_out = L['ue'][e] @ hidden
            fused = silu(gate_out) * up_out
            ffn_out += s * (L['de'][e] @ fused)
        
        hidden = residual2 + RESIDUAL_SCALE * ffn_out
        
        if li < 3 or li == 31:
            print(f"\n[L{li} FINAL] hidden[0..5] = {hidden[:5]}")
            print(f"[L{li} FINAL] norm = {np.linalg.norm(hidden):.4f}")
        
        # Compare with Rust at specific layers
        if li == 0:
            rust_final = [0.18452999, -0.17343995, -0.03152647, -1.3824012, 0.022338182]
            print(f"  RUST:  {rust_final}")
            print(f"  MATCH: {np.allclose(hidden[:5], rust_final, atol=1e-4)}")
        elif li == 1:
            rust_final = [0.11106729, -0.16368194, -0.094443075, -1.5446923, 0.090278156]
            print(f"  RUST:  {rust_final}")
            print(f"  MATCH: {np.allclose(hidden[:5], rust_final, atol=1e-4)}")
    
    # Output norm + logits
    normed = rms_norm(hidden, out_norm)
    print(f"\n[OUTPUT_NORM] normed[0..5] = {normed[:5]}")
    print(f"[OUTPUT_NORM] norm = {np.linalg.norm(normed):.4f}")
    
    # Compute logits
    logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
    for v in range(VOCAB_SIZE):
        logits[v] = np.dot(embd[v*D_MODEL:(v+1)*D_MODEL], normed) / LOGIT_SCALE
    
    # Stats
    mean = np.mean(logits); std = np.std(logits)
    max_val = np.max(logits); min_val = np.min(logits)
    top5_idx = np.argsort(logits)[-5:][::-1]
    top5_vals = [logits[i] for i in top5_idx]
    
    print(f"\n[LOGITS] mean={mean:.4f} std={std:.4f} max={max_val:.2f} min={min_val:.2f}")
    print(f"[LOGITS] top5={list(top5_idx)} vals={[f'{v:.2f}' for v in top5_vals]}")
    
    # Compare with reference
    print(f"\n{'='*60}")
    print(f"REFERENCE (llama-cpp-python):")
    print(f"  mean=-2.7153, max=12.60")
    print(f"  top-5=[34, 36, 35, 37, 308]")
    
    # Correlation with reference
    ref_logits = np.array([-2.7153])  # placeholder
    print(f"\n  Correlation: requires full reference logits")
    
    # Save logits
    np.savetxt('/tmp/python_logits.txt', logits, fmt='%.6f')
    print(f"\nLogits saved to /tmp/python_logits.txt")

if __name__ == '__main__':
    main()
