#!/usr/bin/env python3
"""Compare FFN output between Python and Rust for layer 0, single token."""
import struct
import numpy as np

MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

D_MODEL = 1536
N_HEADS = 24
N_KV_HEADS = 8
HEAD_DIM = 64
N_REP = N_HEADS // N_KV_HEADS
N_EXPERTS = 40
N_EXPERTS_ACTIVE = 8
D_FF = 512
VOCAB_SIZE = 49155
ROPE_THETA = 1e7
EMBEDDING_SCALE = 12.0
RESIDUAL_SCALE = 0.22
LOGIT_SCALE = 6.0
ATTENTION_SCALE = 0.015625

# ── Dequantization ──
def f16_to_f32(h):
    sign = (h >> 15) & 1; exp = (h >> 10) & 0x1F; mant = h & 0x3FF
    if exp == 0:
        return (-0.0 if sign else 0.0) if mant == 0 else ((-1 if sign else 1) * 2.0**(-14) * (mant / 1024.0))
    elif exp == 31:
        return (float('-inf') if sign else float('inf')) if mant == 0 else float('nan')
    else:
        return (-1 if sign else 1) * 2.0**(exp - 15) * (1.0 + mant / 1024.0)

def get_scale_min_k4(j, s):
    if j < 4: return s[j] & 63, s[j + 4] & 63
    return ((s[j+4] & 0xF) | ((s[j-4] >> 6) << 4)), ((s[j+4] >> 4) | ((s[j] >> 6) << 4))

def dequant_q4_k(data, n):
    out = np.empty(n, dtype=np.float32); idx = 0
    for b in range(n // 256):
        o = b * 144
        d = f16_to_f32(struct.unpack('<H', bytes(data[o:o+2]))[0])
        dm = f16_to_f32(struct.unpack('<H', bytes(data[o+2:o+4]))[0])
        d = d if np.isfinite(d) else 0.0; dm = dm if np.isfinite(dm) else 0.0
        sc = data[o+4:o+16]; qs = data[o+16:o+144]; ii = 0; qi = 0
        for _ in range(4):
            s0,m0 = get_scale_min_k4(ii, sc); s1,m1 = get_scale_min_k4(ii+1, sc)
            d1,m0v = d*s0, dm*m0; d2,m1v = d*s1, dm*m1
            for l in range(32):
                v = d1*(qs[qi+l]&0xF)-m0v; out[idx]=v if np.isfinite(v) else 0.0; idx+=1
            for l in range(32):
                v = d2*(qs[qi+l]>>4)-m1v; out[idx]=v if np.isfinite(v) else 0.0; idx+=1
            qi+=32; ii+=2
    return out

def dequant_q6_k(data, n):
    out = np.empty(n, dtype=np.float32); idx = 0
    for b in range(n // 256):
        o = b * 210
        ql = data[o:o+128]; qh = data[o+128:o+192]
        sc = [int(x) if x < 128 else int(x)-256 for x in data[o+192:o+208]]
        d = f16_to_f32(struct.unpack('<H', bytes(data[o+208:o+210]))[0])
        for (qlo,qho,so) in [(0,0,0),(64,32,8)]:
            for l in range(32):
                iv = l//16
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
    out = np.empty(n, dtype=np.float32)
    for b in range(n // 32):
        o = b*34; d = struct.unpack('<e', bytes(data[o:o+2]))[0]
        for j in range(32): out[b*32+j] = d * struct.unpack('b', bytes([data[o+2+j]]))[0]
    return out

def load_tensor(reader, name):
    for t in reader.tensors:
        if t.name == name:
            raw = t.data.tobytes(); n = int(np.prod(t.shape))
            if t.tensor_type == 0: return np.frombuffer(raw, dtype=np.float32)[:n].copy()
            if t.tensor_type == 8: return dequant_q8_0(raw, n)
            if t.tensor_type == 12: return dequant_q4_k(raw, n)
            if t.tensor_type == 14: return dequant_q6_k(raw, n)
    return None

def rms_norm(x, w, eps=1e-6):
    return x / np.sqrt(np.mean(x**2) + eps) * w

def silu(x):
    return x / (1.0 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x)); return e / np.sum(e)

def mul_mat(w, x, ne0, ne1):
    """GGML mul_mat: y[j] = sum_i w[i + j*ne0] * x[i]"""
    W = w.reshape(ne1, ne0).T  # (ne0, ne1)
    return (W.T @ x).astype(np.float32)  # (ne1,)

def main():
    from gguf import GGUFReader
    print("Loading tensors...")
    reader = GGUFReader(MODEL_PATH)
    
    embd = load_tensor(reader, 'token_embd.weight')
    out_norm = load_tensor(reader, 'output_norm.weight')
    l0_an = load_tensor(reader, 'blk.0.attn_norm.weight')
    l0_q = load_tensor(reader, 'blk.0.attn_q.weight')
    l0_k = load_tensor(reader, 'blk.0.attn_k.weight')
    l0_v = load_tensor(reader, 'blk.0.attn_v.weight')
    l0_wo = load_tensor(reader, 'blk.0.attn_output.weight')
    l0_fn = load_tensor(reader, 'blk.0.ffn_norm.weight')
    l0_gi = load_tensor(reader, 'blk.0.ffn_gate_inp.weight')
    l0_ge = load_tensor(reader, 'blk.0.ffn_gate_exps.weight')
    l0_ue = load_tensor(reader, 'blk.0.ffn_up_exps.weight')
    l0_de = load_tensor(reader, 'blk.0.ffn_down_exps.weight')
    
    # Reshape expert weights
    gate_exps = l0_ge.reshape(N_EXPERTS, D_FF, D_MODEL)  # [n_experts, d_ff, d_model]
    up_exps = l0_ue.reshape(N_EXPERTS, D_FF, D_MODEL)
    down_exps = l0_de.reshape(N_EXPERTS, D_MODEL, D_FF)  # [n_experts, d_model, d_ff]
    gate_inp = l0_gi.reshape(N_EXPERTS, D_MODEL)  # [n_experts, d_model]
    
    print(f"gate_exps shape: {gate_exps.shape}")
    print(f"gate_exps[0,0,:5] = {gate_exps[0,0,:5]}")
    print(f"down_exps shape: {down_exps.shape}")
    print(f"down_exps[0,0,:5] = {down_exps[0,0,:5]}")
    
    # Embedding
    token = 49
    hidden = embd[token*D_MODEL:(token+1)*D_MODEL] * EMBEDDING_SCALE
    residual = hidden.copy()
    
    # Attn norm
    hidden = rms_norm(hidden, l0_an)
    
    # Attention (single token: V + output proj)
    v = mul_mat(l0_v, hidden, D_MODEL, N_KV_HEADS * HEAD_DIM)
    attn_out = np.zeros(D_MODEL, dtype=np.float32)
    for h in range(N_HEADS):
        kv_h = h // N_REP
        attn_out[h*HEAD_DIM:(h+1)*HEAD_DIM] = v[kv_h*HEAD_DIM:(kv_h+1)*HEAD_DIM]
    attn_out = mul_mat(l0_wo, attn_out, D_MODEL, D_MODEL)
    
    # Residual
    hidden = residual + RESIDUAL_SCALE * attn_out
    print(f"\n[L0 RESIDUAL] hidden[0..5] = {hidden[:5]}")
    print(f"[L0 RESIDUAL] norm = {np.linalg.norm(hidden):.4f}")
    
    # FFN norm
    residual2 = hidden.copy()
    hidden = rms_norm(hidden, l0_fn)
    print(f"\n[L0 FFN_NORM] hidden[0..5] = {hidden[:5]}")
    print(f"[L0 FFN_NORM] norm = {np.linalg.norm(hidden):.4f}")
    
    # Routing
    scores = gate_inp @ hidden  # [n_experts]
    all_probs = softmax(scores)
    top_k_idx = np.argsort(all_probs)[-N_EXPERTS_ACTIVE:][::-1]
    top_k_scores = all_probs[top_k_idx]
    norm_scores = top_k_scores / np.sum(top_k_scores)
    
    print(f"\n[L0 ROUTE] experts = {list(top_k_idx)}")
    print(f"[L0 ROUTE] scores = {[f'{s:.4f}' for s in norm_scores]}")
    
    # Expert FFN
    ffn_out = np.zeros(D_MODEL, dtype=np.float32)
    for ki, e in enumerate(top_k_idx):
        score = norm_scores[ki]
        
        # gate_proj: hidden @ gate_exps[e]^T -> [d_ff]
        gate_out = gate_exps[e] @ hidden  # [d_ff, d_model] @ [d_model] = [d_ff]
        up_out = up_exps[e] @ hidden
        fused = silu(gate_out) * up_out
        
        # down_proj: fused @ down_exps[e]^T -> [d_model]
        # down_exps[e] shape: [d_model, d_ff]
        ffn_out += score * (down_exps[e] @ fused)
        
        if ki == 0:
            print(f"\n  [FFN] expert={e}, score={score:.4f}")
            print(f"  [FFN] gate_out norm={np.linalg.norm(gate_out):.4f}")
            print(f"  [FFN] up_out norm={np.linalg.norm(up_out):.4f}")
            print(f"  [FFN] fused norm={np.linalg.norm(fused):.4f}")
    
    print(f"\n[L0 FFN_OUT] ffn_out[0..5] = {ffn_out[:5]}")
    print(f"[L0 FFN_OUT] norm = {np.linalg.norm(ffn_out):.4f}")
    
    # Residual 2
    hidden = residual2 + RESIDUAL_SCALE * ffn_out
    print(f"\n[L0 FINAL] hidden[0..5] = {hidden[:5]}")
    print(f"[L0 FINAL] norm = {np.linalg.norm(hidden):.4f}")
    
    # Compare with Rust
    print(f"\n{'='*60}")
    print(f"RUST VALUES:")
    print(f"  [L0 FFN_NORM] hidden[0][0..5] = [-0.36569044, -0.7210174, -0.25504905, -2.2798188, 1.041973]")
    print(f"  [L0 FFN_NORM] norm = 75.5614")
    print(f"  [L0 FFN_OUT] ffn_out[0][0..5] = [0.95596075, -0.5432063, 0.109732, -4.006235, -0.9046377]")
    print(f"  [L0 FFN_OUT] norm = 56.2523")
    print(f"  [L0 FINAL] hidden[0][0..5] = [0.18452999, -0.17343995, -0.03152647, -1.3824012, 0.022338182]")
    print(f"  [L0 FINAL] norm = 15.8893")

if __name__ == '__main__':
    main()
