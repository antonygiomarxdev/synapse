#!/usr/bin/env python3
"""Compare attention output between Python and Rust for layer 0, single token."""
import struct
import numpy as np

MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

D_MODEL = 1536
N_HEADS = 24
N_KV_HEADS = 8
HEAD_DIM = 64
N_REP = N_HEADS // N_KV_HEADS  # 3
EMBEDDING_SCALE = 12.0
RESIDUAL_SCALE = 0.22
LOGIT_SCALE = 6.0
ATTENTION_SCALE = 0.015625
ROPE_THETA = 1e7

# ── Dequantization (same as before) ──

def f16_to_f32(h):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        return (-0.0 if sign else 0.0) if mant == 0 else ((-1 if sign else 1) * 2.0**(-14) * (mant / 1024.0))
    elif exp == 31:
        return (float('-inf') if sign else float('inf')) if mant == 0 else float('nan')
    else:
        return (-1 if sign else 1) * 2.0**(exp - 15) * (1.0 + mant / 1024.0)

def get_scale_min_k4(j, scales):
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    else:
        return ((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)), ((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4))

def dequant_q4_k(data, n_elems):
    n_blocks = n_elems // 256
    out = np.empty(n_elems, dtype=np.float32)
    idx = 0
    for b in range(n_blocks):
        off = b * 144
        d_raw = f16_to_f32(struct.unpack('<H', bytes(data[off:off+2]))[0])
        dmin_raw = f16_to_f32(struct.unpack('<H', bytes(data[off+2:off+4]))[0])
        scales = data[off+4:off+16]
        qs = data[off+16:off+144]
        d = d_raw if np.isfinite(d_raw) else 0.0
        dmin = dmin_raw if np.isfinite(dmin_raw) else 0.0
        is_idx, q_idx = 0, 0
        for _ in range(4):
            sc0, m0 = get_scale_min_k4(is_idx, scales)
            d1, m1 = d * sc0, dmin * m0
            sc1, m1v = get_scale_min_k4(is_idx + 1, scales)
            d2, m2 = d * sc1, dmin * m1v
            for l in range(32):
                v = d1 * (qs[q_idx + l] & 0xF) - m1
                out[idx] = v if np.isfinite(v) else 0.0; idx += 1
            for l in range(32):
                v = d2 * (qs[q_idx + l] >> 4) - m2
                out[idx] = v if np.isfinite(v) else 0.0; idx += 1
            q_idx += 32; is_idx += 2
    return out

def dequant_q6_k(data, n_elems):
    n_blocks = n_elems // 256
    out = np.empty(n_elems, dtype=np.float32)
    idx = 0
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

def dequant_q8_0(data, n_elems):
    out = np.empty(n_elems, dtype=np.float32)
    for b in range(n_elems // 32):
        off = b * 34
        d = struct.unpack('<e', bytes(data[off:off+2]))[0]
        for j in range(32):
            out[b*32+j] = d * struct.unpack('b', bytes([data[off+2+j]]))[0]
    return out

def load_tensor(reader, name):
    for t in reader.tensors:
        if t.name == name:
            raw = t.data.tobytes()
            n = int(np.prod(t.shape))
            if t.tensor_type == 0: return np.frombuffer(raw, dtype=np.float32)[:n].copy()
            if t.tensor_type == 8: return dequant_q8_0(raw, n)
            if t.tensor_type == 12: return dequant_q4_k(raw, n)
            if t.tensor_type == 14: return dequant_q6_k(raw, n)
            raise ValueError(f"type {t.tensor_type}")
    return None

def rms_norm(x, weight, eps=1e-6):
    return x / np.sqrt(np.mean(x**2) + eps) * weight

def main():
    from gguf import GGUFReader
    
    print("Loading tensors...")
    reader = GGUFReader(MODEL_PATH)
    
    embd = load_tensor(reader, 'token_embd.weight')
    out_norm = load_tensor(reader, 'output_norm.weight')
    l0_q = load_tensor(reader, 'blk.0.attn_q.weight')
    l0_k = load_tensor(reader, 'blk.0.attn_k.weight')
    l0_v = load_tensor(reader, 'blk.0.attn_v.weight')
    l0_wo = load_tensor(reader, 'blk.0.attn_output.weight')
    l0_an = load_tensor(reader, 'blk.0.attn_norm.weight')
    l0_fn = load_tensor(reader, 'blk.0.ffn_norm.weight')
    
    # Embedding
    token = 49
    hidden = embd[token*D_MODEL:(token+1)*D_MODEL] * EMBEDDING_SCALE
    print(f"[EMBD] hidden[0..5] = {hidden[:5]}")
    print(f"[EMBD] norm = {np.linalg.norm(hidden):.4f}")
    
    # Residual
    residual = hidden.copy()
    
    # Attn norm
    hidden = rms_norm(hidden, l0_an)
    print(f"\n[L0 ATTN_NORM] hidden[0..5] = {hidden[:5]}")
    print(f"[L0 ATTN_NORM] norm = {np.linalg.norm(hidden):.4f}")
    
    # Q, K, V projections using column-major layout
    # w has shape [ne0, ne1] stored column-major: w[i + j*ne0]
    # mat_vec: y[j] = sum_i w[i + j*ne0] * x[i] = w[:, j]^T @ x
    # In numpy: reshape to (ne1, ne0) then transpose -> (ne0, ne1), then multiply
    
    def mul_mat(w, x, ne0, ne1):
        W = w.reshape(ne1, ne0).T  # (ne0, ne1)
        return W.T @ x  # (ne1,)
    
    q = mul_mat(l0_q, hidden, D_MODEL, D_MODEL)
    k = mul_mat(l0_k, hidden, D_MODEL, N_KV_HEADS * HEAD_DIM)
    v = mul_mat(l0_v, hidden, D_MODEL, N_KV_HEADS * HEAD_DIM)
    
    print(f"\n[L0 Q] Q[0..5] = {q[:5]}")
    print(f"[L0 Q] Q norm = {np.linalg.norm(q):.4f}")
    print(f"[L0 K] K[0..5] = {k[:5]}")
    print(f"[L0 V] V[0..5] = {v[:5]}")
    print(f"[L0 V] V norm = {np.linalg.norm(v):.4f}")
    
    # RoPE at position 0 (should be identity: cos(0)=1, sin(0)=0)
    for h in range(N_HEADS):
        sl = slice(h*HEAD_DIM, (h+1)*HEAD_DIM)
        for i in range(HEAD_DIM//2):
            theta = 0.0 / (ROPE_THETA ** (2.0 * i / HEAD_DIM))  # = 0
            c, s = np.cos(theta), np.sin(theta)  # cos(0)=1, sin(0)=0
            x0, x1 = q[sl][2*i], q[sl][2*i+1]
            # RoPE should be identity at pos=0!
    
    print(f"\n[L0 Q after RoPE (pos=0)] same as before: {q[:5]}")
    
    # For single token at pos=0: attention is trivial
    # softmax(dot(Q, K) * scale) = softmax of single value = 1.0
    # output = V (grouped by head)
    
    # Group V by head: head h uses KV head h//N_REP
    attn_out = np.zeros(D_MODEL, dtype=np.float32)
    for h in range(N_HEADS):
        kv_h = h // N_REP
        attn_out[h*HEAD_DIM:(h+1)*HEAD_DIM] = v[kv_h*HEAD_DIM:(kv_h+1)*HEAD_DIM]
    
    print(f"\n[L0 ATTN_PRE_PROJ] attn_out[0..5] = {attn_out[:5]}")
    print(f"[L0 ATTN_PRE_PROJ] norm = {np.linalg.norm(attn_out):.4f}")
    
    # Output projection
    attn_out = mul_mat(l0_wo, attn_out, D_MODEL, D_MODEL)
    print(f"\n[L0 ATTN_OUT] attn_out[0..5] = {attn_out[:5]}")
    print(f"[L0 ATTN_OUT] norm = {np.linalg.norm(attn_out):.4f}")
    
    # Residual
    hidden = residual + RESIDUAL_SCALE * attn_out
    print(f"\n[L0 RESIDUAL] hidden[0..5] = {hidden[:5]}")
    print(f"[L0 RESIDUAL] norm = {np.linalg.norm(hidden):.4f}")
    
    print(f"\n{'='*60}")
    print(f"RUST VALUES (from trace_logits_by_layer_count):")
    print(f"  [L0 ATTN_NORM] hidden[0][0..5] = [3.29864, 3.0720913, 0.12317485, 2.0247412, -0.71808314]")
    print(f"  [L0 ATTN_OUT] attn_out[0][0..5] = [-0.65535444, 0.26849353, 0.032885097, -0.26019982, 0.34972763]")
    print(f"  [L0 ATTN_OUT] attn_out[0] norm = 14.4440")
    print(f"  [L0 RESIDUAL] hidden[0][0..5] = [-0.03788525, -0.051653013, -0.03262505, -0.6197096, 0.23637918]")
    print(f"  [L0 RESIDUAL] hidden[0] norm = 9.8722")

if __name__ == '__main__':
    main()
