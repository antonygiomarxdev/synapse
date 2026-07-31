#!/usr/bin/env python3
"""Python implementation of the forward pass for layer-by-layer comparison with Rust."""
import struct
import numpy as np

MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

D_MODEL = 1536
N_HEADS = 24
N_KV_HEADS = 8
HEAD_DIM = 64
N_REP = N_HEADS // N_KV_HEADS  # 3
N_LAYERS = 32
N_EXPERTS = 40
N_EXPERTS_ACTIVE = 8
D_FF = 512
VOCAB_SIZE = 49155
ROPE_THETA = 1e7
EMBEDDING_SCALE = 12.0
RESIDUAL_SCALE = 0.22
LOGIT_SCALE = 6.0
ATTENTION_SCALE = 0.015625

# ── Dequantization ────────────────────────────────────────────────────

def f16_to_f32(h):
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

def get_scale_min_k4(j, scales):
    if j < 4:
        d = scales[j] & 63
        m = scales[j + 4] & 63
    else:
        d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return d, m

def dequant_q4_k(data, n_elems):
    """Dequantize Q4_K: 256 elems/block, 144 bytes/block."""
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
        
        is_idx = 0
        q_idx = 0
        for _ in range(4):
            sc0, m0 = get_scale_min_k4(is_idx, scales)
            d1 = d * sc0
            m1 = dmin * m0
            sc1, m1v = get_scale_min_k4(is_idx + 1, scales)
            d2 = d * sc1
            m2 = dmin * m1v
            
            for l in range(32):
                v = d1 * (qs[q_idx + l] & 0xF) - m1
                out[idx] = v if np.isfinite(v) else 0.0
                idx += 1
            for l in range(32):
                v = d2 * (qs[q_idx + l] >> 4) - m2
                out[idx] = v if np.isfinite(v) else 0.0
                idx += 1
            q_idx += 32
            is_idx += 2
    return out

def dequant_q6_k(data, n_elems):
    """Dequantize Q6_K: 256 elems/block, 210 bytes/block."""
    n_blocks = n_elems // 256
    out = np.empty(n_elems, dtype=np.float32)
    idx = 0
    for b in range(n_blocks):
        off = b * 210
        ql = data[off:off+128]
        qh = data[off+128:off+192]
        sc_raw = data[off+192:off+208]
        sc = [int(x) if x < 128 else int(x) - 256 for x in sc_raw]  # signed i8
        d = f16_to_f32(struct.unpack('<H', bytes(data[off+208:off+210]))[0])
        
        # Two groups of 128 elements each
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
    n_blocks = n_elems // 32
    out = np.empty(n_elems, dtype=np.float32)
    for b in range(n_blocks):
        off = b * 34
        d = struct.unpack('<e', bytes(data[off:off+2]))[0]
        for j in range(32):
            q = struct.unpack('b', bytes([data[off+2+j]]))[0]
            out[b * 32 + j] = d * q
    return out

# ── Tensor loading ────────────────────────────────────────────────────

def load_tensor(reader, name):
    tensor = None
    for t in reader.tensors:
        if t.name == name:
            tensor = t
            break
    if tensor is None:
        return None
    
    raw_data = tensor.data.tobytes()
    n_elems = int(np.prod(tensor.shape))
    
    type_map = {0: 'F32', 1: 'F16', 8: 'Q8_0', 12: 'Q4_K', 14: 'Q6_K'}
    ttype = tensor.tensor_type
    
    if ttype == 0:  # F32
        return np.frombuffer(raw_data, dtype=np.float32)[:n_elems].copy()
    elif ttype == 8:  # Q8_0
        return dequant_q8_0(raw_data, n_elems)
    elif ttype == 12:  # Q4_K
        return dequant_q4_k(raw_data, n_elems)
    elif ttype == 14:  # Q6_K
        return dequant_q6_k(raw_data, n_elems)
    else:
        raise ValueError(f"Unsupported tensor type: {ttype} ({type_map.get(ttype, '?')})")

# ── Operations ────────────────────────────────────────────────────────

def rms_norm(x, weight, eps=1e-6):
    ss = np.sum(x**2)
    rms = np.sqrt(ss / len(x) + eps)
    return x / rms * weight

def rope_inplace(x, pos, theta_base):
    d = len(x)
    for i in range(d // 2):
        theta = pos / (theta_base ** (2.0 * i / d))
        cos_val = np.cos(theta)
        sin_val = np.sin(theta)
        x0, x1 = float(x[2*i]), float(x[2*i+1])
        x[2*i] = x0 * cos_val - x1 * sin_val
        x[2*i+1] = x0 * sin_val + x1 * cos_val

def mat_vec_transposed(w, x, ne0, ne1):
    """GGML mul_mat: y[j] = sum_i w[i + j*ne0] * x[i], column-major."""
    # w is flat, shape [ne0, ne1] column-major
    W = w.reshape(ne1, ne0).T  # [ne0, ne1]
    return W.T @ x  # [ne1]

def softmax(x):
    max_val = np.max(x)
    exp_x = np.exp(x - max_val)
    return exp_x / np.sum(exp_x)

def silu(x):
    return x / (1.0 + np.exp(-x))

# ── Forward pass ──────────────────────────────────────────────────────

def forward_one_layer(hidden_in, layer_tensors, debug=True, layer_idx=0):
    """Run one transformer layer. Returns hidden state after the layer."""
    d_model = D_MODEL
    n_tokens = 1
    
    attn_norm = layer_tensors['attn_norm']
    attn_q = layer_tensors['attn_q']
    attn_k = layer_tensors['attn_k']
    attn_v = layer_tensors['attn_v']
    attn_output = layer_tensors['attn_output']
    ffn_norm = layer_tensors['ffn_norm']
    gate_inp = layer_tensors['gate_inp']
    gate_exps = layer_tensors['gate_exps']
    up_exps = layer_tensors['up_exps']
    down_exps = layer_tensors['down_exps']
    
    hidden = hidden_in.copy()
    
    # ── Attention ──
    residual = hidden.copy()
    hidden = rms_norm(hidden, attn_norm)
    if debug:
        print(f"  [L{layer_idx} ATTN_NORM] hidden[0..5] = {hidden[:5]}")
        print(f"  [L{layer_idx} ATTN_NORM] hidden norm = {np.linalg.norm(hidden):.4f}")
    
    # Q, K, V projections (for single token)
    q = mat_vec_transposed(attn_q, hidden, d_model, d_model)
    k = mat_vec_transposed(attn_k, hidden, d_model, N_KV_HEADS * HEAD_DIM)
    v = mat_vec_transposed(attn_v, hidden, d_model, N_KV_HEADS * HEAD_DIM)
    
    if debug:
        print(f"  [L{layer_idx} Q] Q[0..5] = {q[:5]}")
        print(f"  [L{layer_idx} Q] Q norm = {np.linalg.norm(q):.4f}")
        print(f"  [L{layer_idx} K] K[0..5] = {k[:5]}")
        print(f"  [L{layer_idx} V] V[0..5] = {v[:5]}")
        print(f"  [L{layer_idx} V] V norm = {np.linalg.norm(v):.4f}")
    
    # RoPE (for single token at position 0 — should be identity!)
    for h in range(N_HEADS):
        rope_inplace(q[h*HEAD_DIM:(h+1)*HEAD_DIM], 0.0, ROPE_THETA)
    for h in range(N_KV_HEADS):
        rope_inplace(k[h*HEAD_DIM:(h+1)*HEAD_DIM], 0.0, ROPE_THETA)
    
    if debug:
        print(f"  [L{layer_idx} Q after RoPE] Q[0..5] = {q[:5]}")
        print(f"  [L{layer_idx} K after RoPE] K[0..5] = {k[:5]}")
    
    # For single token: attention = V values grouped by head, then output proj
    attn_out = np.zeros(d_model, dtype=np.float32)
    for h in range(N_HEADS):
        kv_h = h // N_REP
        for d in range(HEAD_DIM):
            attn_out[h * HEAD_DIM + d] = v[kv_h * HEAD_DIM + d]
    
    if debug:
        print(f"  [L{layer_idx} ATTN_PRE_PROJ] attn_out[0..5] = {attn_out[:5]}")
        print(f"  [L{layer_idx} ATTN_PRE_PROJ] norm = {np.linalg.norm(attn_out):.4f}")
    
    # Output projection
    attn_out = mat_vec_transposed(attn_output, attn_out, d_model, d_model)
    
    if debug:
        print(f"  [L{layer_idx} ATTN_OUT] attn_out[0..5] = {attn_out[:5]}")
        print(f"  [L{layer_idx} ATTN_OUT] norm = {np.linalg.norm(attn_out):.4f}")
    
    # Residual
    hidden = residual + RESIDUAL_SCALE * attn_out
    if debug:
        print(f"  [L{layer_idx} RESIDUAL] hidden[0..5] = {hidden[:5]}")
        print(f"  [L{layer_idx} RESIDUAL] norm = {np.linalg.norm(hidden):.4f}")
    
    # ── FFN ──
    residual2 = hidden.copy()
    hidden = rms_norm(hidden, ffn_norm)
    if debug:
        print(f"  [L{layer_idx} FFN_NORM] hidden[0..5] = {hidden[:5]}")
        print(f"  [L{layer_idx} FFN_NORM] norm = {np.linalg.norm(hidden):.4f}")
    
    # Routing
    n_experts = gate_inp.shape[0] // d_model
    scores = np.zeros(n_experts, dtype=np.float32)
    for e in range(n_experts):
        scores[e] = np.dot(hidden, gate_inp[e*d_model:(e+1)*d_model])
    
    all_probs = softmax(scores)
    top_k_idx = np.argsort(all_probs)[-N_EXPERTS_ACTIVE:][::-1]
    top_k_scores = all_probs[top_k_idx]
    
    if debug:
        print(f"  [L{layer_idx} ROUTE] experts={list(top_k_idx)}")
        print(f"  [L{layer_idx} ROUTE] scores={[f'{s:.4f}' for s in top_k_scores]}")
    
    # Expert FFN
    ffn_out = np.zeros(d_model, dtype=np.float32)
    norm_scores = top_k_scores / np.sum(top_k_scores)
    
    for ki, e in enumerate(top_k_idx):
        score = norm_scores[ki]
        
        # gate_proj: hidden @ gate_exps[e]^T -> [d_ff]
        # gate_exps layout: data[e * d_ff * d_model + j * d_model + d]
        gate_w = gate_exps[e*D_FF*D_MODEL:(e+1)*D_FF*D_MODEL]
        gate_out = np.zeros(D_FF, dtype=np.float32)
        for j in range(D_FF):
            gate_out[j] = np.dot(hidden, gate_w[j*d_model:(j+1)*d_model])
        
        # up_proj
        up_w = up_exps[e*D_FF*D_MODEL:(e+1)*D_FF*D_MODEL]
        up_out = np.zeros(D_FF, dtype=np.float32)
        for j in range(D_FF):
            up_out[j] = np.dot(hidden, up_w[j*d_model:(j+1)*d_model])
        
        # SiLU(gate) * up
        fused = silu(gate_out) * up_out
        
        # down_proj
        # down_exps layout: data[e * d_model * d_ff + d * d_ff + j]
        down_w = down_exps[e*d_model*D_FF:(e+1)*d_model*D_FF]
        for d in range(d_model):
            acc = 0.0
            for j in range(D_FF):
                acc += fused[j] * down_w[d*D_FF+j]
            ffn_out[d] += score * acc
    
    if debug:
        print(f"  [L{layer_idx} FFN_OUT] ffn_out[0..5] = {ffn_out[:5]}")
        print(f"  [L{layer_idx} FFN_OUT] norm = {np.linalg.norm(ffn_out):.4f}")
    
    # Residual 2
    hidden = residual2 + RESIDUAL_SCALE * ffn_out
    if debug:
        print(f"  [L{layer_idx} FINAL] hidden[0..5] = {hidden[:5]}")
        print(f"  [L{layer_idx} FINAL] norm = {np.linalg.norm(hidden):.4f}")
    
    return hidden

def main():
    from gguf import GGUFReader
    
    print("Loading GGUF file...")
    reader = GGUFReader(MODEL_PATH)
    
    # Load embedding
    print("Loading embedding...")
    embd_weight = load_tensor(reader, 'token_embd.weight')
    output_norm = load_tensor(reader, 'output_norm.weight')
    
    # Load layer 0 weights
    print("Loading layer 0 weights...")
    layer_tensors = {}
    for suffix in ['attn_norm', 'attn_q', 'attn_k', 'attn_v', 'attn_output',
                   'ffn_norm', 'ffn_gate_inp', 'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps']:
        short = suffix.replace('ffn_', '').replace('attn_', '')
        if suffix == 'ffn_gate_inp':
            short = 'gate_inp'
        elif suffix == 'ffn_gate_exps':
            short = 'gate_exps'
        elif suffix == 'ffn_up_exps':
            short = 'up_exps'
        elif suffix == 'ffn_down_exps':
            short = 'down_exps'
        layer_tensors[short] = load_tensor(reader, f'blk.0.{suffix}.weight')
    
    # Single token: 49
    token = 49
    
    # Phase 0: Embedding
    hidden = embd_weight[token * D_MODEL:(token + 1) * D_MODEL] * EMBEDDING_SCALE
    print(f"\n[EMBD] hidden[0..5] = {hidden[:5]}")
    print(f"[EMBD] norm = {np.linalg.norm(hidden):.4f}")
    
    # Layer 0
    print(f"\n=== Layer 0 ===")
    hidden = forward_one_layer(hidden, layer_tensors, debug=True, layer_idx=0)
    
    # Output norm + logits
    normed = rms_norm(hidden, output_norm)
    print(f"\n[OUTPUT_NORM] normed[0..5] = {normed[:5]}")
    print(f"[OUTPUT_NORM] norm = {np.linalg.norm(normed):.4f}")
    
    # Logits (embedding as output projection)
    logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
    for v in range(min(1000, VOCAB_SIZE)):  # Only first 1000 for speed
        logits[v] = np.dot(embd_weight[v*D_MODEL:(v+1)*D_MODEL], normed) / LOGIT_SCALE
    
    print(f"\nFirst 1000 logits:")
    print(f"  mean = {np.mean(logits[:1000]):.4f}")
    print(f"  max = {np.max(logits[:1000]):.2f}")
    top5 = np.argsort(logits[:1000])[-5:][::-1]
    print(f"  top-5 (in first 1000) = {list(top5)}")
    print(f"  top-5 vals = {[f'{logits[i]:.2f}' for i in top5]}")

if __name__ == '__main__':
    main()
