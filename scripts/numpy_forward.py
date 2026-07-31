#!/usr/bin/env python3
"""Fast forward pass in Python using numpy. Compare hidden states with Rust layer by layer."""
import struct, sys
import numpy as np

MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

D = 1536; NH = 24; NKV = 8; HD = 64; NR = NH // NKV
NE = 40; NEA = 8; DFF = 512; VS = 49155
ROPE_THETA = 1e7; EMB_SCALE = 12.0; RES_SCALE = 0.22; LOG_SCALE = 6.0; ATTN_SCALE = 0.015625

# ── Dequant ──
def f16(h):
    s=(h>>15)&1; e=(h>>10)&0x1F; m=h&0x3FF
    if e==0: return (-0.0 if s else 0.0) if m==0 else ((-1 if s else 1)*2.**-14*(m/1024.))
    if e==31: return (float('-inf') if s else float('inf')) if m==0 else float('nan')
    return (-1 if s else 1)*2.**(e-15)*(1.+m/1024.)

def smk4(j,s):
    if j<4: return s[j]&63, s[j+4]&63
    return ((s[j+4]&0xF)|((s[j-4]>>6)<<4)), ((s[j+4]>>4)|((s[j]>>6)<<4))

def dq_q4k(d,n):
    o=np.empty(n,dtype=np.float32); idx=0
    for b in range(n//256):
        off=b*144; dr=f16(struct.unpack('<H',bytes(d[off:off+2]))[0])
        dm=f16(struct.unpack('<H',bytes(d[off+2:off+4]))[0])
        dr=dr if np.isfinite(dr) else 0.; dm=dm if np.isfinite(dm) else 0.
        sc=d[off+4:off+16]; qs=d[off+16:off+144]; ii=qi=0
        for _ in range(4):
            s0,m0=smk4(ii,sc); s1,m1=smk4(ii+1,sc)
            d1,dm1=dr*s0,dm*m0; d2,dm2=dr*s1,dm*m1
            for l in range(32): v=d1*(qs[qi+l]&0xF)-dm1; o[idx]=v if np.isfinite(v) else 0.; idx+=1
            for l in range(32): v=d2*(qs[qi+l]>>4)-dm2; o[idx]=v if np.isfinite(v) else 0.; idx+=1
            qi+=32; ii+=2
    return o

def dq_q6k(d,n):
    o=np.empty(n,dtype=np.float32); idx=0
    for b in range(n//256):
        off=b*210; ql=d[off:off+128]; qh=d[off+128:off+192]
        sc=[int(x) if x<128 else int(x)-256 for x in d[off+192:off+208]]
        dd=f16(struct.unpack('<H',bytes(d[off+208:off+210]))[0])
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

def dq_q80(d,n):
    o=np.empty(n,dtype=np.float32)
    for b in range(n//32):
        off=b*34; s=struct.unpack('<e',bytes(d[off:off+2]))[0]
        for j in range(32): o[b*32+j]=s*struct.unpack('b',bytes([d[off+2+j]]))[0]
    return o

def load(reader, name):
    for t in reader.tensors:
        if t.name==name:
            r=t.data.tobytes(); n=int(np.prod(t.shape))
            if t.tensor_type==0: return np.frombuffer(r,dtype=np.float32)[:n].copy()
            if t.tensor_type==8: return dq_q80(r,n)
            if t.tensor_type==12: return dq_q4k(r,n)
            if t.tensor_type==14: return dq_q6k(r,n)
    return None

def rms(x,w,e=1e-6): return x/np.sqrt(np.mean(x**2)+e)*w
def silu(x): return x/(1.+np.exp(-x))
def softmax(x): e=np.exp(x-np.max(x)); return e/np.sum(e)

def mul(w,x,ne0,ne1):
    """GGML mul_mat: y[j] = sum_i w[i+j*ne0]*x[i]
    w.reshape(ne1,ne0)[j,i] = w[j*ne0+i] = tensor[i,j]
    So (ne1,ne0) @ (ne0,) = (ne1,) gives y[j] = sum_i tensor[i,j]*x[i]"""
    return (w.reshape(ne1,ne0) @ x).astype(np.float32)

def main():
    from gguf import GGUFReader
    print("Loading...", flush=True)
    reader = GGUFReader(MODEL_PATH)

    embd = load(reader, 'token_embd.weight')
    onorm = load(reader, 'output_norm.weight')

    MAX_LAYERS = 5  # Load 5 layers to find divergence point
    layers = []
    for li in range(MAX_LAYERS):
        sys.stdout.write(f"\r  Layer {li}/31...")
        sys.stdout.flush()
        L = {}
        L['an'] = load(reader, f'blk.{li}.attn_norm.weight')
        L['q'] = load(reader, f'blk.{li}.attn_q.weight')
        L['k'] = load(reader, f'blk.{li}.attn_k.weight')
        L['v'] = load(reader, f'blk.{li}.attn_v.weight')
        L['wo'] = load(reader, f'blk.{li}.attn_output.weight')
        L['fn'] = load(reader, f'blk.{li}.ffn_norm.weight')
        gi = load(reader, f'blk.{li}.ffn_gate_inp.weight')
        ge = load(reader, f'blk.{li}.ffn_gate_exps.weight')
        ue = load(reader, f'blk.{li}.ffn_up_exps.weight')
        de = load(reader, f'blk.{li}.ffn_down_exps.weight')
        L['gi'] = gi.reshape(NE, D)
        L['ge'] = ge.reshape(NE, DFF, D)
        L['ue'] = ue.reshape(NE, DFF, D)
        L['de'] = de.reshape(NE, D, DFF)
        layers.append(L)
    print(" done")

    # Forward pass
    tok = 49
    h = embd[tok*D:(tok+1)*D] * EMB_SCALE
    print(f"[EMBD] h[0..5]={h[:5]} norm={np.linalg.norm(h):.4f}")

    # Rust reference values for comparison
    rust = {
        0: {'final': np.array([0.18452999, -0.17343995, -0.03152647, -1.3824012, 0.022338182])},
        1: {'final': np.array([0.11106729, -0.16368194, -0.094443075, -1.5446923, 0.090278156])},
    }

    for li in range(MAX_LAYERS):
        L = layers[li]
        res = h.copy()

        # Attention norm
        h = rms(h, L['an'])

        # Q, K, V projections
        q = mul(L['q'], h, D, D)
        k = mul(L['k'], h, D, NKV*HD)
        v = mul(L['v'], h, D, NKV*HD)

        # RoPE at pos=0 (identity)

        # Single token: V grouped by head
        attn = np.zeros(D, dtype=np.float32)
        for hd in range(NH):
            kv_h = hd // NR
            attn[hd*HD:(hd+1)*HD] = v[kv_h*HD:(kv_h+1)*HD]

        # Output projection
        attn = mul(L['wo'], attn, D, D)

        # Residual
        h = res + RES_SCALE * attn

        # FFN
        res2 = h.copy()
        h = rms(h, L['fn'])

        # Routing
        scores = L['gi'] @ h
        probs = softmax(scores)
        top_idx = np.argsort(probs)[-NEA:][::-1]
        nscores = probs[top_idx] / np.sum(probs[top_idx])

        # Expert FFN
        ffn = np.zeros(D, dtype=np.float32)
        for ki, e in enumerate(top_idx):
            s = nscores[ki]
            gate = L['ge'][e] @ h
            up = L['ue'][e] @ h
            fused = silu(gate) * up
            ffn += s * (L['de'][e] @ fused)

        h = res2 + RES_SCALE * ffn

        # Compare with Rust
        rust_norms = {0: 15.8893, 1: 16.6882, 2: 16.9603, 3: 17.1199, 4: 17.1344}
        if li in rust:
            rv = rust[li]['final']
            match = np.allclose(h[:5], rv, atol=1e-4)
            diff = np.max(np.abs(h[:5] - rv))
            print(f"[L{li}] h[0..5]={h[:5]}")
            print(f"  RUST={rv}")
            print(f"  match={match} max_diff={diff:.2e}")
        else:
            rn = rust_norms.get(li)
            norm_match = f"rust_norm={rn}" if rn else ""
            print(f"[L{li}] h[0..5]={h[:5]} norm={np.linalg.norm(h):.4f} {norm_match}")

    # Output norm + logits
    hn = rms(h, onorm)
    print(f"\n[OUTPUT_NORM] hn[0..5]={hn[:5]}")
    print(f"[OUTPUT_NORM] norm={np.linalg.norm(hn):.4f}")

    # Compare with Rust output norm
    rust_normed = np.array([-13.39707, -29.21777, -20.49918, -17.117159, -3.0791874])
    print(f"  RUST={rust_normed}")
    print(f"  max_diff={np.max(np.abs(hn[:5] - rust_normed)):.2e}")

    # Compute logits
    logits = np.zeros(VS, dtype=np.float32)
    for v in range(VS):
        logits[v] = np.dot(embd[v*D:(v+1)*D], hn) / LOG_SCALE

    top5 = np.argsort(logits)[-5:][::-1]
    print(f"\n[LOGITS] mean={np.mean(logits):.4f} max={np.max(logits):.2f}")
    print(f"[LOGITS] top5={list(top5)} vals={[f'{logits[i]:.2f}' for i in top5]}")

    # Compare with reference
    from llama_cpp import Llama
    llm = Llama(model_path=MODEL_PATH, n_ctx=1, n_threads=1, logits_all=True, verbose=False)
    llm.reset(); llm.eval([49])
    ref = np.array(llm.scores[0])
    corr = np.corrcoef(ref, logits)[0, 1]
    print(f"\n[REF] mean={np.mean(ref):.4f} max={np.max(ref):.2f}")
    print(f"[REF] top5={np.argsort(ref)[-5:][::-1].tolist()}")
    print(f"\nCorrelation: {corr:.4f}")

if __name__ == '__main__':
    main()
