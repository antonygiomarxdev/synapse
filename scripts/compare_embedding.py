#!/usr/bin/env python3
"""Compare our dequantized embedding with llama-cpp-python's internal embedding."""
import struct
import numpy as np

MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

def f16_to_f32(h):
    s=(h>>15)&1; e=(h>>10)&0x1F; m=h&0x3FF
    if e==0: return (-0.0 if s else 0.0) if m==0 else ((-1 if s else 1)*2.**-14*(m/1024.))
    if e==31: return (float('-inf') if s else float('inf')) if m==0 else float('nan')
    return (-1 if s else 1)*2.**(e-15)*(1.+m/1024.)

def dequant_q8_0(data, n):
    out = np.empty(n, dtype=np.float32)
    for b in range(n // 32):
        off = b * 34
        d = struct.unpack('<e', bytes(data[off:off+2]))[0]
        for j in range(32):
            out[b*32+j] = d * struct.unpack('b', bytes([data[off+2+j]]))[0]
    return out

def main():
    from gguf import GGUFReader
    
    reader = GGUFReader(MODEL_PATH)
    
    # Get embedding tensor
    embd_tensor = None
    for t in reader.tensors:
        if t.name == 'token_embd.weight':
            embd_tensor = t
            break
    
    print(f"Embedding: type={embd_tensor.tensor_type}, shape={list(embd_tensor.shape)}")
    
    # Dequantize our way (Q8_0)
    raw = embd_tensor.data.tobytes()
    n_elems = int(np.prod(embd_tensor.shape))
    our_embd = dequant_q8_0(raw, n_elems)
    
    # Reshape to [vocab, d_model]
    d_model = 1536
    vocab = n_elems // d_model
    our_embd_2d = our_embd.reshape(vocab, d_model)
    
    # Get embedding for token 49
    our_tok49 = our_embd_2d[49] * 12.0  # embedding_scale
    print(f"\nOur embedding for token 49:")
    print(f"  [0..5] = {our_tok49[:5]}")
    print(f"  norm = {np.linalg.norm(our_tok49):.4f}")
    
    # Now get llama-cpp-python's embedding
    from llama_cpp import Llama
    llm = Llama(model_path=MODEL_PATH, n_ctx=1, n_threads=1, 
                embedding=True, verbose=False)
    
    # Get embedding vector
    llama_embd = np.array(llm.embed([49]))
    print(f"\nllama-cpp embedding for token 49:")
    print(f"  [0..5] = {llama_embd[:5]}")
    print(f"  norm = {np.linalg.norm(llama_embd):.4f}")
    
    # Compare
    print(f"\n=== Comparison ===")
    print(f"  Shape: ours={our_tok49.shape}, llama={llama_embd.shape}")
    
    if our_tok49.shape == llama_embd.shape:
        corr = np.corrcoef(our_tok49, llama_embd)[0, 1]
        cos = np.dot(our_tok49, llama_embd) / (np.linalg.norm(our_tok49) * np.linalg.norm(llama_embd))
        max_diff = np.max(np.abs(our_tok49 - llama_embd))
        print(f"  Correlation: {corr:.6f}")
        print(f"  Cosine sim: {cos:.6f}")
        print(f"  Max diff: {max_diff:.2e}")
        
        # Check if they're scaled versions
        ratio = llama_embd / (our_tok49 + 1e-10)
        valid = np.abs(our_tok49) > 0.01
        if np.any(valid):
            print(f"  Mean ratio (where |ours|>0.01): {np.mean(ratio[valid]):.6f}")
            print(f"  Std ratio: {np.std(ratio[valid]):.6f}")
    else:
        print("  Shape mismatch!")
        # Try without embedding_scale
        our_tok49_noscale = our_embd_2d[49]
        print(f"\n  Our embedding (no scale) [0..5] = {our_tok49_noscale[:5]}")
        print(f"  llama-cpp embedding [0..5] = {llama_embd[:5]}")
        
        # Check if llama applies scale differently
        if llama_embd.shape[0] == our_tok49_noscale.shape[0]:
            corr = np.corrcoef(our_tok49_noscale, llama_embd)[0, 1]
            print(f"  Correlation (no scale): {corr:.6f}")

if __name__ == '__main__':
    main()
