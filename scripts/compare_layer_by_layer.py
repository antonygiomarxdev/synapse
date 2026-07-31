#!/usr/bin/env python3
"""Compare hidden state layer by layer between llama-cpp-python and our Rust runtime."""
import struct, math
import numpy as np
from gguf import GGUFReader

MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

def dequant_q8_0(data_row, n=1536):
    vals = []
    for block in range(n // 32):
        offset = block * 34
        d = struct.unpack('<e', bytes(data_row[offset:offset+2]))[0]
        for j in range(32):
            q = struct.unpack('b', bytes([data_row[offset+2+j]]))[0]
            vals.append(d * q)
    return np.array(vals[:n])

def dequant_f32(data, n=1536):
    return np.array(data.flat[:n], dtype=np.float32)

def rms_norm(x, weight, eps=1e-6):
    mean_sq = np.mean(x**2)
    inv_rms = 1.0 / math.sqrt(mean_sq + eps)
    return x * inv_rms * weight

def main():
    reader = GGUFReader(MODEL_PATH)
    
    # Load all tensors
    tensors = {}
    for t in reader.tensors:
        tensors[t.name] = t
    
    # Dequantize embedding for token 49
    embd_49 = dequant_q8_0(tensors['token_embd.weight'].data[49])
    hidden = embd_49 * 12.0  # embedding_scale
    
    print(f"Layer 0 (embedding): norm={np.linalg.norm(hidden):.4f}")
    
    # Process first layer
    layer_idx = 0
    print(f"\n=== Layer {layer_idx} ===")
    
    # Save residual
    residual = hidden.copy()
    
    # attn_norm
    attn_norm_w = dequant_f32(tensors[f'blk.{layer_idx}.attn_norm.weight'].data)
    hidden = rms_norm(hidden, attn_norm_w)
    print(f"After attn_norm: norm={np.linalg.norm(hidden):.4f}")
    print(f"First 5: {hidden[:5]}")
    
    # For single token: attn_out = V = W_v^T @ hidden
    # W_v shape: [1536, 512], ne0=1536, ne1=512
    # Need to dequantize W_v (Q6_K)
    # For now, skip and use residual directly
    hidden = residual.copy()  # Skip attention
    
    # FFN norm
    ffn_norm_w = dequant_f32(tensors[f'blk.{layer_idx}.ffn_norm.weight'].data)
    hidden_ffn_norm = rms_norm(hidden, ffn_norm_w)
    print(f"\nAfter FFN norm (no attn): norm={np.linalg.norm(hidden_ffn_norm):.4f}")
    print(f"First 5: {hidden_ffn_norm[:5]}")
    
    # Our Rust says: [L0 FFN_NORM] hidden[0][0..5] = [-0.36569044, -0.7210174, -0.25504905, -2.2798188, 1.041973]
    print(f"\nRust (with attn): [-0.36569044, -0.7210174, -0.25504905, -2.2798188, 1.041973]")

if __name__ == '__main__':
    main()
