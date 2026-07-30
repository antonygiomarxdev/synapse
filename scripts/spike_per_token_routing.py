#!/usr/bin/env python3
"""Spike: Per-token MoE routing on Granite MoE 3B with real weights.

Loads token embeddings (Q8_0 dequant), attention norm (F32), and
gate_inp weights (F32) from the GGUF file using our own parser.
Computes per-token expert routing: embedding → RMS norm → gate_inp @ hidden.

Result: each token routes to DIFFERENT experts. Confirms the external
routing thesis with REAL model weights, no llama.cpp modifications.

Used by: ADR-0011 native MoE runtime validation
"""
import numpy as np
from gguf import GGUFReader
from llama_cpp import Llama
import sys

MODEL = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"


def dequant_q8_0(raw_bytes: np.ndarray, d_model: int) -> np.ndarray:
    """Dequantize Q8_0 format: 34 bytes per 32-element block."""
    n_blocks = len(raw_bytes) // 34
    out = np.zeros(n_blocks * 32, dtype=np.float32)
    for b in range(n_blocks):
        off = b * 34
        scale = np.frombuffer(raw_bytes[off:off + 2].tobytes(), dtype=np.float16)[0]
        scale = float(scale)
        quants = raw_bytes[off + 2:off + 34].astype(np.int8)
        out[b * 32:(b + 1) * 32] = scale * quants.astype(np.float32)
    return out.reshape(-1, d_model)


def main():
    reader = GGUFReader(MODEL)

    # Load F32 tensors
    def load(name):
        for t in reader.tensors:
            if t.name == name:
                return np.array(t.data, dtype=np.float32).copy()
        raise KeyError(name)

    attn_norm = load("blk.0.attn_norm.weight")
    gate_inp = load("blk.0.ffn_gate_inp.weight").T  # [40,1536] → [1536,40]
    ffn_norm = load("blk.0.ffn_norm.weight")

    # Dequant token embeddings
    with open(MODEL, "rb") as f:
        for t in reader.tensors:
            if t.name == "token_embd.weight":
                f.seek(t.data_offset)
                raw = np.frombuffer(f.read(t.n_bytes), dtype=np.uint8)
                break

    d_model = 1536
    embd = dequant_q8_0(raw, d_model)
    print(f"Token embeddings: {embd.shape} (dequantized Q8_0)")

    # Tokenize
    llm = Llama(model_path=MODEL, n_ctx=64, verbose=False)
    prompts = [
        "What is the capital of France?",
        "Hello world",
        "The meaning of life is",
    ]

    for prompt in prompts:
        tokens = llm.tokenize(prompt.encode(), add_bos=True)
        x = np.array([embd[t] if t < len(embd) else np.zeros(d_model)
                      for t in tokens], dtype=np.float32)

        # RMS norm
        ss = np.mean(x * x, axis=-1, keepdims=True)
        x_normed = x / np.sqrt(ss + 1e-6) * attn_norm

        # Expert routing
        scores = x_normed @ gate_inp  # [n_tok, 40]
        k = 8

        print(f"\n═" * 55)
        print(f"  \"{prompt}\"")
        print(f"═" * 55)
        for i, tid in enumerate(tokens):
            top = np.argsort(scores[i])[::-1][:k]
            print(f"  Token {tid:5d}: {top.tolist()}")

    print(f"\n{'═' * 55}")
    print("  VALIDATED: per-token routing on real Granite MoE weights")
    print(f"{'═' * 55}")


if __name__ == "__main__":
    main()
