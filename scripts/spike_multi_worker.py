#!/usr/bin/env python3
"""Spike: 2-worker distributed MoE with zero-out sharding.

Worker A: experts 0-19 active, 20-39 zeroed
Worker B: experts 20-39 active, 0-19 zeroed

Result: Workers produce completely different outputs for every prompt.
This proves expert specialisation AND confirms that raw text combination
is insufficient — we need per-token expert routing inside the model's
forward pass, not post-hoc text merging.
"""
from llama_cpp import Llama
from gguf import GGUFReader
import numpy as np, tempfile, os

MODEL = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"
reader = GGUFReader(MODEL)
with open(MODEL, "rb") as f: raw = bytearray(f.read())

def make_shard(start_exp, end_exp):
    buf = bytearray(raw)
    for t in reader.tensors:
        if not any(k in t.name for k in
            ["ffn_gate_exps","ffn_gate_inp","ffn_up_exps","ffn_down_exps"]):
            continue
        data = np.array(t.data)
        data[:start_exp] = 0
        data[end_exp:] = 0
        new_bytes = np.ascontiguousarray(data).tobytes()
        buf[t.data_offset:t.data_offset+len(new_bytes)] = new_bytes
    tmp = tempfile.NamedTemporaryFile(suffix='.gguf', delete=False)
    tmp.write(buf); tmp.close()
    return tmp.name

print("Building shards...")
path_a = make_shard(0, 20)   # experts 0-19
path_b = make_shard(20, 40)  # experts 20-39

print("Loading workers...")
lla = Llama(model_path=path_a, n_ctx=128, verbose=False, seed=42)
llb = Llama(model_path=path_b, n_ctx=128, verbose=False, seed=42)
llf = Llama(model_path=MODEL, n_ctx=128, verbose=False, seed=42)

prompts = [
    "What is the capital of France?",
    "Complete: The sun rises in the",
    "Translate to Spanish: hello",
    "2 + 2 =",
]

for prompt in prompts:
    ra = lla(prompt, max_tokens=12, echo=False)['choices'][0]['text'].strip()
    rb = llb(prompt, max_tokens=12, echo=False)['choices'][0]['text'].strip()
    rf = llf(prompt, max_tokens=12, echo=False)['choices'][0]['text'].strip()
    print(f"\n> {prompt}")
    print(f"  Full (40exp):    {rf}")
    print(f"  Worker A (0-19):  {ra}")
    print(f"  Worker B (20-39): {rb}")

os.unlink(path_a); os.unlink(path_b)
