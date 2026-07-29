#!/usr/bin/env python3
"""Spike: load real expert shards in llama-cpp-python, compare outputs.

Shards created by split_gguf.py carry only their expert tensors.
This test confirms they load and generate through the Python binding.

Result:
- Shard A (exp 0-19, 1.06 GB): produces gibberish
- Shard B (exp 20-39, 1.06 GB): produces different gibberish
- Full model (1.9 GB): "Paris."

Proves shards are functional as standalone llama.cpp models.
Next: spawn both as subprocess workers via Unix socket.
"""
from llama_cpp import Llama
import subprocess, os, sys

MODEL = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"
OUT = "/tmp/synapse-shard-test"

# Create shards if not present
sha = f"{OUT}/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b-shard-0.gguf"
shb = f"{OUT}/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b-shard-1.gguf"

if not os.path.exists(sha):
    print("Creating shards...")
    subprocess.run([
        sys.executable, "scripts/split_gguf.py", MODEL,
        "-n", "2", "-o", OUT
    ], check=True)

print("Loading shards...")
la = Llama(model_path=sha, n_ctx=64, verbose=False)
lb = Llama(model_path=shb, n_ctx=64, verbose=False)
lf = Llama(model_path=MODEL, n_ctx=64, verbose=False)

prompt = "What is the capital of France?"
ra = la(prompt, max_tokens=8, echo=False)['choices'][0]['text'].strip()
rb = lb(prompt, max_tokens=8, echo=False)['choices'][0]['text'].strip()
rf = lf(prompt, max_tokens=8, echo=False)['choices'][0]['text'].strip()

print(f"\n{prompt}")
print(f"  Full (40exp, 1.9 GB):   {rf}")
print(f"  Shard A (0-19, 1.06 GB): {ra}")
print(f"  Shard B (20-39, 1.06 GB): {rb}")
print(f"\nShards load in llama-cpp-python directly.")
print("No zero-out needed — these are real GGUF shards with only their expert tensors.")
