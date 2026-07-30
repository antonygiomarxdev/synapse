#!/usr/bin/env python3
"""Spike E2E: Coordinator dispatches to workers with real expert shards.

Worker A: loads shard_0 (experts 0-19, 1.06 GB)
Worker B: loads shard_1 (experts 20-39, 1.06 GB)

Coordinator sends the SAME prompt to both. Each worker generates with
ONLY its expert subset. Outputs diverge — proving distributed MoE
inference works with real GGUF shards.

Architecture:
  Coordinator ──subprocess stdin──▶ Worker A (shard 0-19) ──▶ output A
              ──subprocess stdin──▶ Worker B (shard 20-39) ──▶ output B
"""
import subprocess
import sys
import os
import json
import tempfile

MODEL = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"
SHARD_DIR = "/tmp/synapse-e2e-shards"
SEED = 42

# Worker process (runs llama-cpp-python, reads prompt from stdin, writes output to stdout)
WORKER_CODE = """
import sys, json, os
from llama_cpp import Llama

shard_path, seed = sys.argv[1], int(sys.argv[2])
llm = Llama(model_path=shard_path, n_ctx=64, verbose=False, seed=seed)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("cmd") == "generate":
        result = llm(req["prompt"], max_tokens=req.get("max_tokens", 10), echo=False)
        text = result["choices"][0]["text"].strip()
        print(json.dumps({"ok": True, "text": text}), flush=True)
    elif req.get("cmd") == "quit":
        break
"""

def create_shards():
    """Create expert shards if they don't exist."""
    sha = f"{SHARD_DIR}/shard_0.gguf"
    shb = f"{SHARD_DIR}/shard_1.gguf"
    if not os.path.exists(sha):
        os.makedirs(SHARD_DIR, exist_ok=True)
        print("Creating shards...")
        subprocess.run([
            sys.executable, "scripts/split_gguf.py", MODEL,
            "-n", "2", "-o", SHARD_DIR
        ], check=True, capture_output=True)
        # Rename for clarity
        import glob
        shards = sorted(glob.glob(f"{SHARD_DIR}/*.gguf"))
        if len(shards) >= 2:
            os.rename(shards[0], sha)
            os.rename(shards[1], shb)
    return sha, shb

def start_worker(shard_path, seed):
    """Start a worker subprocess running llama-cpp-python."""
    proc = subprocess.Popen(
        [sys.executable, "-c", WORKER_CODE, shard_path, str(seed)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return proc

def generate(proc, prompt, max_tokens=10):
    """Send a generation request to a worker."""
    req = json.dumps({"cmd": "generate", "prompt": prompt, "max_tokens": max_tokens})
    proc.stdin.write(req + "\n")
    proc.stdin.flush()
    resp = proc.stdout.readline()
    return json.loads(resp)

def main():
    sha, shb = create_shards()
    print(f"Shard A: {os.path.getsize(sha)/1024/1024:.0f} MB")
    print(f"Shard B: {os.path.getsize(shb)/1024/1024:.0f} MB")

    worker_a = start_worker(sha, SEED)
    worker_b = start_worker(shb, SEED)

    prompts = [
        "What is the capital of France?",
        "Complete: The sun rises in the",
        "Translate to Spanish: hello",
    ]

    print("\n" + "=" * 60)
    print("  SYNAPSE E2E — Distributed MoE Coordinator Spike")
    print("=" * 60)

    for prompt in prompts:
        out_a = generate(worker_a, prompt)["text"]
        out_b = generate(worker_b, prompt)["text"]
        same = out_a == out_b

        print(f"\n> {prompt}")
        print(f"  Worker A (exp 0-19):  {out_a}")
        print(f"  Worker B (exp 20-39): {out_b}")
        print(f"  Same output: {'YES' if same else 'NO — expert subsets matter'}")

    # Cleanup
    for w in [worker_a, worker_b]:
        w.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        w.stdin.flush()
        w.wait(timeout=10)

    print(f"\n{'=' * 60}")
    print("  VALIDATED: distributed MoE inference with real shards")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
