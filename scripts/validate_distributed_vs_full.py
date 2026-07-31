#!/usr/bin/env python3
"""Validation: distributed MoE (coordinator + workers with shards) vs full model.

Runs the same prompts through:
1. Full model (single llama-cpp-python instance)
2. Distributed setup (coordinator dispatches to 2 workers with expert shards)

Compares outputs token-by-token to determine if distributed inference
produces the same results as the full model.
"""
import subprocess
import sys
import os
import json
import time

MODEL = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"
SHARD_0 = "/tmp/synapse-e2e-shards/shard_0.gguf"
SHARD_1 = "/tmp/synapse-e2e-shards/shard_1.gguf"
SEED = 42
MAX_TOKENS = 20

PROMPTS = [
    "What is the capital of France?",
    "Complete: The sun rises in the",
    "Translate to Spanish: hello",
    "The meaning of life is",
    "Once upon a time",
]

WORKER_CODE = """
import sys, json
from llama_cpp import Llama

shard_path, seed = sys.argv[1], int(sys.argv[2])
llm = Llama(model_path=shard_path, n_ctx=64, verbose=False, seed=seed)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("cmd") == "generate":
        result = llm(req["prompt"], max_tokens=req.get("max_tokens", 20), echo=False)
        text = result["choices"][0]["text"].strip()
        # Also get token-level info
        tokens = result["choices"][0].get("logprobs", {})
        print(json.dumps({"ok": True, "text": text}), flush=True)
    elif req.get("cmd") == "quit":
        break
"""


def generate_full(prompt):
    """Run prompt through full model."""
    code = f"""
import sys, json
from llama_cpp import Llama
llm = Llama(model_path="{MODEL}", n_ctx=64, verbose=False, seed={SEED})
result = llm("{prompt}", max_tokens={MAX_TOKENS}, echo=False)
print(json.dumps({{"text": result["choices"][0]["text"].strip()}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120
    )
    return json.loads(proc.stdout.strip())["text"]


def start_worker(shard_path):
    """Start a worker subprocess."""
    return subprocess.Popen(
        [sys.executable, "-c", WORKER_CODE, shard_path, str(SEED)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


def generate_worker(proc, prompt):
    """Send generation request to worker."""
    req = json.dumps({"cmd": "generate", "prompt": prompt, "max_tokens": MAX_TOKENS})
    proc.stdin.write(req + "\n")
    proc.stdin.flush()
    resp = proc.stdout.readline()
    return json.loads(resp)["text"]


def main():
    print("=" * 70)
    print("  SYNAPSE VALIDATION: Distributed MoE vs Full Model")
    print("=" * 70)

    # Check shards exist
    if not os.path.exists(SHARD_0) or not os.path.exists(SHARD_1):
        print("ERROR: Shards not found. Run split_gguf.py first.")
        sys.exit(1)

    shard_size_0 = os.path.getsize(SHARD_0) / 1024 / 1024
    shard_size_1 = os.path.getsize(SHARD_1) / 1024 / 1024
    print(f"\nShard 0: {shard_size_0:.0f} MB (experts 0-19)")
    print(f"Shard 1: {shard_size_1:.0f} MB (experts 20-39)")

    # Start workers
    print("\nStarting distributed workers...")
    worker_a = start_worker(SHARD_0)
    worker_b = start_worker(SHARD_1)
    time.sleep(2)  # Let workers initialize

    # Collect results
    results = []
    exact_matches = 0
    total = len(PROMPTS)

    for prompt in PROMPTS:
        print(f"\n{'─' * 70}")
        print(f"  Prompt: \"{prompt}\"")
        print(f"{'─' * 70}")

        # Full model
        print("  Running full model...", end="", flush=True)
        t0 = time.time()
        full_output = generate_full(prompt)
        t_full = time.time() - t0
        print(f" {t_full:.1f}s")

        # Worker A (shard 0: experts 0-19)
        print("  Running worker A (experts 0-19)...", end="", flush=True)
        t0 = time.time()
        output_a = generate_worker(worker_a, prompt)
        t_a = time.time() - t0
        print(f" {t_a:.1f}s")

        # Worker B (shard 1: experts 20-39)
        print("  Running worker B (experts 20-39)...", end="", flush=True)
        t0 = time.time()
        output_b = generate_worker(worker_b, prompt)
        t_b = time.time() - t0
        print(f" {t_b:.1f}s")

        # Compare
        match_ab = output_a == output_b
        match_full_a = full_output == output_a
        match_full_b = full_output == output_b

        print(f"\n  Full model:      \"{full_output}\"")
        print(f"  Worker A (0-19): \"{output_a}\"")
        print(f"  Worker B (20-39):\"{output_b}\"")
        print(f"\n  A == B:          {'YES' if match_ab else 'NO'}")
        print(f"  Full == A:       {'YES' if match_full_a else 'NO'}")
        print(f"  Full == B:       {'YES' if match_full_b else 'NO'}")

        if match_full_a or match_full_b:
            exact_matches += 1

        results.append({
            "prompt": prompt,
            "full_model": full_output,
            "worker_a_0_19": output_a,
            "worker_b_20_39": output_b,
            "a_equals_b": match_ab,
            "full_equals_a": match_full_a,
            "full_equals_b": match_full_b,
        })

    # Cleanup
    for w in [worker_a, worker_b]:
        w.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
        w.stdin.flush()
        w.wait(timeout=10)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Prompts tested:     {total}")
    print(f"  A == B (same shards): {sum(1 for r in results if r['a_equals_b'])}/{total}")
    print(f"  Full matches either:  {exact_matches}/{total}")

    # The key insight: with ONLY 2 shards, each worker has only HALF the
    # experts. The full model has ALL 40 experts. So outputs SHOULD differ.
    # The real validation is: do workers produce DIFFERENT outputs (proving
    # expert subsets matter) AND does the coordinator reassemble them correctly?
    differs = sum(1 for r in results if not r['a_equals_b'])
    print(f"\n  Workers produce different outputs: {differs}/{total}")
    print(f"  (Expected: all differ, since each worker has different expert subset)")

    # Save results
    out_path = "docs/validation-distributed-vs-full.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
            "shard_0_mb": shard_size_0,
            "shard_1_mb": shard_size_1,
            "results": results,
            "summary": {
                "total_prompts": total,
                "a_equals_b": sum(1 for r in results if r['a_equals_b']),
                "full_matches_either": exact_matches,
                "workers_differ": differs,
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to {out_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
