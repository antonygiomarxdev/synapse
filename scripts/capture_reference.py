#!/usr/bin/env python3
"""Capture reference outputs from Granite MoE 3B via ollama API.

Runs 5 prompts with fixed seed (42) and temperature 0 for reproducibility.
Saves results to docs/reference-outputs.json.
"""
import json
import urllib.request
import time

MODEL = "granite3.1-moe:3b"
SEED = 42
MAX_TOKENS = 20

PROMPTS = [
    "What is the capital of France?",
    "Complete: The sun rises in the",
    "Translate to Spanish: hello",
    "The meaning of life is",
    "Once upon a time",
]

def generate(prompt: str) -> dict:
    """Call ollama API to generate text."""
    url = "http://localhost:11434/api/generate"
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,
        "options": {
            "seed": SEED,
            "temperature": 0.0,
            "num_predict": MAX_TOKENS,
        },
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main():
    results = []
    print(f"Model: {MODEL}")
    print(f"Seed: {SEED}, Max tokens: {MAX_TOKENS}")
    print("=" * 60)

    for prompt in PROMPTS:
        print(f"\n> {prompt}")
        t0 = time.time()
        data = generate(prompt)
        elapsed = time.time() - t0

        response = data.get("response", "").strip()
        print(f"  {response}")
        print(f"  [{elapsed:.1f}s, {data.get('eval_count', '?')} tokens]")

        logprobs = data.get("logprobs", [])
        results.append({
            "prompt": prompt,
            "response": response,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
            "eval_count": data.get("eval_count"),
            "eval_duration_ns": data.get("eval_duration"),
            "total_duration_ns": data.get("total_duration"),
            "logprobs": logprobs,
        })

    out_path = "docs/reference-outputs.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Saved {len(results)} reference outputs to {out_path}")


if __name__ == "__main__":
    main()
