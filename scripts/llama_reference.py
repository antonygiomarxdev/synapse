#!/usr/bin/env python3
"""Get reference logits and hidden states from llama-cpp-python for comparison with Rust runtime."""
import json
import struct
import numpy as np

MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

def get_reference_logits():
    """Use llama-cpp-python to get reference logits for token 49."""
    from llama_cpp import Llama
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=1,
        n_threads=1,
        logits_all=True,
        verbose=False,
    )
    
    # Single token: 49
    tokens = [49]
    llm.reset()
    
    # Get logits
    result = llm.eval(tokens)
    logits = llm.scores[0]  # logits for the first (and only) token
    
    print(f"Reference logits shape: {len(logits)}")
    print(f"Reference mean: {np.mean(logits):.4f}")
    print(f"Reference std: {np.std(logits):.4f}")
    print(f"Reference max: {np.max(logits):.2f}")
    print(f"Reference min: {np.min(logits):.2f}")
    
    # Top-5
    top5_idx = np.argsort(logits)[-5:][::-1]
    top5_vals = [logits[i] for i in top5_idx]
    print(f"Reference top-5: {list(top5_idx)}")
    print(f"Reference top-5 vals: {[f'{v:.2f}' for v in top5_vals]}")
    
    return logits

def compare_with_rust():
    """Compare reference logits with our Rust logits."""
    ref_logits = get_reference_logits()
    
    # Load our Rust logits
    try:
        with open('/tmp/our_single_logits.txt') as f:
            our_logits = [float(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print("ERROR: /tmp/our_single_logits.txt not found. Run the Rust test first.")
        return
    
    our_logits = np.array(our_logits)
    
    print(f"\n=== Comparison ===")
    print(f"Our mean: {np.mean(our_logits):.4f}")
    print(f"Our max: {np.max(our_logits):.2f}")
    print(f"Ref mean: {np.mean(ref_logits):.4f}")
    print(f"Ref max: {np.max(ref_logits):.2f}")
    
    # Correlation
    corr = np.corrcoef(ref_logits, our_logits)[0, 1]
    print(f"Correlation: {corr:.4f}")
    
    # Linear fit
    slope, intercept = np.polyfit(our_logits, ref_logits, 1)
    print(f"Linear fit: ref = {slope:.3f} * ours + {intercept:.3f}")
    
    # Direction cosine
    cos_sim = np.dot(ref_logits, our_logits) / (np.linalg.norm(ref_logits) * np.linalg.norm(our_logits))
    print(f"Direction cosine: {cos_sim:.4f}")
    
    # Save comparison
    comparison = {
        'reference_mean': float(np.mean(ref_logits)),
        'reference_max': float(np.max(ref_logits)),
        'our_mean': float(np.mean(our_logits)),
        'our_max': float(np.max(our_logits)),
        'correlation': float(corr),
        'slope': float(slope),
        'intercept': float(intercept),
        'cosine_similarity': float(cos_sim),
    }
    
    with open('/tmp/logits_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to /tmp/logits_comparison.json")

if __name__ == '__main__':
    compare_with_rust()
