#!/usr/bin/env python3
"""Compare Rust logits at each layer count with llama-cpp-python reference."""
import numpy as np

def load_logits(path):
    return np.loadtxt(path)

def main():
    # Reference logits from llama-cpp-python (full 32 layers)
    from llama_cpp import Llama
    llm = Llama(
        model_path="/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b",
        n_ctx=1, n_threads=1, logits_all=True, verbose=False,
    )
    llm.reset()
    llm.eval([49])
    ref_logits = np.array(llm.scores[0])
    
    print(f"Reference: mean={np.mean(ref_logits):.4f} max={np.max(ref_logits):.2f}")
    print(f"Reference top-5: {np.argsort(ref_logits)[-5:][::-1].tolist()}")
    print()
    
    # Compare with our logits at each layer count
    for n_layers in [0, 1, 2, 5, 10, 32]:
        path = f"/tmp/rust_logits_{n_layers}layers.txt"
        try:
            our_logits = load_logits(path)
        except FileNotFoundError:
            continue
        
        corr = np.corrcoef(ref_logits, our_logits)[0, 1]
        cos_sim = np.dot(ref_logits, our_logits) / (np.linalg.norm(ref_logits) * np.linalg.norm(our_logits))
        mean_diff = np.mean(our_logits) - np.mean(ref_logits)
        max_ratio = np.max(our_logits) / np.max(ref_logits)
        
        top5_ours = np.argsort(our_logits)[-5:][::-1].tolist()
        top5_ref = np.argsort(ref_logits)[-5:][::-1].tolist()
        overlap = len(set(top5_ours) & set(top5_ref))
        
        print(f"Layers={n_layers:2}: corr={corr:.4f} cos={cos_sim:.4f} "
              f"mean_diff={mean_diff:+.2f} max_ratio={max_ratio:.2f} "
              f"top5_overlap={overlap}/5")
        print(f"  ours top-5: {top5_ours}")
        print(f"  ref  top-5: {top5_ref}")

if __name__ == '__main__':
    main()
