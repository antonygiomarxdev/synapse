#!/usr/bin/env python3
"""Deep comparison of logits distributions."""
import numpy as np

MODEL_PATH = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

def main():
    from llama_cpp import Llama
    llm = Llama(model_path=MODEL_PATH, n_ctx=1, n_threads=1, logits_all=True, verbose=False)
    llm.reset()
    llm.eval([49])
    ref = np.array(llm.scores[0])
    
    our = np.loadtxt('/tmp/our_single_logits.txt')
    
    # Save reference for later
    np.savetxt('/tmp/ref_single_logits.txt', ref, fmt='%.6f')
    
    print("=== Distribution Comparison ===")
    print(f"Reference: mean={np.mean(ref):.4f} std={np.std(ref):.4f} max={np.max(ref):.2f} min={np.min(ref):.2f}")
    print(f"Ours:      mean={np.mean(our):.4f} std={np.std(our):.4f} max={np.max(our):.2f} min={np.min(our):.2f}")
    
    # Per-element correlation
    corr = np.corrcoef(ref, our)[0, 1]
    cos = np.dot(ref, our) / (np.linalg.norm(ref) * np.linalg.norm(our))
    print(f"\nCorrelation: {corr:.4f}")
    print(f"Cosine sim:  {cos:.4f}")
    
    # Linear fit
    slope, intercept = np.polyfit(our, ref, 1)
    print(f"Linear fit:  ref = {slope:.4f} * ours + {intercept:.4f}")
    
    # Residual analysis
    predicted = slope * our + intercept
    residual = ref - predicted
    print(f"\nResidual after linear fit: mean={np.mean(residual):.4f} std={np.std(residual):.4f}")
    print(f"Residual max: {np.max(np.abs(residual)):.2f}")
    
    # Check if the error is uniform or varies by token range
    print(f"\n=== Error by token range ===")
    for start in [0, 100, 1000, 5000, 10000, 20000, 30000, 40000]:
        end = min(start + 1000, len(ref))
        r = ref[start:end]
        o = our[start:end]
        c = np.corrcoef(r, o)[0, 1]
        diff = np.mean(o - r)
        print(f"  tokens {start:5d}-{end:5d}: corr={c:.4f} mean_diff={diff:+.4f}")
    
    # Check specific reference top tokens
    print(f"\n=== Reference top-20 logits ===")
    top_idx = np.argsort(ref)[-20:][::-1]
    for i in top_idx:
        print(f"  token {i:5d}: ref={ref[i]:8.4f} ours={our[i]:8.4f} diff={our[i]-ref[i]:+8.4f}")
    
    # Check our top-20 logits
    print(f"\n=== Our top-20 logits ===")
    top_idx = np.argsort(our)[-20:][::-1]
    for i in top_idx:
        print(f"  token {i:5d}: ref={ref[i]:8.4f} ours={our[i]:8.4f} diff={our[i]-ref[i]:+8.4f}")
    
    # Check what happens with the output_norm norm
    print(f"\n=== Output norm comparison ===")
    print(f"Ref normed (estimated from logits): max logit={np.max(ref):.2f}")
    print(f"Ours normed: max logit={np.max(our):.2f}")
    
    # Check the ratio distribution
    # Avoid division by zero
    mask = np.abs(ref) > 0.01
    ratios = our[mask] / ref[mask]
    print(f"\n=== our/ref ratio (where |ref| > 0.01) ===")
    print(f"  mean ratio: {np.mean(ratios):.4f}")
    print(f"  median ratio: {np.median(ratios):.4f}")
    print(f"  std ratio: {np.std(ratios):.4f}")
    print(f"  ratio range: [{np.min(ratios):.4f}, {np.max(ratios):.4f}]")

if __name__ == '__main__':
    main()
