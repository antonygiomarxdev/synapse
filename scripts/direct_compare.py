#!/usr/bin/env python3
"""Direct comparison: run llama-cpp-python forward, dump hidden states via embeddings API."""
import numpy as np
from llama_cpp import Llama

MODEL = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

def main():
    # Load model with embedding support
    llm = Llama(model_path=MODEL, n_ctx=128, n_threads=1, 
                embedding=True, logits_all=True, verbose=False)
    
    # Test different tokens to see if the error pattern is consistent
    tokens_to_test = [49, 34, 36, 35, 37, 308]
    
    for tok in tokens_to_test:
        llm.reset()
        llm.eval([tok])
        logits = np.array(llm.scores[0])
        
        top5 = np.argsort(logits)[-5:][::-1]
        print(f"Token {tok:5d}: mean={np.mean(logits):.4f} max={np.max(logits):.2f} "
              f"top5={top5.tolist()}")
    
    # Now test with our Rust logits
    print(f"\n=== Our Rust logits (token 49, 32 layers) ===")
    our = np.loadtxt('/tmp/our_single_logits.txt')
    ref = np.array(llm.scores[0])  # Reload token 49
    llm.reset(); llm.eval([49]); ref = np.array(llm.scores[0])
    
    # Detailed comparison for token 49
    print(f"\nReference (llama-cpp):")
    print(f"  mean={np.mean(ref):.4f} std={np.std(ref):.4f} max={np.max(ref):.2f}")
    top5_ref = np.argsort(ref)[-5:][::-1]
    for i in top5_ref:
        print(f"  token {i:5d}: {ref[i]:.4f}")
    
    print(f"\nOurs (Rust):")
    print(f"  mean={np.mean(our):.4f} std={np.std(our):.4f} max={np.max(our):.2f}")
    top5_our = np.argsort(our)[-5:][::-1]
    for i in top5_our:
        print(f"  token {i:5d}: {our[i]:.4f}")
    
    # Check if the error is in the embedding/output projection
    # by comparing the logits for tokens that SHOULD be similar
    print(f"\n=== Error analysis ===")
    # For tokens 34, 35, 36, 37 (consecutive), the logits should be similar
    for t in [34, 35, 36, 37]:
        print(f"  token {t}: ref={ref[t]:.4f} ours={our[t]:.4f} diff={our[t]-ref[t]:+.4f}")
    
    # Check the ratio
    mask = np.abs(ref) > 0.5
    ratios = our[mask] / ref[mask]
    print(f"\n  Ratio (|ref|>0.5): mean={np.mean(ratios):.4f} std={np.std(ratios):.4f}")
    
    # Check if there's a linear relationship
    slope, intercept = np.polyfit(our, ref, 1)
    print(f"  Linear fit: ref = {slope:.4f} * ours + {intercept:.4f}")
    
    # Check residual after removing linear fit
    predicted = slope * our + intercept
    residual = ref - predicted
    print(f"  Residual: mean={np.mean(residual):.4f} std={np.std(residual):.4f}")
    print(f"  Residual max: {np.max(np.abs(residual)):.2f}")

if __name__ == '__main__':
    main()
