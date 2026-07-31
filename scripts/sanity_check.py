#!/usr/bin/env python3
"""Sanity check: verify our inputs match what llama-cpp-python actually uses."""
import numpy as np
from llama_cpp import Llama

MODEL = "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"

def main():
    llm = Llama(model_path=MODEL, n_ctx=128, n_threads=1, 
                logits_all=True, verbose=False, add_bos=False)
    
    # Test 1: What token does llama-cpp-python see?
    print("=== Test 1: Token processing ===")
    tokens = llm.tokenize(b"?")  # token 49 = '?'
    print(f"tokenize('?') = {tokens}")
    print(f"detokenize([49]) = {llm.detokenize([49])}")
    
    # Test 2: Does add_bos matter?
    print("\n=== Test 2: BOS token effect ===")
    llm_bos = Llama(model_path=MODEL, n_ctx=128, n_threads=1, 
                    logits_all=True, verbose=False, add_bos=True)
    
    llm.reset(); llm.eval([49])
    l1 = np.array(llm.scores[0])
    
    llm_bos.reset(); llm_bos.eval([49])
    l2 = np.array(llm_bos.scores[0])
    
    print(f"add_bos=False: mean={np.mean(l1):.4f} top3={np.argsort(l1)[-3:][::-1].tolist()}")
    print(f"add_bos=True:  mean={np.mean(l2):.4f} top3={np.argsort(l2)[-3:][::-1].tolist()}")
    print(f"corr: {np.corrcoef(l1,l2)[0,1]:.6f}")
    print(f"identical: {np.allclose(l1, l2)}")
    
    # Test 3: What if we pass the token as text?
    print("\n=== Test 3: Text vs token ID ===")
    llm_text = Llama(model_path=MODEL, n_ctx=128, n_threads=1, 
                     logits_all=True, verbose=False, add_bos=False)
    llm_text.reset()
    tokens_text = llm_text.tokenize(b"?")
    print(f"tokenize('?') = {tokens_text}")
    llm_text.eval(tokens_text)
    l3 = np.array(llm_text.scores[0])
    
    print(f"eval([49]):     mean={np.mean(l1):.4f} top3={np.argsort(l1)[-3:][::-1].tolist()}")
    print(f"eval(tokenize): mean={np.mean(l3):.4f} top3={np.argsort(l3)[-3:][::-1].tolist()}")
    print(f"corr: {np.corrcoef(l1,l3)[0,1]:.6f}")
    
    # Test 4: Multiple tokens like Rust test
    print("\n=== Test 4: Multi-token sequence ===")
    llm_multi = Llama(model_path=MODEL, n_ctx=128, n_threads=1, 
                      logits_all=True, verbose=False, add_bos=False)
    
    # Same tokens as Rust trace test: [8197, 438, 322, 18926, 432, 45600, 49]
    multi_tokens = [8197, 438, 322, 18926, 432, 45600, 49]
    llm_multi.reset()
    llm_multi.eval(multi_tokens)
    
    # logits_all=True gives logits for all positions
    # The last position should predict next token after 49
    l4 = np.array(llm_multi.scores[-1])  # Last token's logits
    
    print(f"Multi-token (7 tokens):")
    print(f"  mean={np.mean(l4):.4f} max={np.max(l4):.2f}")
    print(f"  top5={np.argsort(l4)[-5:][::-1].tolist()}")
    
    # Compare with our Rust multi-token logits
    print(f"\n  Our Rust (7 tokens, 32 layers):")
    print(f"  mean=-1.62 max=11.12")
    print(f"  top5=[996, 372, 34]")
    
    # Test 5: Single token reference
    print(f"\n=== Test 5: Final reference ===")
    print(f"Single token 49 (add_bos=False):")
    print(f"  mean={np.mean(l1):.4f} max={np.max(l1):.2f}")
    print(f"  top5={np.argsort(l1)[-5:][::-1].tolist()}")
    
    # Our Rust
    our = np.loadtxt('/tmp/our_single_logits.txt')
    corr = np.corrcoef(l1, our)[0, 1]
    print(f"\nOur Rust (single token 49, 32 layers):")
    print(f"  mean={np.mean(our):.4f} max={np.max(our):.2f}")
    print(f"  top5={np.argsort(our)[-5:][::-1].tolist()}")
    print(f"\nCorrelation: {corr:.4f}")

if __name__ == '__main__':
    main()
