# Next Session Context: Native MoE Runtime — Forward Pass Validated

## Quick Start

```
cd /home/ksante/orca/workspaces/synapse/issue-13-session-2026-07
cargo test --release --lib single_token_logits -- --nocapture
```

## Current State

**Issue #20: ✅ RESOLVED** — Correlation 0.999334 (>0.99 target)

Forward pass produces logits that match llama-cpp-python with correlation 0.999.

## What's Working

- **Full forward pass:** 32 layers, single-token, correlation 0.999 with llama.cpp
- **Dequantization:** Q8_0, Q4_K, Q6_K — all produce correct values in correct order
- **Attention:** GQA with RoPE, correct for single-token inference
- **MoE FFN:** Gate routing + SiLU + weighted expert aggregation
- **Residual connections:** `hidden = residual + 0.22 * sublayer_output`
- **Output norm + logits:** tied embedding weights, logit_scale=6.0

## Bug Fixed This Session

**Q6_K element ordering** (`synapse-core/src/native_moe/quant.rs`):
- ggml outputs: all q1(l=0..31), all q2(l=0..31), all q3(l=0..31), all q4(l=0..31)
- We were outputting: q1(l=0),q2(l=0),q3(l=0),q4(l=0),q1(l=1)...
- Affected: `attn_v` (Q6_K) and `ffn_down_exps` (Q6_K)
- Fix: changed loop order in `dequant_q6_k_raw()`

## Previous Bugs (already fixed, don't redo)

1. Embedding indexing: `data[t * d + dim]` (column-major)
2. Expert weight indexing: `data[e * d_ff * d_model + j * d_model + d]`
3. Gate_inp indexing: `data[e * d_model + d]`
4. mat_vec_transposed: stride `data[i + j * ne0]`
5. Softmax routing: apply to ALL scores before top-k
6. Expert scores normalization: normalize to sum to 1
7. attention_scale: use model's (0.015625)

## Next Steps for the Project

### 1. Multi-token verification
The single-token forward is validated. Need to verify multi-token (causal attention with KV cache). The RoPE for pos>0 needs testing.

### 2. Performance optimization
Current forward pass is triple-loop pure Rust. Needs:
- SIMD/BLAS for mat_vec_transposed
- Parallel expert execution
- Cache-friendly memory access patterns

### 3. Distributed expert execution (the real goal)
The whole point of the native runtime is to enable distributed inference where:
- N nodes each hold a subset of experts
- Coordinator routes requests to nodes with the needed experts
- Each node executes only its local experts
- Results are aggregated with routing scores

Key files for this:
- `synapse-core/src/native_moe/runtime.rs` — InferencePort implementation
- `synapse-core/src/swarm/coordinator.rs` — ExpertRouter, GateInpLayer
- `synapse-core/src/transport/` — WebRTC, signaling

### 4. InferencePort integration
`NativeMoeRuntime` needs to implement `InferencePort` trait properly so it can be used by the swarm coordinator. The trait is defined but the integration is incomplete.

### 5. Model support
Currently only Granite MoE 3B. Need to generalize for other MoE models (Mixtral, DeepSeek, Kimi K3).

## Reference Values

**Model:** Granite MoE 3B (GGUF, Q4_K/Q6_K quantization)
- d_model=1536, n_heads=24, n_kv_heads=8, head_dim=64
- n_layers=32, n_experts=40, n_expert_used=8, d_ff=512
- vocab_size=49155, rope_theta=1e7
- embedding_scale=12.0, residual_scale=0.22, logit_scale=6.0, attention_scale=0.015625

**Reference logits (token 49, single):**
- mean: -2.72, max: 12.60, top-5: [34, 36, 35, 37, 308]

**Our logits (token 49, single):**
- mean: -2.60, max: 12.53, top-5: [34, 36, 35, 308, 37]
- correlation: 0.999334

## Files

| File | Role |
|------|------|
| `synapse-core/src/native_moe/forward.rs` | Forward pass, attention, FFN, RoPE |
| `synapse-core/src/native_moe/model.rs` | Model loading, config |
| `synapse-core/src/native_moe/quant.rs` | Dequantization (Q8_0, Q4_K, Q6_K) |
| `synapse-core/src/native_moe/gguf.rs` | GGUF v3 parser |
| `synapse-core/src/native_moe/ops.rs` | Low-level tensor ops |
| `synapse-core/src/native_moe/runtime.rs` | InferencePort implementation |
| `docs/adr/0011-native-moe-runtime.md` | Architecture decision record |

## What NOT to Do

1. Don't re-verify single-token forward — already at 0.999 correlation
2. Don't re-verify dequantization — Q4_K and Q6_K are correct
3. Don't re-verify layer-by-layer hidden states — already compared with llama.cpp
4. Don't try ensemble voting — already done, gives 0.79 correlation
