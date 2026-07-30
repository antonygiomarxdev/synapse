# ADR-0009: Native MoE Runtime — Rust + GGUF Parser

**Status:** Spike validated, parser implemented
**Date:** 2026-07-29
**Deciders:** @antonygiomarxdev

## Context

Synapse needs to dynamically control which experts fire per token in a MoE model forward pass. No existing inference engine (vLLM, llama.cpp, Ollama, SGLang, TensorRT-LLM) exposes expert routing as an external API. All assume the router is internal to the model.

We attempted multiple paths before settling on the current approach:

1. **llama.cpp PR for `selected_experts_in`** — `build_moe_ffn()` already accepts external expert IDs, but exposing it via C API touches 20+ model files and requires a 1-line fix in `granite.cpp`. The PR is small (~30 lines) but requires contributor approval, issue first, and we couldn't test locally due to C++ build issues.

2. **Zero-out experts** — Patching `gate_inp` rows to -inf in the GGUF. Result: softmax breaks, produces NaN/comma output. Not viable.

3. **Gate masking in llama.cpp** — Modifying `gate_inp` weights at runtime. Conceptually works but requires C++ build toolchain stability.

4. **Python + ctypes to libllama.so** — C++ symbols are name-mangled, structs don't map cleanly to ctypes. Multiple segfaults.

5. **vLLM expert parallelism** — Exists but is internal. `ExpertPlacementStrategy` distributes experts across GPUs but doesn't expose per-token routing.

6. **FastMoE / Tutel** — Both support custom gates and experts. Tutel supports `gate_type: 'custom'`. But both require CUDA toolkit (nvcc) to compile, which isn't available on this system.

7. **llama.cpp example (`llama-cli`, `llama-simple`)** — These work. The 1-line fix (`t_layer_inp[il] = inpL`) in `granite.cpp` exposes hidden states at layer boundaries. Combined with pre-exported `gate_inp.bin`, we successfully extracted hidden states and computed external expert routing on Granite MoE 3B.

8. **External MoE gate (PyTorch)** — A minimal MoE model with `expert_mask` parameter proves external routing works at the architecture level. Pure PyTorch, no CUDA.

## Decision

**Build a native MoE runtime using Rust + our own GGUF parser + ggml-rs.** Own the forward pass end-to-end. No llama.cpp modifications, no forks, no Python subprocess.

The runtime will:

1. Parse GGUF files natively (Phase 1 ✅)
2. Load model weights (F32, F16, quantized formats)
3. Run the transformer forward pass with external routing hooks
4. Implement the `InferencePort` trait to integrate with existing swarm infrastructure

This is the approach because:
- We can dynamically control expert selection per token
- No dependency on external inference engines
- Integrates directly with `ExpertRouter` and `SwarmCoordinator` traits
- Scales to any MoE model (Granite, Mixtral, DeepSeek, Kimi K3)
- Uses ggml-rs for tensor ops (avoiding CUDA dependency for V0 CPU inference)

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| PR to llama.cpp | Build issues, 20+ model files to touch, contributor approval process |
| Python workers with shards | No per-token routing control |
| FastMoE/Tutel custom gate | Requires CUDA toolkit (nvcc) |
| Zero-out experts in GGUF | Softmax breaks numerically |
| ctypes to libllama.so | C++ name-mangled symbols, struct mismatches |
| Fork of llama.cpp | Maintenance burden, diverges from upstream |

## Phases

### Phase 1: GGUF Parser (✅ Done)
Parse GGUF v3 binary format: header, metadata KV pairs, tensor index.
- File: `synapse-core/src/native_moe/gguf.rs`
- Tests: parse Granite MoE 3B, verify 322 tensors, 42 KV pairs, correct dimensions
- Finds tensors by name, reads F32 data directly

### Phase 2: Model Loading
- Module: `synapse-core/src/native_moe/model.rs`
- Structs: `MoeConfig`, `MoeLayer`, `MoeModel`
- Load all tensors organized by layer

### Phase 3: Forward Loop
- Module: `synapse-core/src/native_moe/forward.rs`
- Implement: embedding, RMS norm, attention, MoE FFN, output projection
- External routing hook via `ExpertRouter` trait
- Use ggml-rs for efficient tensor operations

### Phase 4: InferencePort Implementation
- Module: `synapse-core/src/native_moe/runtime.rs`
- Implement `InferencePort` trait
- Integrate with `SwarmCoordinator` and `Libp2pSwarmCoordinator`

## Learnings from 2026-07-29 spikes

1. **Shared layers are identical across shards** — 194/194 tensors bit-identical. Coordinators don't need full model.
2. **Expert specialization is real** — Shards produce divergent outputs. Zero-out experts consistently change model output (5/5 prompts diverge).
3. **External routing is mathematically correct** — Rust coordinator produces 8/8 expert matches vs direct broadcast.
4. **Hidden state extraction works** — `llama_get_embeddings_layer_inp()` extracts real hidden states at MoE layer boundaries. Requires 1-line `t_layer_inp` initialization in `granite.cpp`.
5. **Gate masking works at architecture level** — Same prompt, different expert subsets produce different outputs in a PyTorch MoE model.

## Constraints

- No CUDA dependency for V0 — CPU-only forward pass acceptable for spike validation
- Must not require external inference engines (no vLLM, no Ollama)
- Must integrate with existing `InferencePort`, `ExpertRouter`, `SwarmCoordinator` traits
- Must parse GGUF v3 (the llama.cpp model format)
- Must handle Granite MoE 3B as the validation model (40 experts, 32 layers, d_model=1536)
