# vLLM as Primary ML Runtime

The compute node uses vLLM as its inference engine in V1. Support for llama.cpp and SGLang is planned for V2+ behind the same `InferencePort` trait.

**Why vLLM:** Best MoE support among open runtimes — dynamic expert loading via the V1 engine, PagedAttention for efficient VRAM usage, deterministic generation with seed=0 (required for audit mode), and an OpenAI-compatible API for testing without the swarm.

**Why not llama.cpp (V1):** MoE expert loading requires custom changes — it's not a first-class feature. Python bindings are less mature. Output determinism across BLAS backends is harder to guarantee. Deferred to V2+.

**Why not custom Rust engine:** Building an MoE inference engine (CUDA kernels, quantization formats, dynamic batching) is a multi-year effort. The protocol adds value through distribution, not by rewriting inference.

**Architecture:** The Rust core spawns vLLM as a Python subprocess. Communication is protobuf over Unix socket through the `InferencePort` trait. This means:
- Rust core and vLLM can be versioned independently
- vLLM crashes don't take down the node
- Adding a new runtime (llama.cpp, SGLang) is a new adapter implementing the same trait
