# MoE-Only Protocol Design

The protocol only supports Mixture-of-Experts models. Dense models (LLaMA, GPT) are out of scope.

**Why:** Synapse's architectural premise is distributing inference across consumer GPUs. This only works with MoE models because they activate a fraction of their total parameters per token — Mixtral 8x7B activates 2 of 8 experts (~25%), Kimi K3 activates 16 of 896 (~1.8%). The remaining 75-98% of parameters are idle for any given token, which means they can live on other nodes.

Dense models activate 100% of parameters per token — there's nothing to distribute. They need >70GB VRAM for frontier quality, which means datacenter GPUs.

**What this means for the protocol:**
- No tensor parallelism, no pipeline parallelism, no dense-model sharding logic
- Expert routing, co-activation heat maps, and DAG execution are first-class primitives
- V1 catalog: Mixtral 8x7B, Mixtral 8x22B, Qwen2.5-MoE, DeepSeek-V2 Lite, Kimi K3
- Community PRs must verify MoE compatibility before catalog listing
- Unique market position: no existing inference service does swarm-based MoE

**Trade-off:** Excludes most currently popular open models. If MoE adoption slows, the addressable market is constrained. But supporting both dense and MoE would double protocol complexity for no architectural synergy.
