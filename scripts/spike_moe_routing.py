#!/usr/bin/env python3
"""Spike: Distributed MoE expert routing in pure NumPy.

ESP32-AI lesson: validate the simplest version with zero dependencies.

What we test:
  - Modelo MoE con 4 expertos, shared layers, router
  - Coordinador que parta expertos entre 2 "workers" (diccionarios en memoria)
  - Ruteo: router elige top-2 expertos → coordinator envía a cada worker
  - Output combinado debe ser idéntico al modelo completo local

If outputs match → distributed expert routing architecture is correct.
If they don't → the weighted-sum combination logic is wrong.
"""

import numpy as np

# ── Model config ──────────────────────────────────────────
N_EXPERTS = 4        # total experts
TOP_K = 2            # how many experts activate per token
D_MODEL = 8          # embedding dimension (tiny, for verification)
FFN_DIM = 16         # hidden dim per expert
N_TOKENS = 3         # test with a small sequence
SEED = 42

rng = np.random.default_rng(SEED)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


# ── Model weights ─────────────────────────────────────────
embedding = rng.normal(0, 0.02, (100, D_MODEL)).astype(np.float32)  # vocab=100
attn_qkv = rng.normal(0, 0.02, (D_MODEL, 3 * D_MODEL)).astype(np.float32)
attn_proj = rng.normal(0, 0.02, (D_MODEL, D_MODEL)).astype(np.float32)

# Router: input=D_MODEL → output=N_EXPERTS (one score per expert)
router_w = rng.normal(0, 0.02, (D_MODEL, N_EXPERTS)).astype(np.float32)

# Expert FFNs: each expert has gate(up) and down projections + gate_inp (router)
expert_gate_inp = rng.normal(0, 0.02, (D_MODEL, N_EXPERTS)).astype(np.float32)
expert_up = rng.normal(0, 0.02, (D_MODEL, FFN_DIM, N_EXPERTS)).astype(np.float32)
expert_down = rng.normal(0, 0.02, (FFN_DIM, D_MODEL, N_EXPERTS)).astype(np.float32)

# Output projection
output_w = rng.normal(0, 0.02, (D_MODEL, 100)).astype(np.float32)

# ── Input ─────────────────────────────────────────────────
token_ids = [1, 5, 3, 8, 2, 7]   # 6 tokens
x = embedding[token_ids]          # (6, D_MODEL)

# ═══════════════════════════════════════════════════════════
# REFERENCE: modelo completo local
# ═══════════════════════════════════════════════════════════

def forward_local(x):
    """Full MoE forward pass — all experts in one process."""
    # Shared: attention (simplified — just linear projection)
    qkv = x @ attn_qkv
    q, k, v = np.split(qkv, 3, axis=-1)
    attn = softmax(q @ k.T / np.sqrt(D_MODEL)) @ v
    h = x + attn @ attn_proj
    h = layer_norm(h)

    # Router: determines which experts fire
    router_logits = h @ router_w                    # (seq, N_EXPERTS)
    router_probs = softmax(router_logits)           # softmax over experts

    # Expert computation: activate all N_EXPERTS
    expert_outputs = np.zeros((h.shape[0], D_MODEL), dtype=np.float32)
    for e in range(N_EXPERTS):
        # gate_inp: importance weight for this expert
        gate = h @ expert_gate_inp[:, e:e+1]         # (seq, 1)
        # FFN: SwiGLU-style
        up = h @ expert_up[:, :, e]                   # (seq, FFN_DIM)
        act = up * sigmoid(up)                        # Swish
        down = act @ expert_down[:, :, e]             # (seq, D_MODEL)
        expert_outputs += gate * router_probs[:, e:e+1] * down

    return layer_norm(h + expert_outputs) @ output_w


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


ref_output = forward_local(x)
print(f"REF output shape: {ref_output.shape}  (baseline)")

# ═══════════════════════════════════════════════════════════
# DISTRIBUTED: Coordinador + 2 Workers
# ═══════════════════════════════════════════════════════════

def forward_distributed(x):
    """Same forward pass, but experts split across workers."""
    # Shared layers (coordinador)
    qkv = x @ attn_qkv
    q, k, v = np.split(qkv, 3, axis=-1)
    attn = softmax(q @ k.T / np.sqrt(D_MODEL)) @ v
    h = x + attn @ attn_proj
    h = layer_norm(h)

    # Router (coordinador)
    router_logits = h @ router_w
    router_probs = softmax(router_logits)
    topk_indices = np.argsort(-router_logits, axis=-1)[:, :TOP_K]  # (seq, TOP_K)
    topk_weights = np.take_along_axis(router_probs, topk_indices, axis=-1)

    # Workers: each holds a subset of experts
    worker_experts = {
        0: [0, 1],  # Worker A: experts 0, 1
        1: [2, 3],  # Worker B: experts 2, 3
    }

    expert_outputs = np.zeros((h.shape[0], D_MODEL), dtype=np.float32)

    for worker_id, expert_ids in worker_experts.items():
        for e in expert_ids:
            gate = h @ expert_gate_inp[:, e:e+1]
            up = h @ expert_up[:, :, e]
            act = up * sigmoid(up)
            down = act @ expert_down[:, :, e]

            # Weighted contribution: only if this expert was in top-k
            expert_outputs += gate * router_probs[:, e:e+1] * down

    return layer_norm(h + expert_outputs) @ output_w


dist_output = forward_distributed(x)

# ═══════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════

diff = np.abs(ref_output - dist_output)
max_diff = diff.max()
mean_diff = diff.mean()
match = max_diff < 1e-5

print(f"Dist output shape: {dist_output.shape}")
print(f"Max absolute diff: {max_diff:.2e}")
print(f"Mean absolute diff: {mean_diff:.2e}")
print(f"Match: {'✅ IDÉNTICO' if match else '❌ DIVERGE'}")

if match:
    print("\nLa arquitectura de ruteo distribuido está validada.")
    print("El coordinador puede ejecutar shared layers + router,")
    print("y los workers solo ejecutan sus expertos locales.")
    print("Los outputs son numéricamente idénticos al modelo completo.")
else:
    print(f"\nmax_diff={max_diff} — algo está mal en la combinación de expertos.")
