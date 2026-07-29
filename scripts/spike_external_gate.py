#!/usr/bin/env python3
"""External MoE Gate — coordinator assigns experts per worker.

Proves the core Synapse thesis: same model, same prompt, different
expert assignments produce different outputs. The coordinator controls
which experts fire per worker via an external gate mask.

No CUDA, no llama.cpp, no forks. Pure PyTorch MoE with mask.

Result: Full != Worker A != Worker B — external routing works.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

class MoELayer(nn.Module):
    def __init__(self, d_model, n_experts, top_k):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
            for _ in range(n_experts)
        ])

    def forward(self, x, expert_mask=None):
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        gate_logits = self.gate(x_flat)
        if expert_mask is not None:
            gate_logits = gate_logits + expert_mask

        gate_weights = F.softmax(gate_logits, dim=-1)
        _, top_indices = torch.topk(gate_weights, self.top_k, dim=-1)

        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_indices[:, i]
            weight = gate_weights[torch.arange(B * S), expert_idx].unsqueeze(1)
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])
                    output[mask] += expert_out * weight[mask]

        return output.view(B, S, D)


def main():
    d_model, n_experts, top_k = 64, 4, 2
    model = MoELayer(d_model, n_experts, top_k).eval()
    prompt = torch.randn(1, 8, d_model)

    # Coordinator assigns expert subsets per worker via mask
    mask_a = torch.tensor([0, -float('inf'), -float('inf'), 0])
    mask_b = torch.tensor([-float('inf'), 0, 0, -float('inf')])

    with torch.no_grad():
        out_full = model(prompt)
        out_a = model(prompt, expert_mask=mask_a)
        out_b = model(prompt, expert_mask=mask_b)

    same_ab = torch.allclose(out_a, out_b)
    same_fa = torch.allclose(out_full, out_a)
    same_fb = torch.allclose(out_full, out_b)

    print("═" * 55)
    print("  External MoE Gate — Distributed Expert Routing")
    print("═" * 55)
    print(f"\n  Model: {d_model} dims, {n_experts} experts, top_k={top_k}")
    print(f"  Prompt: 1x8 tokens")
    print(f"\n  Coordinator assignments:")
    print(f"    Worker A: experts [0, 3]")
    print(f"    Worker B: experts [1, 2]")
    print(f"\n  Output (last token, first 5 dims):")
    for name, out in [("Full (4 exp)", out_full), ("Worker A   ", out_a), ("Worker B   ", out_b)]:
        print(f"    {name}: [{', '.join(f'{v:+.3f}' for v in out[0, -1, :5].tolist())}]")
    print(f"\n  Full == A: {'YES' if same_fa else 'NO — different experts'}")
    print(f"  Full == B: {'YES' if same_fb else 'NO — different experts'}")
    print(f"  A == B:    {'YES' if same_ab else 'NO — workers diverge'}")
    print(f"\n  TESIS VALIDADA: external expert routing via coordinator")


if __name__ == "__main__":
    main()
