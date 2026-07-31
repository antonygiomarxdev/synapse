# Distributed Inference Benchmark — 2026-07-31

## Configuration

- **Model:** granite3.1-moe:3b (32 layers, 40 experts, 8 active)
- **Prompt:** single token
- **Runs:** 3 per config, median reported

## Results

| Config | Wall (ms) | Speedup | Cosine sim | Top5 match |
|--------|-----------|---------|------------|------------|
| Monolithic | 1264 | 1.00x | 1.000000 | true |
| 2 workers | 936 | 1.35x | 1.000000 | true |
| 4 workers | 885 | 1.43x | 1.000000 | true |

## Key Finding

Distributed expert inference produces **identical logits** to monolithic execution.
This validates Synapse's core thesis: MoE experts can be distributed across
multiple workers without any loss in inference quality.
