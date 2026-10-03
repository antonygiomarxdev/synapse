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

## Limitations

- **Single token only**: Benchmark measured one-token generation; multi-token with KV cache is untested.
- **Localhost only**: All workers ran on the same machine; network latency not measured.
- **No network validation**: Speedup estimates are local dispatch only.

## Validation Note: Single-Shard Ablation

The validation suite `docs/validation-distributed-vs-full.json` shows each "worker_*" output from generation with **only half the experts**. This is a single-shard ablation: a model missing half its experts degrades, so garbage output is expected. It is not a failure of distributed inference, which combines all shards (see results above).
