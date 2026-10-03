# Async Throughput Benchmark — 2026-10-02

Issue: [#67](https://github.com/antonygiomarxdev/synapse/issues/67)

## Question

For async/batch workloads, per-token latency does not matter; aggregate
throughput does. Does batching many sequences per FFN request amortize
network round-trip time, so throughput stays roughly flat as latency grows?

## Setup

- **Model:** granite3.1-moe:3b (32 layers, 40 experts, top-8)
- **Topology:** coordinator (attention + routing) + 2 expert workers
  (experts 0–19, 20–39), all on one machine (22 cores, 93 GB RAM), release build
- **Latency:** simulated with `expert_worker --delay-ms D`, a sleep per FFN request
- **Workload:** `DistributedModel::forward_batch` on K single-token sequences.
  One FFN request per worker per layer carries all K rows.
- **Command:** `cargo run --release --bin bench_async_throughput -- --k 1,8,32,128 --delay 0,20,100`
- **Latency fraction:** `32 × D / wall` — share of wall time spent in simulated round-trips
- Single run per cell; laptop, so expect run-to-run noise of roughly ±20%
  (the same D=0, K=1 cell measured 789 ms and 592 ms in two runs)

## Results

| D (ms) | K | Wall (ms) | Tokens/s | Latency fraction |
|-------:|----:|----------:|---------:|-----------------:|
| 0 | 1 | 592 | 1.69 | 0.000 |
| 0 | 8 | 4421 | 1.81 | 0.000 |
| 0 | 32 | 16506 | 1.94 | 0.000 |
| 0 | 128 | 52301 | 2.45 | 0.000 |
| 20 | 1 | 1124 | 0.89 | 0.569 |
| 20 | 8 | 4293 | 1.86 | 0.149 |
| 20 | 32 | 17118 | 1.87 | 0.037 |
| 20 | 128 | 59226 | 2.16 | 0.011 |
| 100 | 1 | 3836 | 0.26 | 0.834 |
| 100 | 8 | 7089 | 1.13 | 0.451 |
| 100 | 32 | 17249 | 1.86 | 0.186 |
| 100 | 128 | 67416 | 1.90 | 0.048 |

K=128 rows come from a rerun after raising the worker body limit (the first run
hit axum's 2 MB default and silently substituted zeros for 13 of 32 layers).

## Findings

1. **Batching amortizes latency.** At 100 ms per round-trip, one sequence spends
   83% of wall time waiting on the network and runs at 0.26 tokens/s. With 32
   sequences in flight the same latency costs 19% of wall time, and throughput
   (1.86 tokens/s) is within 4% of the zero-latency figure (1.94).
2. **The ceiling is compute, not network.** Throughput tops out near 2–2.5
   tokens/s regardless of latency. Workers compute rows serially on one thread
   with naive loops (no SIMD/BLAS), and the coordinator runs attention per
   sequence serially.
3. **Correctness holds.** `forward_batch` logits match running each sequence
   alone (cosine > 0.9999, identical top-5), and distributed matches monolithic
   (cosine > 0.999, identical top-5). Tests: `forward_batch_matches_individual_sequences`,
   `distributed_matches_monolithic_logits` in `tests/e2e_distributed.rs`.

## Limitations

- **Slow FFN makes the curve flat early.** CPU FFN takes ~0.5 s per token for all
  layers, so compute dwarfs latency at small K. With GPU-speed FFN the same
  latency fraction needs proportionally larger K.
- **Simulated latency, one machine.** The sleep models round-trip time, not
  bandwidth limits, packet loss or jitter. Real multi-machine runs are pending.
- **Single-token sequences.** FFN cost per row is the same for prefill and
  decode steps, so the measurement transfers to decode, but multi-token
  sequences also need per-token routing ([#66](https://github.com/antonygiomarxdev/synapse/issues/66))
  and the KV cache ([#60](https://github.com/antonygiomarxdev/synapse/issues/60)).
- **Single run per cell.** Treat differences under ~20% as noise.

## Next

- Parallelize row computation inside each worker (multi-threaded or BLAS) to lift the compute ceiling
- Repeat on two physical machines
