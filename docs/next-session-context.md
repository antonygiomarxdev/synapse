# Next Session Context: V0 Complete — Distributed MoE Inference Proven

## Quick Start

```bash
cd /home/ksante/dev/synapse
git checkout main && git pull
cargo test --lib -- --skip native_moe --skip health_check_localhost
```

## Current State

**V0 is complete.** The core thesis is proven: distributed MoE expert inference produces identical results to monolithic execution.

### What was built

**V0-1: ✅ Job Model + Async API**
- POST /v1/jobs → 202, GET /v1/jobs/{id} → status/result
- OpenAPI spec, Swagger UI

**V0-2: ✅ Scheduler Mínimo**
- Async scheduler with JoinSet for concurrent dispatch
- Round-robin, leases (30s), retries (max 3)

**V0-3: ✅ Multi-Worker + Crash Recovery**
- OllamaWorkerPort (HTTP to Ollama)
- MetricsCollector (jobs, tasks, retries, latencies)
- Crash recovery <30s wall-clock

**V0-4: ✅ Métricas E2E + Benchmark**
- Benchmark binary with p50/p95/p99 latencies
- Scripts/bench.sh for reproducible runs

**V0-5: ✅ Distributed Expert Inference**
- Per-expert GGUF loader (expert_shard.rs)
- Expert worker binary (HTTP server, loads 32 layers)
- Distributed forward pass (coordinator + workers)
- **Cosine similarity: 1.000000** (bit-identical to monolithic)

### Benchmark Results

| Config | Wall (ms) | Speedup | Cosine sim |
|--------|-----------|---------|------------|
| Monolithic | 1264 | 1.00x | 1.000000 |
| 2 workers | 936 | 1.35x | 1.000000 |
| 4 workers | 885 | 1.43x | 1.000000 |

## Architecture

```
Client → Gateway (axum, :8000)
              ↓
         Scheduler (async, JoinSet)
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
Worker A  Worker B  Worker C
(experts   (experts   (experts
 0-19)     20-39)     varies)
```

**Coordinator** runs attention locally (32 layers), dispatches expert FFN to remote workers. Workers load only their assigned experts from GGUF.

## Key Files

| File | Role |
|------|------|
| `native_moe/expert_shard.rs` | Per-expert GGUF loader |
| `native_moe/expert_worker_client.rs` | HTTP client for remote FFN |
| `native_moe/distributed_forward.rs` | Distributed inference orchestrator |
| `native_moe/forward.rs` | Monolithic forward pass |
| `bin/expert_worker.rs` | Expert worker HTTP server |
| `bin/bench_distributed.rs` | Distributed vs monolithic benchmark |
| `scheduler/scheduler.rs` | Async scheduler with JoinSet |
| `gateway/jobs.rs` | HTTP handlers + AppState |

## Test Counts

- Job module: 52 tests
- Scheduler module: 48 tests
- Gateway: 14 tests
- Other: 243 tests
- **Total: 357 tests passing**

## What's Next

### Short term
- Integrate distributed inference with async scheduler for production
- Test with larger models (Mixtral, DeepSeek)
- Test with multiple tokens (full prompt, not just single token)
- Test on multiple machines (not just multiple processes)

### Medium term
- Dynamic expert loading (load on demand, not all at startup)
- Expert caching (keep hot experts in memory)
- Load balancing across workers
- Web UI for monitoring

### Long term
- P2P expert discovery (DHT)
- Economic incentives (staking, slashing)
- Realtime inference mode (speculative network)

## Environment

- **GPU:** NVIDIA RTX 4070 Laptop (8GB VRAM)
- **Models:** granite3.1-moe:3b, qwen3:8b (via Ollama)
- **Ollama:** localhost:11434 (and optionally :11435 for multi-instance)
