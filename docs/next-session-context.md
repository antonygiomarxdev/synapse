# Next Session Context: V0 Complete — V1 Roadmap Charted

## Quick Start

```bash
cd /home/ksante/dev/synapse
git checkout main && git pull
cargo test --lib -- --skip native_moe --skip health_check_localhost
```

## Current State

**V0 is complete.** The core thesis is proven: distributed MoE expert inference produces identical results to monolithic execution.

**V1 Roadmap is charted.** Wayfinder map created at #42 with 14 tickets.

### V0 Summary

| Issue | Status | Description |
|-------|--------|-------------|
| #20 | ✅ | Native MoE forward pass — correlation 0.999 |
| #21 | ✅ | Job Model + Async API |
| #22 | ✅ | Scheduler Mínimo |
| #23 | ✅ | Multi-Worker + Crash Recovery |
| #24 | ✅ | Métricas E2E + Benchmark |
| #25 | ✅ | Distributed Expert Inference — thesis proven |

### Benchmark Results

| Config | Wall (ms) | Speedup | Cosine sim |
|--------|-----------|---------|------------|
| Monolithic | 1264 | 1.00x | 1.000000 |
| 2 workers | 936 | 1.35x | 1.000000 |
| 4 workers | 885 | 1.43x | 1.000000 |

## V1 Roadmap (Wayfinder Map #42)

**Destination:** Synapse Core V1 — open source distributed MoE inference, Linux+NVIDIA first, designed to scale.

**Constraints:**
- Linux first, cross-platform in V2+
- NVIDIA (CUDA) first, cross-GPU in V2+
- MoE models only
- Apache 2.0 license

### Tickets

**Research (unblocks decisions):**
- #43: What do similar projects do? (Ollama, vLLM, llama.cpp, Petals)

**Decisions (blocked by #43):**
- #44: Installation strategy (cargo vs brew vs docker)
- #45: Networking strategy (TCP vs libp2p vs HTTP)
- #46: Model formats and supported models
- #47: Observability strategy (logging vs metrics vs traces)
- #48: Storage strategy (SQLite vs file vs RocksDB)

**Implementation (blocked by decisions):**
- #49: Gateway→Scheduler→Worker pipeline
- #50: TCP transport for multi-machine
- #51: /v1/chat/completions real routing
- #52: Observability (metrics + logging)
- #53: Persistent storage
- #54: Multi-token generation
- #55: E2E integration tests
- #56: Deployment guide

### Frontier (takeable now)

**#43 — Research: What do similar projects do?**

This is the first ticket to resolve. It unblocks all decision tickets.

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

## Environment

- **GPU:** NVIDIA RTX 4070 Laptop (8GB VRAM)
- **Models:** granite3.1-moe:3b, qwen3:8b (via Ollama)
- **Ollama:** localhost:11434 (and optionally :11435 for multi-instance)
