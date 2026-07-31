# Next Session Context: V1 Core Complete

## Quick Start

```bash
cd /home/ksante/dev/synapse
git checkout main && git pull
cargo test --lib -- --skip native_moe --skip health_check_localhost
cargo test --test e2e
```

## Current State

**V1 core is complete.** The gateway, scheduler, workers, observability, and E2E tests are all working.

### V1 Summary

| Issue | Status | Description | PR |
|-------|--------|-------------|-----|
| #49 | ✅ | Gateway→Scheduler→Worker pipeline | #57 |
| #50 | ✅ | TCP transport for multi-machine | #61 |
| #51 | ✅ | /v1/chat/completions real routing | #58 |
| #52 | ✅ | Observability (metrics + logging) | #62 |
| #54 | ✅ | Multi-token generation | #59 |
| #55 | ✅ | E2E integration tests | #63 |
| #56 | ✅ | Deployment guide | #63 |

### What's Working

- **Gateway**: axum HTTP server with OpenAI-compatible endpoints
- **Scheduler**: async scheduler with JoinSet for concurrent dispatch
- **Workers**: expert workers that load GGUF and serve FFN requests
- **TCP Transport**: multi-machine deployment via TCP
- **Observability**: Prometheus metrics endpoint
- **E2E Tests**: 8 integration tests covering full lifecycle

### What's Pending

- **#53**: Persistent storage (Turso/libSQL)
- **#60**: KV cache optimization
- **#62**: TLS/encryption
- **#63**: Authentication

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
| `gateway/api.rs` | HTTP router builder |
| `gateway/jobs.rs` | Job CRUD handlers |
| `gateway/router.rs` | Chat completions handler |
| `scheduler/scheduler.rs` | Async scheduler with JoinSet |
| `scheduler/metrics.rs` | MetricsCollector with Prometheus export |
| `transport/tcp.rs` | TCP transport for multi-machine |
| `native_moe/generate.rs` | Multi-token generation |
| `tests/e2e.rs` | E2E integration tests |
| `docs/deployment.md` | Deployment guide |

## Test Counts

- Job module: 52 tests
- Scheduler module: 48 tests
- Gateway: 25 tests
- E2E: 8 tests
- Other: 243 tests
- **Total: 376 tests passing**

## Environment

- **GPU:** NVIDIA RTX 4070 Laptop (8GB VRAM)
- **Models:** granite3.1-moe:3b, qwen3:8b (via Ollama)
- **Ollama:** localhost:11434 (and optionally :11435 for multi-instance)
