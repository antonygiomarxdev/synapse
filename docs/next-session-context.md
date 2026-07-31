# Next Session Context: V0-1, V0-2, V0-3 Complete — Next: V0-4

## Quick Start

```bash
cd /home/ksante/dev/synapse
git checkout main && git pull
cargo test --lib -- --skip native_moe --skip health_check_localhost
```

## Current State

**V0 Pivot:** Permissioned async job network for batch inference. See [ADR-0001](adr/0001-v0-permissioned-async-job-network.md).

**V0-1: ✅ CLOSED** — Job Model + Async API (PR #27 merged)
- POST /v1/jobs → 202, GET /v1/jobs/{id} → status/result
- OpenAPI spec, Swagger UI, 60 tests
- All 4 acceptance criteria verified

**V0-2: ✅ CLOSED** — Scheduler Mínimo (PR #28 merged)
- Scheduler: decompose → tick → round-robin → lease → retry
- 44 tests, all 4 acceptance criteria verified (including idempotency)

**V0-3: ✅ CLOSED** — Multi-Worker + Crash Recovery (PR #29, #30 merged)
- OllamaWorkerPort (HTTP to Ollama), MetricsCollector
- 7 integration tests, all 7 acceptance criteria verified
- Crash recovery <30s wall-clock verified

**V0-4: 🔜 NEXT** — Métricas E2E (Issue #24)

## V0-4 Spec (from V0-issues.md)

Publicar un benchmark que compare: 1 nodo local vs 2 workers coordinados vs 2 workers con fallo inducido.

**Scope:**
- `MetricsCollector`: success_rate, retry_rate, queue_time_ms, execution_time_ms, tokens_total, cost_per_1m_tokens
- Script `scripts/bench.sh` que ejecuta el benchmark y produce un reporte
- Reporte markdown con tabla comparativa
- Publicar en `docs/benchmarks/v0-<date>.md`

**Acceptance:**
- Benchmark reproducible con un solo comando
- Tabla comparativa con las 3 configuraciones
- Métricas incluyen p50/p95/p99 para latencia

## Architecture (current)

```
Gateway (axum) → POST/GET /v1/jobs
       ↓
   JobStore (in-memory)
       ↓
   Scheduler → decompose → TaskStore
       ↓
   tick() → round-robin → WorkerPort.dispatch()
       ↓
   OllamaWorkerPort / MockWorkerPort
       ↓
   MetricsCollector (jobs, tasks, retries, latencies)
```

## Key Files

| File | Role |
|------|------|
| `synapse-core/src/job/job.rs` | Job aggregate |
| `synapse-core/src/scheduler/scheduler.rs` | Scheduler with MetricsCollector |
| `synapse-core/src/scheduler/metrics.rs` | MetricsCollector + MetricsReport |
| `synapse-core/src/scheduler/infrastructure/ollama_worker_port.rs` | Real Ollama HTTP client |
| `synapse-core/src/scheduler/integration_tests.rs` | 7 acceptance tests |
| `synapse-core/src/gateway/jobs.rs` | HTTP handlers + AppState |

## Issues

| # | Status | Description |
|---|--------|-------------|
| #20–#23 | ✅ | All closed |
| #24 | 🔜 | Métricas E2E |
| #25 | Planned | Native MoE — InferencePort Validation |
| #26 | Open | Fix clippy warnings (native_moe naming) |

## Test Counts

- Job module: 52 tests
- Scheduler module: 48 tests
- Gateway: 14 tests
- Other: 243 tests
- **Total: 357 tests passing**

## What NOT to Do

1. Don't re-implement V0-1, V0-2, V0-3 — done and merged
2. Don't modify native_moe — tracked in #26
3. Don't add P2P/DHT/payments — deferred to V1+
4. Don't skip TDD
