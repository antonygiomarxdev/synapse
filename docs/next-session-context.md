# Next Session Context: V0-1 + V0-2 Complete, Next: V0-3

## Quick Start

```bash
cd /home/ksante/dev/synapse
git checkout main && git pull
cargo test --lib -- --skip native_moe
```

## Current State

**V0 Pivot:** Permissioned async job network for batch inference. See [ADR-0001](adr/0001-v0-permissioned-async-job-network.md).

**V0-1: ✅ CLOSED** — Job Model + Async API (PR #27 merged)
- `POST /v1/jobs` → 202 Accepted + job_id
- `GET /v1/jobs/{id}` → status, result, error
- OpenAI-compatible format, OpenAPI spec, Swagger UI
- 60 tests (47 domain + 11 gateway + 3 API)

**V0-2: ✅ CLOSED** — Scheduler Mínimo (PR #28 merged)
- Scheduler: decompose job → tasks, tick(), round-robin dispatch
- Leases: 30s timeout per task
- Retries: max 3 per task, then job fails
- 44 tests (TaskId, TaskStatus, Task, TaskStore, WorkerPort, Scheduler)

**V0-3: 🔜 NEXT** — Multi-Worker + Crash Recovery (Issue #23)
- 2 workers via Ollama (granite3.1-moe:3b + qwen3:8b)
- Round-robin dispatch between workers
- Crash test: kill worker-0 mid-job → worker-1 rescues
- Heartbeat: workers → coordinator
- Acceptance: 50 jobs with 2 workers → ≥95% success, crash recovery <30s

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
   Worker (MockWorker for V0, real in V0-3)
```

## Key Files

| File | Role |
|------|------|
| `synapse-core/src/job/job.rs` | Job aggregate: submit, transition_to, complete, fail |
| `synapse-core/src/job/ports.rs` | JobStore trait |
| `synapse-core/src/job/infrastructure/in_memory_job_store.rs` | In-memory store |
| `synapse-core/src/scheduler/scheduler.rs` | Scheduler: decompose, tick, round-robin |
| `synapse-core/src/scheduler/task.rs` | Task aggregate with leases and retries |
| `synapse-core/src/scheduler/ports.rs` | TaskStore + WorkerPort traits |
| `synapse-core/src/scheduler/infrastructure/mock_worker_port.rs` | Mock worker for testing |
| `synapse-core/src/gateway/jobs.rs` | HTTP handlers + AppState |
| `synapse-core/src/gateway/api.rs` | Router builder + OpenAPI |
| `synapse-core/src/shared/domain_error.rs` | All domain errors |

## Issues

| # | Status | Description |
|---|--------|-------------|
| #20 | ✅ | Native MoE forward pass |
| #21 | ✅ | Job Model + Async API |
| #22 | ✅ | Scheduler Mínimo |
| #23 | 🔜 | Multi-Worker + Crash Recovery |
| #24 | Planned | Métricas E2E |
| #25 | Planned | Native MoE — InferencePort Validation |
| #26 | Open | Fix clippy warnings (native_moe naming) |

## Test Counts

- Job module: 49 tests
- Scheduler module: 44 tests
- Gateway: 14 tests
- Other (identity, swarm, economic, etc.): 234 tests
- **Total: 341 tests passing** (native_moe excluded — OOMs in parallel)

## What NOT to Do

1. Don't re-implement V0-1 or V0-2 — already done and merged
2. Don't modify native_moe — it works, clippy warnings are tracked in #26
3. Don't add P2P/DHT/payments — deferred to V1+
4. Don't skip TDD — write tests first for V0-3
