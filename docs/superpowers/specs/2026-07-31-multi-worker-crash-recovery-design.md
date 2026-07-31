# V0-3: Multi-Worker + Crash Recovery — Design Spec

**Date:** 2026-07-31
**Issue:** #23
**Status:** Approved

## Overview

Connect the scheduler to real Ollama workers, add heartbeat-based health checking, and validate crash recovery: when one worker dies mid-job, the other rescues the tasks.

## Architecture

### OllamaWorkerPort

New `WorkerPort` implementation that sends HTTP requests to Ollama's `/api/generate` endpoint.

- Uses `reqwest` (already in dev-dependencies, promoted to regular)
- Configurable base URL (default: `http://localhost:11434`)
- Maps `WorkerId` → model name via configuration
- `dispatch()` sends prompt, waits for response, returns generated text
- Timeout: 120s per request

### Health Check

Extend `WorkerPort` trait with `health_check(&self, worker_id: &WorkerId) -> Result<bool>`.

- Calls Ollama's `GET /api/tags` to verify the worker is alive
- Scheduler calls health check before dispatch
- Unhealthy workers are skipped in round-robin
- Background health check task updates `WorkerInfo.healthy` every 10s

### MetricsCollector

New struct that collects inference metrics:

```rust
struct MetricsCollector {
    total_jobs: AtomicU64,
    completed_jobs: AtomicU64,
    failed_jobs: AtomicU64,
    total_tasks: AtomicU64,
    retried_tasks: AtomicU64,
    queue_times_ms: Mutex<Vec<u64>>,
    execution_times_ms: Mutex<Vec<u64>>,
}
```

Methods: `record_job_submit()`, `record_job_complete()`, `record_job_fail()`, `record_task_dispatch(queue_ms, exec_ms)`, `record_task_retry()`, `report() -> MetricsReport`.

### Gateway Integration

- `AppState` gets `metrics: Arc<MetricsCollector>`
- New endpoint: `GET /v1/metrics` → JSON metrics report
- Scheduler wired with real OllamaWorkerPort

### Worker Configuration

```rust
struct WorkerConfig {
    id: WorkerId,
    model: String,
    base_url: String, // e.g., "http://localhost:11434"
}
```

Two workers:
- `worker-0`: `granite3.1-moe:3b` at `http://localhost:11434`
- `worker-1`: `qwen3:8b` at `http://localhost:11434`

### Crash Recovery Test

1. Submit job with 10 messages
2. After 5 tasks dispatched, kill Ollama (simulate by stopping the process or using a flaky port)
3. Verify remaining tasks are reassigned to the other worker
4. Verify job eventually completes

For V0, crash simulation uses MockWorkerPort with configurable failure injection rather than killing real Ollama processes.

## Non-Goals

- More than 2 workers
- Real multi-machine networking
- Load balancing beyond round-robin
- Persistent metrics storage

## Test Strategy

1. Unit: OllamaWorkerPort with real Ollama (smoke test)
2. Unit: Health check mechanism
3. Unit: MetricsCollector
4. Integration: 50 jobs with 2 mock workers → ≥95% success
5. Integration: Crash recovery with mock worker failure injection

## Dependencies

- `reqwest` (promoted from dev to regular dependency)
