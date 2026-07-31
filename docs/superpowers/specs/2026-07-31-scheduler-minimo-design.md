# V0-2: Scheduler Mínimo — Design Spec

**Date:** 2026-07-31
**Issue:** #22
**Status:** Approved

## Overview

Add a scheduler that picks up `Pending` jobs, decomposes them into tasks, dispatches tasks to known workers via round-robin, handles leases (timeouts), and retries failed tasks on other workers. Jobs complete when all their tasks complete; they fail after 3 retries.

## Architecture

### Domain Layer — `synapse-core/src/scheduler/`

New module alongside `job/`, following the same DDD patterns.

- **`TaskId`** — UUID wrapper value object (like JobId).
- **`WorkerId`** — String wrapper value object.
- **`TaskStatus`** — Enum: `Pending | Leased | Completed | Failed`.
- **`Task`** — Aggregate: `id`, `job_id`, `model`, `message`, `status`, `retry_count`, `created_at`.
- **`Lease`** — Value object: `task_id`, `worker_id`, `granted_at`, `expires_at`.
- **`WorkerInfo`** — Value object: `id`, `model`, `healthy`.
- **`TaskStore`** — Port trait: `save`, `find_by_id`, `find_pending`, `claim`, `complete`, `fail`.
- **`WorkerPort`** — Port trait: `dispatch(&self, task: &Task) -> Result<String, DomainError>` (returns generated text).
- **`Scheduler`** — Orchestrator: `decompose(job)`, `tick()`, `dispatch_pending()`.

### Scheduler Logic

```
tick():
  1. Find all pending tasks
  2. For each: round-robin pick a healthy worker
  3. Dispatch task to worker → lease it (30s deadline)
  4. If lease expired → fail task, retry on next worker
  5. If retry_count >= 3 → mark task failed, check if job should fail
  6. If all tasks completed → mark job completed
```

### Lease Model

- Default lease: 30 seconds
- Lease stored on Task (not separate entity for V0)
- `tick()` checks `expires_at < now` to detect timeouts

### Retry Logic

- Max 3 retries per task
- After each failure: `retry_count++`, re-dispatch to next worker in round-robin
- After 3 failures: task → Failed, check job failure

### Infrastructure

- **`InMemoryTaskStore`** — `Mutex<HashMap<TaskId, Task>>`
- **`MockWorkerPort`** — For testing: configurable success/failure/timeout

### Gateway Integration

- `AppState` grows: `scheduler: Arc<Scheduler>`
- `POST /v1/jobs` saves job, scheduler picks it up via `tick()`
- Scheduler runs as tokio task, polling every 100ms

### Ports Reused

- `JobStore` — scheduler reads/writes jobs
- `Job::transition_to` — scheduler owns Pending→Running, Running→Completed/Failed

## Domain Model

```
Task {
    id: TaskId,
    job_id: JobId,
    model: String,
    message: Message,
    status: TaskStatus,
    retry_count: u32,
    worker_id: Option<WorkerId>,
    lease_expires_at: Option<DateTime<Utc>>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}
```

## Non-Goals

- Dynamic pricing
- DAG routing
- Auto-scaling
- Real worker processes (V0-3)

## Test Strategy (TDD)

1. Domain: TaskId, TaskStatus, Task creation, lease grant/expiry
2. Port contract: InMemoryTaskStore
3. Scheduler logic: decompose, tick, dispatch, retry, failure
4. Gateway integration: POST job → tasks created → job completes
