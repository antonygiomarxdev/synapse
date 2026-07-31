# V0-1: Job Model + Async API — Design Spec

**Date:** 2026-07-31
**Issue:** #21
**Status:** Approved

## Overview

Add async job submission to the Synapse gateway. Clients submit inference jobs via `POST /v1/jobs` (returns 202 Accepted), then poll `GET /v1/jobs/{id}` for status and results. All request/response types follow OpenAI-compatible conventions. OpenAPI spec generated via `utoipa` with Swagger UI at `/swagger-ui/`.

## Architecture

### Domain Layer — `synapse-core/src/job/`

Pure domain types, zero I/O, zero framework dependencies.

- **`JobId`** — UUID wrapper value object. `new()`, `as_uuid()`, `Display`, `FromStr`.
- **`JobStatus`** — Enum: `Pending | Running | Completed | Failed`. State machine with validated transitions.
- **`Job`** — Aggregate root. Identity fields immutable after creation (model, messages, priority). Mutable: status, result, error.
- **`Priority`** — Enum: `Low | Normal | High`. Default: `Normal`.
- **`JobStore`** — Port trait: `save`, `find_by_id`, `list`, `update_status`. All return `Result<T, DomainError>`.

### Gateway Layer — `synapse-core/src/gateway/jobs.rs`

Thin HTTP adapter. Handlers delegate to `JobStore` via `AppState`.

- `POST /v1/jobs` → 202 Accepted + `{ job_id }`
- `GET /v1/jobs/{id}` → 200 with job state, or 404

Request format reuses OpenAI message shape (`model`, `messages`, `priority`).

### Infrastructure — `synapse-core/src/job/infrastructure/`

- **`InMemoryJobStore`** — `Mutex<HashMap<JobId, Job>>`. Implements `JobStore`. For V0 only.

### State Injection

`AppState` struct with `Arc<dyn JobStore>` injected via `Router::with_state()`. Resolves the current gap of no shared state in the gateway.

### OpenAPI

- `utoipa` + `utoipa-swagger-ui` crates
- `ToSchema` derives on all request/response types
- `OpenApi` struct aggregating all endpoint paths
- Swagger UI mounted at `/swagger-ui/`

## Domain Model

```
Job {
    id: JobId,
    model: String,
    messages: Vec<Message>,    // { role: String, content: String }
    priority: Priority,
    status: JobStatus,
    result: Option<JobResult>, // { text: String, tokens: u32 }
    error: Option<String>,
    created_at: OffsetDateTime,
    updated_at: OffsetDateTime,
}
```

### Status Transitions

```
Pending → Running
Pending → Failed
Running → Completed
Running → Failed
```

Invalid transitions return `DomainError::InvalidJobTransition`.

## API Contract

### POST /v1/jobs

Request:
```json
{
  "model": "granite-3b-moe",
  "messages": [{ "role": "user", "content": "Hello" }],
  "priority": "normal"
}
```

Response (202):
```json
{ "job_id": "550e8400-e29b-41d4-a716-446655440000" }
```

### GET /v1/jobs/{id}

Response (200):
```json
{
  "job_id": "550e8400-...",
  "object": "job",
  "status": "completed",
  "model": "granite-3b-moe",
  "result": { "text": "Hello! How can I help?", "tokens": 8 },
  "error": null,
  "created_at": "2026-07-31T10:00:00Z",
  "updated_at": "2026-07-31T10:00:05Z"
}
```

### Error Responses

- 400: Invalid payload (missing model, empty messages)
- 404: Job not found

## Non-Goals

- Persistence (SQLite, etc.)
- Job cancellation
- Pagination
- Auth
- Streaming

## Dependencies Added

- `utoipa` — OpenAPI code generation
- `utoipa-swagger-ui` — Swagger UI embedding
- `chrono` — already present, used for timestamps

## Test Strategy (TDD)

1. Domain tests first: JobId, JobStatus transitions, Job creation/validation
2. Port contract tests: InMemoryJobStore against JobStore trait
3. Gateway integration tests: HTTP endpoints via tower::ServiceExt::oneshot
