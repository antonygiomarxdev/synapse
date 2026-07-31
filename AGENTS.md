# Repository Guidelines

## Project Overview

Synapse is **distributed inference infrastructure for Mixture-of-Experts (MoE) models**. It distributes experts across consumer hardware so large MoE models can run without datacenter GPUs.

**Multi-language monorepo:** Rust core (network + gateway), Python vLLM runtime (subprocess), Solidity staking contracts (L2).

## Architecture & Data Flow

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

**Coordinator** runs attention locally (32 layers), dispatches expert FFN to remote workers via HTTP. Workers load only their assigned experts from GGUF.

### Key Components

- **Gateway** (`synapse-core/src/gateway/`): axum HTTP server with OpenAI-compatible endpoints. Routes requests to scheduler.
- **Scheduler** (`synapse-core/src/scheduler/`): Async scheduler with `JoinSet` for concurrent dispatch. Round-robin, leases (30s), retries (max 3).
- **Expert Workers** (`synapse-core/src/bin/expert_worker.rs`): HTTP servers that load expert subsets from GGUF and serve FFN requests.
- **Native MoE** (`synapse-core/src/native_moe/`): GGUF parser, forward pass, expert sharding, distributed inference orchestrator.
- **MetricsCollector** (`synapse-core/src/scheduler/metrics.rs`): Tracks jobs, tasks, retries, latencies (p50/p95/p99).

### Two Network Modes

- **Speculative Network (realtime):** N nodes run full model independently, majority vote per token. Latency = single-node latency.
- **Network DAG (batch):** True expert distribution. Nodes hold 2-5 experts each. Requests flow through expert graph.

## Non-Negotiable Design Principles

These apply to every line of code. No exceptions.

- **DDD: Pure domain layer** — zero I/O, zero framework deps, zero crypto. Domain types are plain structs/enums. I/O boundaries are traits (ports) in domain modules. Infrastructure adapters live in `infrastructure/` subdirectories.
- **Clean Architecture: Dependencies point inward** — Presentation (axum) → Ports (traits) → Infrastructure (adapters) → Domain. Domain never imports infrastructure.
- **TDD: Red-Green-Refactor** — Write the failing test first, confirm it fails, then implement, then refactor. Tests inline with source at `#[cfg(test)] mod tests`.
- **Clean Code: Every public item gets `///` doc comments.** Test names describe the scenario. No dead code. `thiserror` for errors, never manual `Display`/Error`. Conventional Commits.

## Native MoE Runtime Status

**Location:** `synapse-core/src/native_moe/`

**Status:** Distributed inference proven — cosine similarity 1.000000 with monolithic (Issue #25 resolved)

**What works:**
- GGUF v3 parser (F32, F16, Q8_0, Q4_K, Q6_K)
- Full transformer forward pass (32 layers, 40 experts, GQA attention + RoPE)
- Expert routing (gate_inp → softmax → top-k)
- Per-expert GGUF loading (expert_shard.rs)
- Distributed forward pass (coordinator + workers)
- Expert worker HTTP servers
- Benchmark: 2 workers 1.35x, 4 workers 1.43x speedup

**What doesn't work yet:**
- Multi-token causal attention (untested with KV cache)
- Performance optimization (triple-loop CPU, no SIMD/BLAS)
- Dynamic expert loading (load on demand)
- P2P expert discovery

**Main tickets:** [#20](https://github.com/antonygiomarxdev/synapse/issues/20), [#25](https://github.com/antonygiomarxdev/synapse/issues/25) (resolved)

## Key Directories

```
synapse/
├── synapse-core/            # Rust — single crate
│   ├── src/
│   │   ├── main.rs          #   Binary entrypoint (axum server)
│   │   ├── gateway/         #   axum HTTP: api, jobs, catalog, router
│   │   ├── job/             #   Job domain: JobId, JobStatus, Job, JobStore
│   │   │   └── infrastructure/  # InMemoryJobStore
│   │   ├── scheduler/       #   Async scheduler: Task, WorkerPort, MetricsCollector
│   │   │   └── infrastructure/  # InMemoryTaskStore, MockWorkerPort, OllamaWorkerPort
│   │   ├── native_moe/      #   MoE runtime: forward pass, expert sharding
│   │   │   ├── expert_shard.rs     # Per-expert GGUF loader
│   │   │   ├── expert_worker_client.rs  # HTTP client for remote FFN
│   │   │   ├── distributed_forward.rs   # Distributed inference orchestrator
│   │   │   └── forward.rs          # Monolithic forward pass
│   │   ├── identity/        #   NodeId, KeyPair, Node aggregate
│   │   ├── model/           #   ModelId, ExpertId, Catalog
│   │   ├── swarm/           #   Consensus, Speculative engine, DAG engine
│   │   ├── economic/        #   Reputation, Pricing, Stake management
│   │   ├── transport/       #   WebRTC, Signalling
│   │   ├── runtime/         #   InferencePort trait + Unix socket bridge
│   │   ├── shared/          #   DomainError, DomainEvent
│   │   └── dht/             #   Kademlia, Expert registry, Bootstrap
│   ├── bin/
│   │   ├── expert_worker.rs    # Expert worker HTTP server
│   │   ├── bench_distributed.rs # Distributed vs monolithic benchmark
│   │   ├── bench_ollama.rs     # Ollama throughput benchmark
│   │   └── bench_consistency.rs # Output consistency test
│   └── proto/               #   Protobuf schemas (8 message types)
├── synapse-runtime/         # Python — vLLM adapter (subprocess)
│   └── synapse_runtime/     #   Package source
├── contracts/stake/         # Solidity — StakeManager + Hardhat
│   ├── src/                 #   StakeManager.sol
│   └── test/                #   Hardhat tests
├── config/
│   ├── models.toml          #   Curated catalog (Kimi K3, Mixtral, etc.)
│   └── default.toml         #   Node defaults (VRAM, pricing, STUN)
├── docs/
│   ├── adr/                 #   Architecture decision records
│   ├── benchmarks/          #   Benchmark reports
│   ├── superpowers/         #   Design specs + spike docs
│   └── next-session-context.md
├── features/                #   Gherkin BDD specs
├── scripts/
│   ├── validate_distributed_vs_full.py  # Distributed vs full model comparison
│   └── capture_reference.py             # Reference output capture
├── .github/workflows/       #   CI (7 jobs)
└── docs/superpowers/        #   Design spec + implementation plan
```

## Development Commands

```bash
# Rust
cargo build --release              # Build single binary
cargo test                         # Run all Rust tests
cargo fmt --check                  # Check formatting
cargo clippy -- -D warnings        # Lint
cargo llvm-cov --fail-under-lines 80  # Coverage check
cargo mutants -- --workspace       # Mutation testing
cargo deny check                   # License + dependency audit
cargo audit                        # CVE check

# Python
cd synapse-runtime && ruff check . && ruff format --check .
cd synapse-runtime && python -m pytest tests/ -v
cd synapse-runtime && pip-audit

# Solidity
cd contracts/stake && npx hardhat compile && npx hardhat test
cd contracts/stake && npx solhint 'src/**/*.sol'

# Benchmarks
cargo run --release --bin bench_distributed  # Distributed vs monolithic
cargo run --release --bin bench_ollama       # Ollama throughput
cargo run --release --bin bench_consistency  # Output consistency

# Expert workers
cargo run --release --bin expert_worker -- model.gguf 0 1 2 --port 8001

# Everything at once (PR gate)
make gauntlet
```

## Code Conventions & Common Patterns

### Rust

- **Edition:** 2024, pinned to Rust 1.97 via `rust-toolchain.toml`
- **Formatting:** `rustfmt.toml` — max_width 100, 4-space indent, reorder_imports
- **Linting:** `-D warnings` enforced in CI. `clippy.toml` allows `unwrap`/`dbg!` only in tests.
- **Naming:** snake_case files, CamelCase types (e.g., `NodeId`, `StakeManager`). Module names match directory names.
- **Error handling:** `thiserror` for domain errors. `Result<Json<T>, StatusCode>` pattern in axum handlers.
- **Async:** `tokio` (full features). `#[tokio::main]` on binary, `#[tokio::test]` on async tests. `async-trait` for async traits.
- **Testing:** Unit tests inline with `#[cfg(test)] mod tests`. Integration tests in `tests/` directory. Property testing via `proptest`.
- **Protobuf:** `synapse.proto` defines 8 message types (DhtQuery, NodeAnnounce, InferenceRequest, ConsensusVote, etc.). Package: `synapse.proto`.

### Python

- **Package:** `synapse-runtime` v0.1.0, requires Python ≥3.12
- **Linting:** `ruff` with strict ruleset (E, F, W, I, N, UP, B, SIM, C4, RUF). Double quotes, space indent.
- **Testing:** `pytest` with `pytest-asyncio` and `pytest-mock`.

### Solidity

- **Version:** 0.8.36 (pragma in contract is `^0.8.28`, pinned in Hardhat config)
- **Linting:** `solhint` with recommended + reentrancy, visibility, no-empty-blocks rules.
- **Testing:** Hardhat with `@nomicfoundation/hardhat-toolbox`.
- **Pattern:** Single `StakeManager` contract with modifiers (`onlyAuthorized`, `notBanned`, `notFrozen`), graduated penalties (10 flags → freeze 48h, 50 flags → slash 20%).

## Important Files

| File | Role |
|---|---|
| `synapse-core/src/main.rs` | Binary entrypoint — starts axum server |
| `synapse-core/src/lib.rs` | Library root — declares public modules |
| `synapse-core/src/gateway/api.rs` | HTTP router builder (`build_router()`) + OpenAPI |
| `synapse-core/src/gateway/jobs.rs` | Job CRUD handlers + AppState |
| `synapse-core/src/job/job.rs` | Job aggregate: submit, transition_to, complete, fail |
| `synapse-core/src/job/ports.rs` | JobStore port trait |
| `synapse-core/src/scheduler/scheduler.rs` | Async scheduler with JoinSet |
| `synapse-core/src/scheduler/task.rs` | Task aggregate with leases and retries |
| `synapse-core/src/scheduler/ports.rs` | TaskStore + WorkerPort port traits |
| `synapse-core/src/native_moe/expert_shard.rs` | Per-expert GGUF loader |
| `synapse-core/src/native_moe/distributed_forward.rs` | Distributed inference orchestrator |
| `synapse-core/src/native_moe/forward.rs` | Monolithic forward pass |
| `synapse-core/src/bin/expert_worker.rs` | Expert worker HTTP server |
| `synapse-core/src/bin/bench_distributed.rs` | Distributed vs monolithic benchmark |
| `synapse-core/src/identity/node_id.rs` | NodeId value object |
| `synapse-core/src/gateway/router.rs` | Chat completions handler (OpenAI-compatible) |
| `synapse-core/proto/synapse.proto` | Wire protocol — all inter-component messages |
| `config/models.toml` | Curated model catalog |
| `config/default.toml` | Node configuration defaults |
| `contracts/stake/src/StakeManager.sol` | L2 staking + slashing contract |
| `features/swarm.feature` | BDD behavioral contracts (14 scenarios) |
| `Makefile` | All build/test/lint/audit targets + `gauntlet` |
| `.github/workflows/ci.yml` | CI pipeline (7 jobs) |
| `rust-toolchain.toml` | Pins Rust 1.97 |
| `deny.toml` | License + security audit rules |
| `coverage.toml` | 80% line + function coverage thresholds |

## Runtime/Tooling Preferences

- **Rust:** 1.97+ (pinned), edition 2024, cargo as build tool
- **Python:** 3.12+, `pip` for deps, `ruff` for lint/format
- **Solidity:** 0.8.36, Hardhat, `npm` for deps
- **CI:** GitHub Actions (7 parallel jobs). Must pass `gauntlet` before merge.
- **Pre-commit:** `.pre-commit-config.yaml` — trailing-whitespace, end-of-file-fixer, yaml/toml checks, rustfmt, clippy, ruff
- **No TURN servers in V1** — STUN only (~10% miner exclusion accepted)

## Testing & QA

### The Quality Gauntlet (PR gate)

Every PR must pass all of these before merge:

| Gate | Command | Threshold |
|---|---|---|
| Format | `cargo fmt --check` + `ruff format --check` | Exact match |
| Lint | `cargo clippy -- -D warnings` + `ruff check` | Zero warnings |
| Unit tests | `cargo test` + `pytest` + `hardhat test` | All green |
| Coverage | `cargo llvm-cov` | ≥80% lines, ≥80% functions |
| Mutation | `cargo mutants -- --workspace` | All mutants killed |
| Security | `cargo audit` + `cargo deny check` + `pip-audit` | Zero CVEs, licenses OK |
| BDD | Gherkin scenarios in `features/` | All pass |
| Contracts | `hardhat test` + `solhint` | All green |

Run locally: `make gauntlet`

### Testing Patterns

- **Rust unit tests:** Inline with `#[cfg(test)] mod tests` in same file as source
- **Rust integration tests:** `tests/` directory
- **Python:** `pytest` in `synapse-runtime/tests/`
- **Solidity:** Hardhat tests in `contracts/stake/test/`
- **Property tests:** `proptest` crate for domain logic (e.g., consensus voting, pricing)
- **BDD:** Gherkin `.feature` files in `features/` directory — behavior contracts, not implementation

### Philosophy
> *"I don't read my agents' code. I surround them with extreme constraints."* — Uncle Bob

Code can be written by humans, Claude, Kimi, or any agent. The gauntlet is the gatekeeper.

## Agent skills

### Issue tracker

Issues live in GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five canonical triage roles with default label names. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo: `CONTEXT.md` at root + `docs/adr/`. See `docs/agents/domain.md`.
