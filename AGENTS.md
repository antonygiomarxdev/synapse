# Repository Guidelines

## Project Overview

Synapse is a **decentralized P2P inference protocol for Mixture-of-Experts (MoE) models**. It coordinates consumer GPUs into a distributed inference swarm — miners contribute idle hardware, clients consume frontier AI through an OpenAI-compatible API. No datacenter, no gatekeeper, no rate limits.

**Multi-language monorepo:** Rust core (P2P + gateway), Python vLLM runtime (subprocess), Solidity staking contracts (L2).

## Architecture & Data Flow

```
Client → axum Gateway (Rust, :8000) → Swarm Core (Rust)
                                          │
                                    libp2p Kademlia DHT
                                          │
                                    Compute Nodes (Python vLLM)
                                          │
                                    L2 Smart Contracts (Solidity)
```

- **Gateway** (`synapse-core/src/gateway/`): axum HTTP server with OpenAI-compatible endpoints. Market maker pricing. Routes requests to swarm.
- **Swarm Core** (`synapse-core/src/swarm/`): Consensus (ensemble voting + statistical audit), Speculative engine (realtime), DAG engine (batch).
- **DHT** (`synapse-core/src/dht/`): Kademlia-based expert registry, co-activation heat map, node discovery.
- **Economic** (`synapse-core/src/economic/`): Reputation scoring (0-1000, 4 tiers), graduated slashing, route assembly.
- **Compute Node** (`synapse-runtime/`): Python subprocess communicating via Unix socket + protobuf through `InferencePort` trait. V1: vLLM backend. V2+: llama.cpp, SGLang.
- **Contracts** (`contracts/stake/`): StakeManager.sol — USDC staking, flagging, graduated slashing, banning.

## Non-Negotiable Design Principles

These apply to every line of code. No exceptions.

- **DDD: Pure domain layer** — zero I/O, zero framework deps, zero crypto. Domain types are plain structs/enums. I/O boundaries are traits (ports) in domain modules. Infrastructure adapters live in `infrastructure/` subdirectories.
- **Clean Architecture: Dependencies point inward** — Presentation (axum) → Ports (traits) → Infrastructure (adapters) → Domain. Domain never imports infrastructure.
- **TDD: Red-Green-Refactor** — Write the failing test first, confirm it fails, then implement, then refactor. Tests inline with source at `#[cfg(test)] mod tests`.
- **Clean Code: Every public item gets `///` doc comments.** Test names describe the scenario. No dead code. `thiserror` for errors, never manual `Display`/`Error`. Conventional Commits.

### Two Swarm Modes
- **Speculative Swarm (realtime):** N nodes run full model independently, majority vote per token. Latency = single-node latency.
- **Swarm DAG (batch):** True expert distribution. Nodes hold 2-5 experts each. Requests flow through expert graph.

## Key Directories

```
synapse/
├── synapse-core/            # Rust — single crate, single binary
│   ├── src/
│   │   ├── main.rs          #   Binary entrypoint (axum + swarm + DHT)
│   │   ├── gateway/         #   axum HTTP: api, jobs, catalog, router
│   │   ├── job/             #   Job domain: JobId, JobStatus, Job, JobStore
│   │   │   └── infrastructure/  # InMemoryJobStore
│   │   ├── scheduler/       #   Scheduler: Task, TaskStatus, WorkerPort
│   │   │   └── infrastructure/  # InMemoryTaskStore, MockWorkerPort
│   │   ├── identity/        #   NodeId, KeyPair, Node aggregate
│   │   ├── model/           #   ModelId, ExpertId, Catalog
│   │   ├── native_moe/      #   Native MoE runtime (forward pass validated)
│   │   ├── swarm/           #   Consensus, Speculative engine, DAG engine
│   │   ├── economic/        #   Reputation, Pricing, Stake management
│   │   ├── transport/       #   WebRTC, Signalling
│   │   ├── runtime/         #   InferencePort trait + Unix socket bridge
│   │   ├── shared/          #   DomainError, DomainEvent
│   │   └── dht/             #   Kademlia, Expert registry, Bootstrap
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
│   ├── session-2026-07-30.md        # Session log with all debug details
│   ├── next-session-context.md      # Quick start for next session
│   └── validation-distributed-vs-full.json  # Ensemble voting results
├── features/                #   Gherkin BDD specs
├── scripts/
│   ├── validate_distributed_vs_full.py  # Distributed vs full model comparison
│   └── capture_reference.py             # Reference output capture
├── .github/workflows/       #   CI (7 jobs)
└── docs/superpowers/        #   Design spec + implementation plan
```

## Native MoE Runtime Status

**Location:** `synapse-core/src/native_moe/`

**Status:** Forward pass validated — correlation 0.999 with llama.cpp (Issue #20 resolved)

**What works:**
- GGUF v3 parser (F32, F16, Q8_0, Q4_K, Q6_K)
- Full transformer forward pass (32 layers, 40 experts, GQA attention + RoPE)
- Expert routing (gate_inp → softmax → top-k)
- SiLU activation, RMS norm, residual connections
- Output projection with tied embedding weights
- Single-token inference: correlation 0.999334 with llama-cpp-python

**What doesn't work yet:**
- Multi-token causal attention (untested with KV cache)
- Performance optimization (triple-loop CPU, no SIMD/BLAS)
- Distributed expert execution across nodes
- InferencePort integration with swarm coordinator

**Main ticket:** [#20](https://github.com/antonygiomarxdev/synapse/issues/20) (resolved)

**Key learnings:** See `docs/adr/0011-native-moe-runtime.md`

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
- **Async:** `tokio` (full features). `#[tokio::main]` on binary, `#[tokio::test]` on async tests.
- **Testing:** Unit tests inline with `#[cfg(test)] mod tests`. Integration tests planned in `tests/` directory. Property testing via `proptest`.
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
| `synapse-core/src/scheduler/scheduler.rs` | Scheduler: decompose, tick, round-robin dispatch |
| `synapse-core/src/scheduler/task.rs` | Task aggregate with leases and retries |
| `synapse-core/src/scheduler/ports.rs` | TaskStore + WorkerPort port traits |
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
- **Rust integration tests:** `tests/` directory (planned)
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
