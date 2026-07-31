<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/status-validating%20thesis-orange" alt="Status: Validating thesis">
  <img src="https://img.shields.io/badge/rust-1.97%2B-orange" alt="Rust 1.97+">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/coverage-80%25%2B-brightgreen" alt="Coverage 80%+">
  <img src="https://img.shields.io/badge/CI-gauntlet%20passed-brightgreen" alt="Gauntlet passed">
</p>

# Synapse

**Decentralized inference for Mixture-of-Experts models — validating the thesis.**

> *Can a small network of heterogeneous GPUs complete inference jobs reliably, verifiably, and more affordably than a centralized alternative? We're testing that.*

---

## Where we are (July 2026)

Synapse has pivoted to **V0: a permissioned async job network for batch inference**. The original P2P vision is deferred until the core coordination is validated. See [ADR-0001](docs/adr/0001-v0-permissioned-async-job-network.md).

### V0 Progress

| Issue | Status | Description |
|---|---|---|
| #20 | ✅ Closed | Native MoE forward pass — correlation 0.999 with llama.cpp |
| #21 | ✅ Closed | Job Model + Async API — POST/GET /v1/jobs, 60 tests |
| #22 | ✅ Closed | Scheduler Mínimo — round-robin, leases (30s), retries (max 3), 44 tests |
| #23 | 🔜 Next | Multi-Worker + Crash Recovery |
| #24 | Planned | Métricas E2E |
| #25 | Planned | Native MoE Runtime — InferencePort Validation |

### Validated (pre-pivot)

| Claim | Evidence |
|---|---|
| Rust ↔ Python worker via Unix socket + protobuf | [Spike](docs/superpowers/spikes/2026-07-28-vllm-viability-spike.md) — 100% success, <2ms overhead |
| Real GPU inference through the pipeline | Qwen3 8B + Ollama — 100% success, 40 tok/s |
| MoE model through the pipeline | Granite 3.1 MoE 3B — 40 experts, 8 active, 100% success |
| Native MoE forward pass | Correlation 0.999334 with llama-cpp-python (single token) |
| Expert sharding | Granite MoE split into 2 shards, loads and generates in Ollama |
| Coordinator routing | gate_inp weights + hidden state = routing decision, 8/8 experts identical |

---

## The vision (where we're going)

MoE (Mixture-of-Experts) models like Kimi K3, DeepSeek-V2, and Mixtral activate only ~1-2% of their parameters per token. The other 98% sits idle.

**Synapse exploits this.** Instead of cramming the entire model into datacenter GPUs, the swarm distributes experts across consumer hardware. Each node holds a handful of experts. Requests flow through the swarm, activating only the experts they need.

The pitch:
- No API keys, no rate limits, no regional blocks
- Any open-weight MoE model
- Consumer GPUs instead of datacenter H100s
- Censorship-resistant P2P mesh

But first: **prove the core works.** See [pivot plan](docs/superpowers/spikes/2026-07-28-vllm-viability-spike.md).

---

## What actually runs today

```bash
git clone https://github.com/antonygiomarxdev/synapse.git
cd synapse

# Build
cargo build --release

# Run all tests
cargo test --lib -- --skip native_moe

# Start the gateway
cargo run --release
# → Gateway on http://0.0.0.0:8000
```

### What `cargo run` actually does

The gateway serves:
- `GET /health` → `{"status": "ok"}`
- `GET /v1/models` → model catalog
- `POST /v1/jobs` → submit async inference job (202 Accepted)
- `GET /v1/jobs/{id}` → poll job status/result
- `GET /swagger-ui/` → OpenAPI documentation

The job API is real — it creates jobs, stores them, and returns status. The scheduler dispatches tasks to workers with round-robin, leases, and retries. Worker integration (actual inference) is the next step (#23).

---

## Current architecture

```
┌──────────────────────────────────────────────────────────┐
│              SYNAPSE NODE (Rust binary)                  │
│                                                          │
│  ┌──────────────────────┐                                │
│  │  Gateway (axum)      │  ← /health, /v1/models        │
│  │  /v1/jobs            │  ← POST (create), GET (poll)  │
│  │  /swagger-ui/        │  ← OpenAPI docs               │
│  └──────────┬───────────┘                                │
│             │                                            │
│  ┌──────────▼───────────┐                                │
│  │  Scheduler           │  ← round-robin, leases, retry │
│  │  decompose → tick    │     30s timeout, max 3 retries│
│  └──────────┬───────────┘                                │
│             │                                            │
│  ┌──────────▼───────────┐                                │
│  │  WorkerPort (trait)  │  ← dispatch to inference      │
│  │  MockWorker (V0)     │     Real workers in V0-3      │
│  └──────────────────────┘                                │
│                                                          │
│  ┌──────────────────────┐                                │
│  │  JobStore / TaskStore│  ← in-memory for V0           │
│  └──────────────────────┘                                │
└──────────────────────────────────────────────────────────┘
```

---

## Tech stack

| Layer | Language | What's built |
|---|---|---|
| Gateway | Rust (axum) | Health, models, job CRUD, OpenAPI/Swagger |
| Job domain | Rust | JobId, JobStatus, Job aggregate, JobStore port |
| Scheduler | Rust | TaskId, Task, Scheduler, round-robin, leases, retries |
| Task domain | Rust | TaskStatus, WorkerId, TaskStore, WorkerPort |
| Inference runtime | Python | vLLM, Ollama, Mock backends (V0-3 integration) |
| Identity | Rust | NodeId (SHA-256 of Ed25519) |
| Model catalog | Rust | ModelId, ModelEntity |
| Native MoE | Rust | GGUF parser, forward pass (correlation 0.999) |

---

## Project structure

```
synapse/
├── synapse-core/            # Rust — single crate, single binary
│   ├── src/
│   │   ├── main.rs          #   Binary entrypoint (axum server)
│   │   ├── gateway/         #   axum HTTP: api, jobs, catalog, router
│   │   ├── job/             #   Job domain: JobId, JobStatus, Job, JobStore
│   │   │   └── infrastructure/  # InMemoryJobStore
│   │   ├── scheduler/       #   Scheduler: Task, TaskStatus, WorkerPort
│   │   │   └── infrastructure/  # InMemoryTaskStore, MockWorkerPort
│   │   ├── identity/        #   NodeId, KeyPair, Node
│   │   ├── model/           #   ModelId, ExpertId, Catalog
│   │   ├── native_moe/      #   Native MoE runtime (forward pass validated)
│   │   ├── swarm/           #   Consensus, Speculative engine, DAG engine
│   │   ├── economic/        #   Reputation, Pricing, Stake management
│   │   ├── transport/       #   WebRTC, Signalling
│   │   ├── runtime/         #   InferencePort trait + Unix socket bridge
│   │   ├── shared/          #   DomainError, DomainEvent
│   │   └── dht/             #   Kademlia, Expert registry
│   └── proto/               #   Protobuf schemas
├── synapse-runtime/         # Python — vLLM adapter (subprocess)
├── contracts/stake/         # Solidity — StakeManager
├── config/                  # models.toml, default.toml
├── docs/
│   ├── adr/                 #   Architecture decision records
│   ├── superpowers/         #   Design specs + spike docs
│   └── next-session-context.md
└── features/                #   Gherkin BDD specs
```

---

## The quality gauntlet

Every line of code — whether written by humans, Claude, Kimi, or a hamster — must pass the same gauntlet before merging:

```bash
make gauntlet   # Runs: fmt, lint, test, coverage, mutants, audit, BDD
```

| Gate | Tool | Threshold |
|---|---|---|
| Formatting | rustfmt, ruff | Must match exactly |
| Linting | clippy -D warnings, ruff | Zero warnings |
| Unit tests | cargo test, pytest | All green |
| Coverage | cargo-llvm-cov | ≥80% lines, ≥80% functions |
| Mutation testing | cargo-mutants | All mutants killed |
| Security audit | cargo-audit, cargo-deny | Zero CVEs |
| BDD specs | Gherkin (features/) | All scenarios pass |
| Smart contracts | hardhat test, solhint | All green |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All changes go through PR with gauntlet passing.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

*Last updated: 2026-07-31. V0-1 (Job Model) and V0-2 (Scheduler) complete. Next: V0-3 (Multi-Worker + Crash Recovery).*
