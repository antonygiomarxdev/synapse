<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/status-V0--5%20in%20progress-orange" alt="Status: V0-5 in progress">
  <img src="https://img.shields.io/badge/rust-1.97%2B-orange" alt="Rust 1.97+">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/coverage-80%25%2B-brightgreen" alt="Coverage 80%+">
  <img src="https://img.shields.io/badge/CI-gauntlet%20passed-brightgreen" alt="Gauntlet passed">
</p>

# Synapse

**Distributed inference infrastructure for Mixture-of-Experts models.**

MoE models activate only ~2% of their parameters per token. Synapse was built to exploit that — distributing experts across consumer hardware so you can run large models without large hardware.

---

## Why Synapse

Mixture-of-Experts models (Kimi K3, DeepSeek-V2, Mixtral, Qwen2.5-MoE) are architecturally designed for distribution: each token activates only a handful of independent experts, not the full network. Yet today, running them still requires datacenter-grade GPUs because the inference stack treats them like dense models.

Synapse is built specifically for MoE. It handles expert sharding and routing at the infrastructure level — distributing experts across the hardware you already have instead of cramming the entire model into expensive GPUs.

**The result:** run a 57B-parameter MoE model on consumer GPUs that cost a fraction of a single H100.

> **Note:** Synapse is in active development. The current focus is batch inference — the use case where distributed access shines most. Realtime inference follows.

---

## Supported models

| Model | Experts | Active/Token | Expert Size | Context |
|---|---|---|---|---|
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) | 896 | 16 | ~1.5 GB | 1M |
| [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | 8 | 2 | ~3.0 GB | 32K |
| [DeepSeek-V2 Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) | 64 | 6 | ~0.15 GB | 131K |
| [Qwen2.5-MoE](https://huggingface.co/Qwen/Qwen2.5-MoE-57B-A14B) (57B-A14B) | 64 | 8 | ~0.5 GB | 32K |

Models are curated in [`config/models.toml`](config/models.toml). Community proposals welcome via PR.

---

## Two network modes

Synapse supports two modes of distributed inference:

**Speculative Network** (realtime)
Multiple nodes each run the full model independently with different random seeds. A coordinator votes per token — majority wins. Latency equals single-node latency. Designed for chat, autocomplete, and interactive workloads.

**Network DAG** (batch)
Each node holds a subset of experts (2-5). Requests flow through an expert graph — only the activated experts process each token. Multiple requests pipeline simultaneously. Designed for batch processing, dataset analysis, and CI/CD workloads.

---

## Current status

### V0: Permissioned async job network

| Issue | Status | Description |
|---|---|---|
| #20 | ✅ | Native MoE forward pass — correlation 0.999 with llama.cpp |
| #21 | ✅ | Job Model + Async API — POST/GET /v1/jobs, 60 tests |
| #22 | ✅ | Scheduler — round-robin, leases (30s), retries (max 3), 44 tests |
| #23 | ✅ | Multi-Worker + Crash Recovery — OllamaWorkerPort, 7 integration tests |
| #24 | ✅ | E2E Metrics — benchmark binary, p50/p95/p99 report |
| #25 | In progress | Native MoE Runtime — InferencePort Validation |

### Validated claims

| Claim | Evidence |
|---|---|
| Native MoE forward pass | [Correlation 0.999](docs/adr/0011-native-moe-runtime.md) with llama-cpp-python |
| Expert sharding | [Granite MoE split into 2 shards](docs/superpowers/spikes/2026-07-29-expert-sharding-spike.md), loads and generates |
| GPU inference pipeline | [Qwen3 8B via Ollama](docs/superpowers/spikes/2026-07-28-vllm-viability-spike.md) — 40 tok/s |
| Rust ↔ Python bridge | Unix socket + protobuf — <2ms overhead |

---

## Architecture

```
                    ┌─────────────┐
                    │   Client    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Gateway   │  axum HTTP
                    │   :8000     │  /health, /v1/models
                    │             │  /v1/jobs, /swagger-ui
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Scheduler  │  round-robin, leases, retries
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼───┐ ┌─────▼─────┐
        │  Node A   │ │Node B │ │  Node C   │
        │ experts:  │ │experts│ │ experts:  │
        │ 3,17,42   │ │ 8,91  │ │ 5,23,67   │
        └───────────┘ └───────┘ └───────────┘
```

---

## Quick start

```bash
git clone https://github.com/antonygiomarxdev/synapse.git
cd synapse
cargo build --release
cargo test --lib -- --skip native_moe
cargo run --release
# → Gateway on http://0.0.0.0:8000
```

### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/v1/models` | GET | Model catalog |
| `/v1/jobs` | POST | Submit async inference job (202 Accepted) |
| `/v1/jobs/{id}` | GET | Poll job status/result |
| `/swagger-ui/` | GET | OpenAPI documentation |

---

## Tech stack

| Layer | Language | What's built |
|---|---|---|
| Gateway | Rust (axum) | Health, models, job CRUD, OpenAPI/Swagger |
| Job domain | Rust | JobId, JobStatus, Job aggregate, JobStore port |
| Scheduler | Rust | TaskId, Task, Scheduler, round-robin, leases, retries |
| Inference runtime | Python | vLLM, Ollama, Mock backends |
| Native MoE | Rust | GGUF parser, forward pass (correlation 0.999) |
| Identity | Rust | NodeId (SHA-256 of Ed25519) |
| Contracts | Solidity | StakeManager (staking, slashing, banning) |

---

## Project structure

```
synapse/
├── synapse-core/            # Rust — single crate, single binary
│   ├── src/
│   │   ├── main.rs          #   Binary entrypoint (axum server)
│   │   ├── gateway/         #   axum HTTP: api, jobs, catalog, router
│   │   ├── job/             #   Job domain: JobId, JobStatus, Job, JobStore
│   │   ├── scheduler/       #   Scheduler: Task, TaskStatus, WorkerPort
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

*Last updated: 2026-07-31. V0-1 through V0-4 complete. Next: V0-5 (Native MoE Runtime — InferencePort Validation).*
