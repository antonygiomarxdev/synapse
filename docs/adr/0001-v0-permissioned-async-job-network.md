# ADR 001: V0 — Permissioned Async Job Network

**Date:** 2026-07-29
**Status:** Accepted
**Supersedes:** Implicit "global P2P MoE marketplace" vision for V0

## Context

Synapse was designed as a decentralized P2P inference protocol for Mixture-of-Experts models — hundreds of consumer GPUs coordinated via libp2p, with on-chain staking, slashing, dynamic pricing, and real-time consensus. The repo reflected that ambition: modules for DHT, WebRTC, speculative swarm, DAG engine, and L2 contracts.

External feedback (July 2026) identified a structural problem: **we were designing the economy before validating that the core technical idea works.** The recommendation: pivot to a humble MVP — a reliable batch inference system over a small, controlled network.

A viability spike (see `docs/superpowers/spikes/2026-07-28-vllm-viability-spike.md`) validated the most fundamental assumption: Rust can coordinate Python workers running real MoE models (IBM Granite 3B, 40 experts) via Unix sockets and protobuf, with <2ms overhead. The pipeline works.

The question is no longer "can we do inference?" It is: **can we coordinate multiple workers reliably?**

## Decision

**V0 is a permissioned async job network for batch inference workloads.**

We will build:

1. **Job model:** A `Job` is an immutable request with prompts, model ID, priority, deadline, and status. Submitted via `POST /v1/jobs`, queried via `GET /v1/jobs/{id}`. Results are downloadable artifacts.

2. **Scheduler:** Dispatches job tasks to known workers. Implements leases (no task runs forever), timeouts, retries, and re-assignment on worker failure. Round-robin dispatch; no dynamic pricing.

3. **Workers:** Python processes running vLLM or Ollama. Known in advance via configuration (allowlist), authenticated via Ed25519 keys. Heartbeats and capability reporting.

4. **Fault tolerance:** If a worker dies mid-task, the scheduler reassigns to another worker. Jobs are idempotent. No state is lost. No orphaned tasks.

5. **Metrics:** E2E success rate, retry rate, queue time, execution time, cost per 1M tokens, crash recovery time. Published per test run.

**We explicitly defer from V0:**

- DHT / Kademlia node discovery → static allowlist only
- P2P expert distribution → model replication (full model per worker)
- Real-time speculative swarm → async batch only
- On-chain payments / staking / slashing → off-chain reputation logging
- Dynamic pricing / market maker → fixed cost estimation
- OpenAI-compatible chat streaming → async job API
- WebRTC transport → Unix sockets (local) or TCP (remote, V1)
- Multi-runtime production support → vLLM + Ollama validated, second runtime for contract testing only

## Rationale

**Validate reliability before scale.** A 2-worker system that completes 100 jobs with zero losses teaches us more than a 100-worker system that silently drops tasks.

**Batch workloads are the natural first niche.** They tolerate latency, value throughput, and don't require streaming — exactly where a P2P network has structural advantage over centralized APIs. Chat/streaming puts P2P at a latency disadvantage.

**Permissioned first, open later.** Trust tiers: V0 = known nodes, V1 = allowlist, V2 = open network. This lets us debug coordination logic without adversarial actors.

**The two-sided marketplace problem requires bootstrapping.** Without jobs, hosts won't join. Without hosts, clients won't come. V0 bootstraps the demand side with a controlled worker pool, proving the product works before opening supply.

## Success criteria (go / no-go for V1)

| Metric | Target | Decision if missed |
|---|---|---|
| Jobs completed | ≥95% | Fix reliability before adding features |
| Crash recovery | <30s re-assignment | Fix leases/timeouts before opening to external hosts |
| Orphaned jobs | 0 | Idempotency and lease design is broken |
| Cost per 1M tokens | ≤ comparable centralized API | Re-evaluate use case or pricing model |
| Second runtime passes same test suite | 100% same behavior | Fix InferencePort abstraction before continuing |

## Consequences

**Positive:**
- Focused scope: 2-3 week vertical slice instead of months of infrastructure
- Every feature has a concrete acceptance test
- Evidence accumulates incrementally — no "big reveal" that might fail
- README and repo reflect honest state

**Negative:**
- Vision appears smaller to outside observers
- Code already written for DHT, swarm, economic contexts sits unused during V0
- Some dependencies (libp2p, alloy) are heavy but not exercised by V0

**Mitigation:** The existing module structure (dht, swarm, economic, transport) is preserved. They compile and pass tests. They are simply not on the V0 critical path. The vision hasn't changed — we're just validating the foundation first.

## References

- [Spike: vLLM viability validation](../superpowers/spikes/2026-07-28-vllm-viability-spike.md)
- [External feedback that motivated the pivot](../superpowers/spikes/2026-07-28-vllm-viability-spike.md#1-contexto-y-motivacion)
- [ESP32-AI project](https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-developer-runs-28-9-million-parameter-model-on-usd10-esp32-s3-microcontroller-uses-googles-per-layer-embeddings-technique-stores-table-on-16mb-flash-memory) — proof that aggressive quantization + mmap enables MoE on constrained hardware
- [MLPerf inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) — TTFT and TPOT as standard LLM metrics
