# Synapse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Synapse reference implementation — a P2P swarm protocol for MoE model inference — in 6 phases, producing a working V1 MVP.

**Architecture:** Rust single binary (axum gateway + libp2p DHT + swarm core). Python vLLM runtime as subprocess via Unix socket + protobuf. Solidity staking on L2. Two swarm modes: Speculative (realtime ensemble) and DAG (batch expert distribution). Primary example model: Kimi K3.

**Tech Stack:** Rust 1.97+, axum 0.8, libp2p 0.56, alloy v2, ed25519-dalek 3.0, Python 3.12+, vLLM 0.26, Solidity 0.8.36, Hardhat, protobuf.

**Design Spec:** `docs/superpowers/specs/2026-07-27-synapse-design.md`

## Global Constraints

- Single Rust binary: `synapse-node` (gateway + swarm + DHT). No separate Python gateway.
- All node communication via libp2p/WebRTC (DTLS encrypted). No plaintext transport.
- All models identified by SHA256 weight hash. Catalog verifies hashes before listing.
- Inference seed=0 for batch mode (auditable). Variable seed for speculative mode.
- Gateway fee: 15-20%. Miners compete on ask price. Gateway is the market maker.
- V1: no TURN servers (STUN only). ~10% miner exclusion acceptable.
- Catalog: Synapse Inc. curated. Community PR welcome. Kimi K3 as primary model.
- Miner UX: zero-decision default (auto-assign experts + runtime). Power user mode optional.
- Bootstrap: 3-5 VPS seed nodes, HuggingFace as weight fallback.
- Slashing: fully automatic. No human governance in V1.
- InferencePort trait: runtime-agnostic. vLLM V1, llama.cpp/SGLang V2+.
- QA: 20 tests minimum, 80%+ coverage, `make gauntlet` before merge.
- All code in English. Commit messages follow Conventional Commits.
- DDD with Clean Architecture: domain layer pure (no I/O), application ports as traits.

---

## Phase 1: Foundation — Identity + Model Contexts

**GitHub Issue:** [#1](https://github.com/antonygiomarxdev/synapse/issues/1)

**Files:**
- `synapse-core/src/shared/` — domain errors, domain events
- `synapse-core/src/identity/` — NodeId, KeyPair, Node aggregate
- `synapse-core/src/model/` — ModelId, ExpertId, Catalog

**Interfaces produced:**
- `NodeId` — newtype `[u8; 32]`, SHA256 of Ed25519 public key
- `KeyPair` — pure domain entity (no crypto deps in domain)
- `Node` — aggregate: NodeId + stake address + reputation
- `ModelId` — value object
- `ExpertId` — value object
- `Catalog` — aggregate: registers models, rejects duplicates
- `KeySigner` trait (application port)
- `IdentityStore` trait (application port)

### Task 1.0: Shared Kernel

- [ ] Define `DomainError` enum (16 variants: InvalidNodeId, InvalidModelId, DuplicateModel, etc.)
- [ ] Define `DomainEvent` enum (7 events: NodeRegistered, ModelAdded, StakeUpdated, etc.)
- [ ] Write tests for error display and event serialization
- [ ] `cargo test shared` — all green

### Task 1.1: NodeId Value Object

- [ ] Implement `NodeId([u8; 32])` with `from_public_key(pk: &[u8; 32]) -> Self`
- [ ] Derive: Debug, Clone, Copy, PartialEq, Eq, Hash
- [ ] 8 tests: determinism, uniqueness, invalid input (wrong length), hex parsing, display
- [ ] `cargo test identity::node_id` — 8 passed

### Task 1.2: KeyPair Entity (Pure Domain)

- [ ] Define `KeyPair` struct with `public: [u8; 32]`, `secret: [u8; 32]`
- [ ] `generate() -> Self` using rand (no ed25519 dependency in domain)
- [ ] `public_key_bytes(&self) -> &[u8; 32]`
- [ ] Tests: two generates produce different keys, public_key_bytes is 32 bytes

### Task 1.3: Node Aggregate

- [ ] `Node` struct: NodeId, stake_address (String), reputation (u16, 0-1000)
- [ ] `Node::register(keypair: &KeyPair, stake_address: String) -> Result<Self, DomainError>`
- [ ] `Node::derive_node_id(keypair: &KeyPair) -> NodeId` (SHA256 of public key)
- [ ] Tests: register produces valid Node, reputation starts at 100, duplicate stake address rejected

### Task 1.4: Application Ports (Traits)

- [ ] `KeySigner` trait: `sign(&self, data: &[u8]) -> Vec<u8>`, `verify(&self, data: &[u8], sig: &[u8]) -> bool`
- [ ] `IdentityStore` trait: `save(&self, node: &Node) -> Result<(), DomainError>`, `find(&self, id: &NodeId) -> Option<Node>`

### Task 1.5: Ed25519Signer Adapter

- [ ] `Ed25519Signer` struct wrapping `ed25519_dalek::SigningKey`
- [ ] Implement `KeySigner` trait
- [ ] 3 integration tests: sign + verify, wrong signature rejected, different key fails verification
- [ ] `cargo test identity::infrastructure` — 3 passed

### Task 1.6: ModelId + ExpertId Value Objects

- [ ] `ModelId(String)` — newtype, validation (non-empty, kebab-case)
- [ ] `ExpertId { model: ModelId, index: u32 }` — composite VO
- [ ] 10 tests total: empty rejected, valid accepted, kebab-case enforced, expert index zero allowed, equality
- [ ] `cargo test model::model_id` — 10 passed

### Task 1.7: Catalog Aggregate

- [ ] `Model` entity: ModelId, total_params, num_experts, active_per_token, context_window, license
- [ ] `Catalog` aggregate: `register(model: Model) -> Result<(), DomainError>`, `list() -> Vec<&Model>`
- [ ] Tests: duplicate registration rejected, list returns all models, Kimi K3 specs validated
- [ ] `cargo test model::catalog` — all green

**Phase 1 Acceptance:**
- `cargo test` — all tests green (target: 25+)
- `cargo clippy -- -D warnings` — clean
- `cargo fmt --check` — clean
- Domain layer has zero external dependencies (no ed25519, no libp2p, no axum)
- All public types documented with doc comments

---

## Phase 2: Swarm — Consensus + Speculative + DAG

**GitHub Issue:** [#2](https://github.com/antonygiomarxdev/synapse/issues/2)

### Task 2.1: Token Value Object

- [ ] `Token` struct: id (u32), text (String), log_prob (f32)
- [ ] Validation: non-empty text, log_prob in (-inf, 0]
- [ ] Tests: empty text rejected, valid token accepted, log_prob boundaries

### Task 2.2: Consensus Domain (Pure Functions)

- [ ] `vote(tokens: &[Vec<Token>]) -> ConsensusResult` — pure function
- [ ] `ConsensusResult` struct: consensus_tokens, divergent_nodes, majority_count
- [ ] Property tests (proptest): random token streams, verify majority detection at 3/5, 5/8, 8/16
- [ ] `audit(expected: &[Token], actual: &[Token]) -> bool` — used for batch mode verification

### Task 2.3: Speculative Swarm Domain

- [ ] `SpecSwarmConfig`: swarm_size (2-32), seeds (Vec<u32>)
- [ ] Validation: swarm_size must be odd for clear majority, seeds.len() == swarm_size
- [ ] `SwarmSession`: tracks node states, divergence counters, re-sync state
- [ ] Tests: invalid config rejected, session tracks divergences correctly

### Task 2.4: DAG Swarm Domain

- [ ] `DagRoute`: path of ExpertId through the swarm
- [ ] `ExpertDependencyGraph`: which experts must execute before others
- [ ] Route assembly from co-activation heat map
- [ ] Tests: valid route assembled, cycle detection, fallback routes

### Task 2.5: Re-Sync Policy

- [ ] `ReSyncPolicy`: max_divergences_per_request (3), expulsion threshold, chronic flagging (10 in 24h)
- [ ] `should_expel(divergences: u32) -> bool`
- [ ] `should_flag(node_history: &NodeHistory) -> bool`
- [ ] Tests: expel at 3, don't expel at 2, chronic detection

### Task 2.6: Application Ports

- [ ] `InferenceEngine` trait: `generate(model: &ModelId, prompt: &[Token], seed: u32) -> Vec<Token>`
- [ ] `SwarmCoordinator` trait: `coordinate(request: InferenceRequest) -> ConsensusResult`
- [ ] `AuditEngine` trait: `verify(node_a: &[Token], node_b: &[Token]) -> bool`

### Task 2.7: libp2p Coordinator Adapter

- [ ] Implement `SwarmCoordinator` using libp2p for node communication
- [ ] Integration test: 3 simulated nodes, generate tokens, verify consensus
- [ ] `cargo test swarm::integration` — all green

**Phase 2 Acceptance:**
- Consensus: token-level voting with majority detection (3/5, 5/8, 8/16)
- Audit: identical seeds produce identical log_probs
- Re-sync: 3+ divergences → expel, 10+ in 24h → slashing flag
- All domain logic pure (no I/O, no network)
- `cargo test swarm` — all green

---

## Phase 3: Economic — Reputation + Pricing + Stake

**GitHub Issue:** [#3](https://github.com/antonygiomarxdev/synapse/issues/3)

### Task 3.1: Reputation Value Object

- [ ] `Reputation(u16)` — bounds 0-1000
- [ ] `tier(&self) -> Tier` — Bronze (0-299), Silver (300-599), Gold (600-849), Platinum (850-1000)
- [ ] `apply_decay(&mut self, hours_inactive: u32)`
- [ ] Tests: bounds enforcement, tier transitions, decay formula, overflow protection (can't go below 0)

### Task 3.2: Price Domain

- [ ] `TokensPerMillion(u64)` — value object, non-zero validation
- [ ] `RouteCost { expert_prices: Vec<TokensPerMillion> }` — sum of expert costs
- [ ] `cheapest_route(experts: &[(ExpertId, Vec<TokensPerMillion>)]) -> (Vec<ExpertId>, TokensPerMillion)`
- [ ] Property tests: zero price rejected, cheapest correctly identified, proptest on random price sets

### Task 3.3: Stake + Slashing

- [ ] `StakeAmount(u64)` — in USDC cents
- [ ] `SlashingPolicy`: graduated penalties (warning → freeze 48h → 20% slash → full slash + ban)
- [ ] `apply_slashing(stake: &mut StakeAmount, flags: u32) -> SlashingResult`
- [ ] Tests: 10 flags → freeze, 50 flags → slash, minimum stake maintained, ban on insufficient stake

### Task 3.4: Route Assembly

- [ ] `assemble_route(model: &Model, node_registry: &[(NodeId, ExpertId, TokensPerMillion)]) -> Option<DagRoute>`
- [ ] Prefer co-activated experts on same node
- [ ] Fallback to cheapest valid route with replicas
- [ ] Tests: cheapest route selected, co-activation preference, fallback when primary unavailable

### Task 3.5: Application Ports

- [ ] `StakeContract` trait: `stake(node: NodeId, amount: StakeAmount)`, `slash(node: NodeId, amount: StakeAmount)`, `freeze(node: NodeId, duration: Duration)`
- [ ] `PaymentGateway` trait: `pay(node: NodeId, amount: TokensPerMillion, tokens: u64) -> Result<TxHash>`

### Task 3.6: L2 Adapter (alloy v2 + Solidity)

- [ ] Implement `StakeContract` using alloy v2 bindings to `StakeManager.sol`
- [ ] Integration tests: stake, unstake, flag accumulation, slashing, reputation
- [ ] `npx hardhat test` — all green
- [ ] `cargo test economic::infrastructure` — all green

**Phase 3 Acceptance:**
- Reputation: 0-1000 bounded, tier transitions, decay verified
- Pricing: zero rejected, cheapest route assembled
- Slashing: graduated penalties, freeze/unfreeze, ban
- Smart contract: all state transitions tested
- `cargo test economic` + `npx hardhat test` — all green

---

## Phase 4: Runtime — InferencePort + vLLM Adapter

**GitHub Issue:** [#4](https://github.com/antonygiomarxdev/synapse/issues/4)

### Task 4.1: InferencePort Protocol

- [ ] `InferencePort` trait in Rust: load, generate, verify, detect_vram
- [ ] Request/Response protobuf messages for Unix socket communication
- [ ] Python dataclasses mirroring the protobuf schema
- [ ] Tests: serialization roundtrip Rust→Python→Rust

### Task 4.2: vLLM Engine Adapter

- [ ] `synapse_runtime/engine.py` — VLLMEngine class
- [ ] `load_model(model_id: str, experts: list[int])` — loads specific experts
- [ ] `generate(prompt: str, seed: int, max_tokens: int) -> list[Token]`
- [ ] Integration test: load Mixtral 8x7B, generate 10 tokens, verify output non-empty
- [ ] `python -m pytest tests/test_engine.py -v`

### Task 4.3: Weight Loader

- [ ] `synapse_runtime/loader.py` — HuggingFace download
- [ ] `download_model(hf_repo: str) -> Path`
- [ ] `verify_sha256(path: Path, expected_hash: str) -> bool`
- [ ] `extract_experts(checkpoint_path: Path, expert_ids: list[int]) -> list[Path]`
- [ ] Tests: SHA256 verification, expert extraction correctness

### Task 4.4: Deterministic Seed

- [ ] `synapse_runtime/deterministic.py` — enforce seed=0 for batch mode
- [ ] `lock_precision()` — FP16 enforcement
- [ ] Test: two runs with identical input + seed=0 produce identical output tensors
- [ ] `python -m pytest tests/test_deterministic.py -v`

### Task 4.5: Auto-Assign

- [ ] `synapse_runtime/auto_assign.py` — VRAM detection
- [ ] `detect_vram() -> int` — returns available VRAM in MB
- [ ] `select_experts(available_vram: int, model: ModelEntry) -> list[int]` — picks optimal expert set
- [ ] `select_runtime(available_vram: int) -> str` — picks vLLM or llama.cpp based on VRAM
- [ ] Tests: 8GB selects 1 expert, 24GB selects full model, runtime selection at boundaries

### Task 4.6: Unix Socket Server

- [ ] `synapse_runtime/server.py` — protobuf over Unix socket
- [ ] Accept `InferenceRequest`, dispatch to engine, return `InferenceResponse`
- [ ] Error handling: connection drops, reconnection, invalid requests
- [ ] Integration test: Rust binary sends request, receives response
- [ ] `cargo test runtime::integration` — all green

**Phase 4 Acceptance:**
- vLLM generates tokens for Mixtral 8x7B and Kimi K3
- SHA256 matches official HuggingFace hash
- Identical seeds produce identical outputs
- Auto-assign selects optimal experts + runtime
- Architecture supports adding llama.cpp/SGLang without core changes

---

## Phase 5: Gateway — axum API + Catalog + Routing

**GitHub Issue:** [#5](https://github.com/antonygiomarxdev/synapse/issues/5)

### Task 5.1: Gateway Domain

- [ ] Catalog queries: filter by status, sort by price/latency
- [ ] Pricing calculations: final client price = miner cost + gateway fee
- [ ] Route assembly: cheapest valid route from DHT registry
- [ ] Tests: all domain logic pure, no HTTP

### Task 5.2: axum Endpoints

- [ ] `GET /v1/models` — returns curated catalog with pricing and health
- [ ] `POST /v1/chat/completions` — routes to swarm, returns tokens
- [ ] `GET /health` — version, status, swarm node count
- [ ] Request validation: model must exist, messages non-empty
- [ ] Integration test: full HTTP roundtrip to stub response

### Task 5.3: Model Catalog

- [ ] Load from `config/models.toml`
- [ ] Health status: healthy (≥3 replicas), degraded (1-2), unavailable (0)
- [ ] Kimi K3 as primary entry
- [ ] Tests: catalog loads correctly, health transitions, missing file error

### Task 5.4: Market Maker Pricing

- [ ] Collect node asks from DHT: `expert://<hash>/<id> → {price: 0.08, ...}`
- [ ] Compute cheapest valid route: sum cheapest expert prices + gateway fee
- [ ] Publish final price in catalog response
- [ ] Tests: cheapest route assembled, gateway fee applied, price never below cost

### Task 5.5: Request Routing

- [ ] Node selection by: latency (<200ms realtime, any for batch), reputation (≥300), price (cheapest first)
- [ ] Swarm size management: validate 2-32 range, odd numbers preferred
- [ ] Fallback: if primary node fails, retry with replica
- [ ] Tests: node selection logic, fallback routing, invalid swarm_size rejected

### Task 5.6: Billing

- [ ] Token counting: input tokens (prompt) + output tokens (completion)
- [ ] Usage tracking: per-client, per-model, per-day
- [ ] Invoice generation: summary of tokens consumed × price
- [ ] Tests: token counting accuracy, daily aggregation, invoice format

### Task 5.7: Middleware

- [ ] API key authentication via `Authorization: Bearer <key>` header
- [ ] Rate limiting: per-key, per-model limits
- [ ] Request logging: method, path, status, duration
- [ ] CORS: allow all origins (B2B clients may be anywhere)
- [ ] Tests: auth rejects invalid keys, rate limiting enforced, logs include required fields

**Phase 5 Acceptance:**
- Gateway starts as part of `synapse-node` binary
- `GET /v1/models` returns curated catalog with Kimi K3
- `POST /v1/chat/completions` routes to swarm, returns tokens
- Auto mode selects cheapest model
- `priority: "realtime"` uses speculative swarm, `"batch"` uses DAG
- Gateway has redundancy: multiple instances, DHT is source of truth

---

## Phase 6: E2E Integration + QA

**GitHub Issue:** [#6](https://github.com/antonygiomarxdev/synapse/issues/6)

### Task 6.1: E2E Speculative Swarm

- [ ] 5 real nodes, generate tokens with Kimi K3 model stub
- [ ] Verify consensus: 3/5 majority detected
- [ ] `cargo test e2e::speculative` — all green

### Task 6.2: E2E DAG Swarm

- [ ] 4 nodes, 10 batch requests, expert routing through DAG
- [ ] Verify throughput scales with node count
- [ ] `cargo test e2e::dag` — all green

### Task 6.3: Failure Recovery

- [ ] Node disconnect mid-request → re-route to replica
- [ ] Re-sync protocol: divergent node expelled after 3 misses
- [ ] Client receives normal response despite node failure
- [ ] `cargo test e2e::resilience` — all green

### Task 6.4: Network Partition

- [ ] Split swarm → each half operates degraded → reconnect → reconcile
- [ ] Verify no slashing during grace window after reconnect
- [ ] `cargo test e2e::partition` — all green

### Task 6.5: Fuzz Testing

- [ ] proptest: random token streams through consensus voting
- [ ] proptest: random price sets through route assembly
- [ ] proptest: random reputation scores through tier calculation
- [ ] 10k+ iterations without panics
- [ ] `cargo test e2e::fuzz` — all green

### Task 6.6: Load Testing

- [ ] Simulate 100+ node swarm
- [ ] Measure throughput: target >50 tok/s aggregate
- [ ] Measure latency: P99 < 500ms for realtime mode
- [ ] `cargo bench` — results within targets

### Task 6.7: Smart Contract Audit

- [ ] Slither static analysis: zero high-severity findings
- [ ] All state transitions tested: stake, unstake, flag, slash, ban
- [ ] 100% branch coverage on StakeManager.sol
- [ ] `npx hardhat test` + `slither .` — all green

### Task 6.8: Documentation

- [ ] Developer setup guide (from clone to first inference)
- [ ] Miner quickstart (install → stake → start mining)
- [ ] API reference (endpoints, parameters, responses, error codes)
- [ ] Architecture decision records (key technical choices)

**Phase 6 Acceptance:**
- Full flow: client → gateway → 5-node swarm → response <5s (realtime mode)
- Single node failure invisible to client
- Partitioned swarm continues degraded, reconciles cleanly
- Fuzz tests: 10k+ iterations without panics
- Smart contract: 100% branch coverage, zero high-severity Slither findings
- Load test: 100 concurrent requests, zero errors, >50 tok/s aggregate
- `make gauntlet` — all green
