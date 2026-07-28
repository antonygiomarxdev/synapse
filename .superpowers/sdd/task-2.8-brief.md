### Task 2.8: Final Gauntlet — Format, Lint, Test, Coverage

**Files:**
- Modify: any file that `cargo fmt` or `cargo clippy` complains about

**Interfaces:**
- Consumes: all previous tasks

- [ ] **Step 1: Format and lint**

Run:

```bash
cd synapse-core
cargo fmt --check
cargo clippy -- -D warnings
```

Expected: both PASS.

- [ ] **Step 2: Run all swarm tests**

Run:

```bash
cargo test swarm -p synapse-core
```

Expected: all PASS.

- [ ] **Step 3: Run full core test suite**

Run:

```bash
cargo test -p synapse-core
```

Expected: all PASS.

- [ ] **Step 4: Check coverage (informative)**

Run:

```bash
cargo llvm-cov -p synapse-core --lib
```

Expected: swarm module ≥80% line and function coverage.

- [ ] **Step 5: Commit and close task**

```bash
git commit -m "chore(swarm): final gauntlet passes for issue #2"
```

---

## Self-Review

### 1. Spec coverage

| Issue / Spec Requirement | Task |
|---|---|
| Token VO (id, text, log_prob) | 2.1 |
| log_prob validation, empty text edge case | 2.1 |
| Consensus vote counting + majority detection | 2.2 |
| Audit comparison (identical seeds → identical log_probs) | 2.2 |
| Speculative swarm size + seed distribution | 2.3 |
| DAG route assembly + expert dependency graph | 2.4 |
| Re-sync divergence + expulsion + chronic flagging | 2.5 |
| InferenceEngine / SwarmCoordinator ports | 2.6 |
| libp2p coordinator adapter + integration tests | 2.7 |
| `cargo test swarm` green | 2.8 |

### 2. Placeholder scan

- No `TBD`, `TODO`, or `implement later`.
- No vague "add error handling" steps.
- Every test step includes concrete code.
- Every implementation step includes concrete code.

### 3. Type consistency

- `Token` is defined in 2.1 and reused in 2.2, 2.5, 2.6, 2.7.
- `NodeOutput` is defined in 2.2 and reused in 2.6, 2.7.
- `SpecSwarmConfig` is defined in 2.3 and reused in 2.6, 2.7.
- `ConsensusResult` is defined in 2.2 and reused in 2.6, 2.7.
- `DomainError` variants are added incrementally and do not conflict.
- `NodeId` constructor signature `from_public_key(&[u8; 32])` is used consistently.

### 4. Known gaps / V2+ notes

- The libp2p adapter in 2.7 is a trait bridge. Full network protocol messages (gossipsub, request/response) are out of scope for this issue and tracked in V2 roadmap.
- `ReSyncPolicy` does not yet implement the 24-hour chronic window; it counts flags per request. Time-bounded windowing is V2+.
- The DAG route does not include pricing or node selection; those belong to the gateway/economic contexts and are out of scope here.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-28-issue-2-swarm-context.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — I execute the tasks in this session using `superpowers:executing-plans` with checkpoints.

**Which approach?**
