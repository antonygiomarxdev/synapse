# Task 2.7 Report: libp2p Coordinator Adapter + Integration Tests

## Status: Complete

## Commit

`e667ecc` — `feat(swarm): add libp2p coordinator adapter and integration tests`

4 files changed, 101 insertions.

## Files Created/Modified

| File | Change | Lines |
|------|--------|-------|
| `synapse-core/src/swarm/infrastructure/mod.rs` | Created — module scaffold that re-exports `Libp2pSwarmCoordinator` | 3 |
| `synapse-core/src/swarm/infrastructure/libp2p_swarm_coordinator.rs` | Created — `Libp2pSwarmCoordinator` implementing `SwarmCoordinator` trait | 53 |
| `synapse-core/src/swarm/mod.rs` | Modified — added `pub mod infrastructure;` | +1 |
| `synapse-core/tests/libp2p_coordinator_integration.rs` | Created — integration test with `DeterministicEngine` | 44 |

## Implementation Details

### `Libp2pSwarmCoordinator`
- **Fields**: `engine: Arc<dyn InferenceEngine>`, `last_outputs: Vec<NodeOutput>` (V1 placeholder — not yet persisted via interior mutability since `coordinate` takes `&self`)
- **`coordinate()`**: Extracts `SpecSwarmConfig` from request, simulates N nodes by calling `self.engine.generate()` N times (all return identical tokens in V1), applies `vote()` for majority consensus
- **`node_outputs()`**: Returns clone of `last_outputs` (currently always empty — V1 simulation)
- **Debug**: Manual impl that skips the non-`Debug` `Arc<dyn InferenceEngine>` field

### Integration Test
- **`DeterministicEngine`** test helper returns fixed `tokens` on every `generate()` call
- **`coordinator_reaches_consensus_with_unanimous_engine`**: Verifies:
  - Swarm of 5 nodes (quorum = 3) with unanimous tokens reaches consensus
  - Both tokens are present in `consensus_tokens`
  - Token text matches expected values
  - `divergent_nodes` is empty

## Test Summary

```
cargo test --test libp2p_coordinator_integration -p synapse-core
  1 passed (1 suite, 0.00s)
```

`cargo fmt` and `cargo clippy -p synapse-core --all-targets`: all clean (pre-existing warnings in `consensus.rs` and `e2e.rs` only).

## Concerns

1. **`last_outputs` not populated**: `coordinate` takes `&self` (per trait signature), so the outputs vector built during coordination cannot be stored without interior mutability (`Mutex`/`RefCell`). V1 accepts this — `node_outputs()` always returns empty. A future phase should either add interior mutability or change the trait to `&mut self`.
2. **Single-engine simulation**: All N nodes get the same tokens because one engine serves all calls. Per-node divergence is impossible to test until the coordinator accepts per-node engines or the engine incorporates seed information from the request.
3. **`let _ = outputs;`**: The brief includes this line to express intent (outputs could be stored later), even though it's a no-op after `vote()` already consumed the borrow. Harmless but stylistically unusual.

## Fix 2.7 Review Issues

**Status:** Complete — all 3 fixes applied, verified.

### Fixes Applied

| # | Severity | File | Change |
|---|----------|------|--------|
| 1 | Important | `synapse-core/src/swarm/infrastructure/libp2p_swarm_coordinator.rs` | Added `#[derive(Clone)]` to `Libp2pSwarmCoordinator` |
| 2 | Minor | `synapse-core/src/swarm/infrastructure/libp2p_swarm_coordinator.rs` | Removed dead `let _ = outputs;` (and orphaned comment) |
| 3 | Minor | `synapse-core/tests/libp2p_coordinator_integration.rs` | Changed `#[tokio::test]` → `#[test]`, removed `async` |

### Verification

- **`cargo test -p synapse-core`** — 154 passed, 0 failed
- **`cargo clippy -- -D warnings`** — clean

