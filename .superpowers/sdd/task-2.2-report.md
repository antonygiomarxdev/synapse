# Task 2.2 Report: Consensus Domain — Vote + Audit

**Status:** DONE

## Commit

```
ad792aa feat(swarm): add token-level consensus vote and audit functions
```

3 files changed, +246 / -2:
- `synapse-core/src/shared/domain_error.rs` — +2 variants, +2 tests
- `synapse-core/src/identity/node_id.rs` — +serde derives
- `synapse-core/src/swarm/consensus.rs` — new file (impl + tests)

## Test Summary

All 137 tests pass across the crate (0 failures, 0 warnings, clippy-clean).

| Test | Type | Result |
|---|---|---|
| `invalid_consensus_quorum_display` | Unit | PASS |
| `no_consensus_display` | Unit | PASS |
| `majority_3_of_5_selects_consensus_token` | Unit | PASS |
| `audit_detects_log_prob_divergence` | Unit | PASS |
| `audit_rejects_different_lengths` | Unit | PASS |
| `audit_rejects_invalid_tolerance` | Unit | PASS |
| `vote_rejects_empty_outputs` | Unit | PASS |
| `vote_rejects_zero_quorum` | Unit | PASS |
| `vote_rejects_quorum_exceeding_swarm` | Unit | PASS |
| `vote_returns_no_consensus_when_no_quorum` | Unit | PASS |
| `audit_is_reflexive` | Proptest | PASS |
| `consensus_with_unanimity_returns_all_tokens` | Proptest | PASS |
| 125 pre-existing tests | — | PASS |

## Deviations from Brief

1. **`vote()` signature**: The brief's test code called `vote(&outputs, 3)` without `request_id`. Since `ConsensusResult` contains `request_id: Uuid`, the implementation signature is `vote(request_id: Uuid, node_outputs: &[NodeOutput], quorum: usize)`. All test calls updated to pass `Uuid::new_v4()`.

2. **`Eq` dropped from domain types**: `Token` contains `f64` and cannot derive `Eq`. Therefore `NodeOutput` and `ConsensusResult` derive `PartialEq` only (not `Eq`). The struct derives in the brief included `Eq`; corrected to match Rust f64 constraints.

3. **Serde added to `NodeId`**: Required so `NodeOutput` and `ConsensusResult` can derive `Serialize`/`Deserialize` as specified. Added `#[derive(Serialize, Deserialize)]` to `NodeId` and the corresponding `use serde::{Deserialize, Serialize}` import.

4. **Unused `ModelId` import removed**: The brief's test module imported `ModelId` but never used it. Removed to keep clippy-clean.

5. **Additional unit tests**: Added `audit_rejects_different_lengths`, `audit_rejects_invalid_tolerance`, `vote_rejects_empty_outputs`, `vote_rejects_zero_quorum`, `vote_rejects_quorum_exceeding_swarm`, and `vote_returns_no_consensus_when_no_quorum` for fuller coverage.

## Files Changed

- **`synapse-core/src/shared/domain_error.rs`**: Added `InvalidConsensusQuorum { quorum: usize, swarm_size: usize }` and `NoConsensus { token_index: usize }` variants with thiserror display annotations. Added display tests.

- **`synapse-core/src/identity/node_id.rs`**: Added `Serialize`/`Deserialize` derives and import to enable serde in consensus domain types.

- **`synapse-core/src/swarm/consensus.rs`** (new):
  - `NodeOutput` — single node's token output
  - `ConsensusResult` — aggregated consensus output
  - `vote()` — token-by-token majority vote with quorum validation
  - `audit()` — statistical comparison with configurable tolerance
  - Test module with 8 unit tests + 2 proptest properties

## Concerns

None.
