# Task 2.5 Report: Re-Sync Policy

## Status
**Completed.**

## Commits
```
bde661b feat(swarm): add re-sync policy with divergence and chronic flag tracking
```

Files changed:
- `synapse-core/src/swarm/resync.rs` (new, +142 lines)
- `synapse-core/src/swarm/mod.rs` (+2 lines: module declaration + re-export)

## Test Summary
```
cargo test swarm::resync -p synapse-core
4 passed (3 suites, 148 filtered, 0.00s)
```

| Test | Status |
|---|---|
| `node_expelled_after_three_divergences` | PASS |
| `expulsion_resets_after_request_boundary` | PASS |
| `chronic_flag_after_ten_flags` | PASS |
| `different_nodes_tracked_independently` | PASS |

**TDD flow:**
1. **Red:** Created `resync.rs` with only the test module — `cargo test` failed with `ReSyncPolicy` not found.
2. **Green:** Added the `ReSyncPolicy` struct, `Default` impl, and all methods — all 4 tests passed.
3. **Refactor:** `cargo fmt` fixed method-chaining style; `cargo clippy` clean; re-exported from `mod.rs`.

## Concerns
None. The domain is straightforward state tracking using `HashMap<NodeId, u32>` for per-request divergences and chronic flags. The `_token: &Token` parameter is accepted but unused — it's future-proofing for token-content-based analysis.

## Task 2.5 Review Fixes

**Date:** 2026-07-28

### Fix 1 (Important): Add `chronic_window_hours` field
- Added `chronic_window_hours: u32` field to `ReSyncPolicy` struct
- Set default value to `24` in `Default::default()`
- Added `chronic_window_hours` parameter to `ReSyncPolicy::new(…)`
- Added `pub fn chronic_window_hours(&self) -> u32` accessor
- Updated doc comment on struct to reference the field

### Fix 2 (Minor): Remove unused `_token` parameter from `record_divergence`
- Changed signature to `record_divergence(&mut self, node_id: NodeId)`
- Removed `use crate::swarm::Token;` import (no longer needed)
- Removed `fn token()` test helper and `use crate::swarm::Token;` from test module
- Updated all test calls to `record_divergence(n)` (no token argument)

### Verification
```
cargo test swarm::resync -p synapse-core
4 passed (3 suites, 148 filtered, 0.00s)
cargo clippy -p synapse-core -- -D warnings
clean — no warnings
cargo fmt
no formatting changes
```
