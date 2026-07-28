# Task 2.3 Report — Speculative Swarm Domain

## Status
✅ Complete.

## Commits
```
dcbbf3d feat(swarm): add speculative swarm config with seed distribution
  (2 files +103 -1)
```

## Files Changed
| File | Change |
|---|---|
| `synapse-core/src/swarm/speculative.rs` | Created: `SpecSwarmConfig` struct + tests (+95 lines) |
| `synapse-core/src/swarm/mod.rs` | Added re-export `pub use speculative::SpecSwarmConfig;` (+1 line) |

## Test Summary
```
cargo test swarm::speculative -p synapse-core
6 passed (3 suites, 137 filtered, 0.00s)
```

| Test | Status |
|---|---|
| `rejects_swarm_size_below_minimum` (size=1) | ✅ |
| `rejects_swarm_size_above_maximum` (size=33) | ✅ |
| `valid_size_5_has_unique_seeds` (5 unique seeds generated) | ✅ |
| `quorum_for_size_5_is_3` (⌊5/2⌋+1 = 3) | ✅ |
| `quorum_for_size_8_is_5` (⌊8/2⌋+1 = 5) | ✅ |
| `quorum_for_size_3_is_2` (⌊3/2⌋+1 = 2) | ✅ |

## TDD Trace
1. **Write tests** → placed `#[cfg(test)] mod tests` with 6 test functions referencing `SpecSwarmConfig`
2. **Confirm fail** → `cargo test` → `error[E0433]: cannot find type SpecSwarmConfig`
3. **Implement** → added `SpecSwarmConfig { model, swarm_size, seeds }` with `new()`, `model()`, `swarm_size()`, `seeds()`, `quorum()`. Validation range `[2, 32]`. Seeds generated as `1..=swarm_size`. Quorum = `floor(swarm_size / 2) + 1`.
4. **Re-export** → added `pub use speculative::SpecSwarmConfig;` in `swarm/mod.rs`
5. **Pass** → all 6 tests pass
6. **Lint** → `cargo fmt` (no changes), `cargo clippy -- -D warnings` (clean after 2 fixes: `manual_range_contains` → `RangeInclusive::contains`, unnecessary `u32` cast removed)

## Concerns
- `quorum()` returns `usize` while `swarm_size` is `u32` — the cast `swarm_size as usize` is safe in practice (size ≤ 32) but an assertion could document the assumption.
- Seeds are deterministic (`1..=swarm_size`), which is fine for the domain model; actual randomness is an infrastructure concern.
- No `Default` impl — intentional, since there's no sensible default swarm size.
