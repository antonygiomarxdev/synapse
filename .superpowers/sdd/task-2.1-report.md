# Task 2.1 Report: Token Value Object

## Status: DONE

## Commit
```
2048600 feat(swarm): add Token value object with log_prob validation
```

## Files Changed
- `synapse-core/src/shared/domain_error.rs` — Added `InvalidTokenLogProb { value: f64 }` and `InvalidTokenText { reason: String }` variants; removed `Eq` from derive macro (f64 does not implement Eq); added display tests for both new variants
- `synapse-core/src/swarm/token.rs` — Created new `Token` value object with `id`, `text`, `log_prob` fields; validation for finite log_prob and max text length (65,536 bytes); accessor methods; 4 unit tests
- `synapse-core/src/swarm/mod.rs` — Added `pub mod token;` and `pub use token::Token;`

## Changes from Brief
- Removed `Eq` from `DomainError`'s derive macro because `f64` does not implement `Eq`, making `InvalidTokenLogProb { value: f64 }` unrepresentable with `Eq` on the enum. No code in the crate relied on `DomainError: Eq`.

## Test Results
- **125/125 tests passing** across 4 suites (`synapse-core` full suite)
- Token-specific: 4/4 passing (nan rejection, infinite rejection, empty text acceptance, overly long text rejection)
- DomainError-specific: 11/11 passing (9 existing + 2 new)

## Verification
- `cargo fmt` — clean
- `cargo clippy -- -D warnings` — clean
- Full test suite — 125 passed
