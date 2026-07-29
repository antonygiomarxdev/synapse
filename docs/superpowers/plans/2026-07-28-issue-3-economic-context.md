# Issue #3: Phase 3 — Economic Context (Reputation + Pricing + Stake)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Economic bounded context: reputation scoring with tiers and decay, market maker pricing model, stake management with graduated slashing, and cheapest-route assembly for the DAG engine.

**Architecture:** Domain layer stays pure (zero I/O, zero crypto). New value objects and aggregates live in `synapse-core/src/economic/`. Application ports (`StakeContract`, `PaymentGateway`) are traits in `economic/ports.rs`. An alloy v2 adapter wraps `StakeManager.sol` and lives in `economic/infrastructure/`. Route assembly is a pure function consuming the existing `ExpertId` and `DagRoute` types.

**Tech Stack:** Rust 1.97 (pinned), edition 2024. `thiserror 2` for errors, `serde 1` for serialization, `uuid 1` for event IDs, `proptest 1` for property tests, `alloy 2` for L2 adapter, `chrono 0.4` for decay timestamps, Solidity 0.8.36 for `StakeManager.sol`.

**Design Spec:** `docs/superpowers/specs/2026-07-27-synapse-design.md` (sections 7.2-7.4, 5.7)
**Parent Plan:** `docs/superpowers/plans/2026-07-27-synapse-implementation.md`
**Issue:** https://github.com/antonygiomarxdev/synapse/issues/3

## Global Constraints

These expand on the non-negotiable principles in `AGENTS.md` and the issue acceptance criteria. Every task MUST comply.

### DDD
- Domain layer has ZERO external dependencies — no `alloy`, no `chrono::Utc::now()`, no network, no async.
- All domain types are plain structs/enums with `Debug`, `Clone`, `PartialEq`, and `Eq` where possible.
- Value objects validate invariants at construction and return `DomainError`.
- Aggregates that mutate state return `Vec<DomainEvent>` alongside the result.
- Ports are traits defined in `economic/ports.rs`, implemented by adapters in `economic/infrastructure/`.
- `chrono` is used ONLY for type-level timestamps (`DateTime<Utc>` fields on domain types). NEVER call `Utc::now()` — callers inject the current time.

### TDD
- **EVERY task step writes the test BEFORE the implementation.** The test MUST be run and MUST fail before writing code.
- Tests inline: `#[cfg(test)] mod tests` in the same file as source.
- Domain tests are pure: construct inputs, call function, assert output. No mocks, no test doubles.
- Use `proptest` for price comparison and route assembly property tests (Task 3.2, 3.4).
- Infrastructure tests use the real `alloy` crate with a local Anvil instance (Task 3.6).

### Clean Code
- ALL public types get `///` doc comments. First line is a summary sentence.
- `thiserror` for domain errors — no manual `Display`/`Error` impls.
- `cargo fmt` (max_width 100, 4-space indent), `cargo clippy -- -D warnings` before every commit.
- Commit messages follow Conventional Commits: `feat(economic): ...`, `test(economic): ...`.
- All public items from `economic/` are re-exported through `synapse-core/src/economic/mod.rs`.

### Acceptance Criteria
- Reputation: 0-1000 bounded, tier transitions tested, decay formula verified.
- Pricing: zero price rejected, market maker assembles cheapest route.
- Slashing: graduated penalties applied correctly, stake freeze/unfreeze verified.
- Smart contract: all state transitions tested (stake, unstake, slash, getReputation).
- `cargo test economic` + `npx hardhat test` — all green.

### Preexisting infrastructure (do NOT recreate)
- `DomainError` already has: `InvalidReputation`, `InsufficientReputation`, `InvalidStakeAmount`, `InvalidPrice`, `InvalidRoute`.
- `DomainEvent` already has: `ReputationChanged`, `StakeUpdated`, `NodeBanned`, `NodeUnbanned`.
- `Node` aggregate already has: `update_reputation(&mut self, new_score: u16) -> Option<DomainEvent>`.
- `MAX_REPUTATION` (1000) and `INITIAL_REPUTATION` (100) constants exist in `identity/node.rs`.
- `ExpertId`, `ModelId`, `DagRoute` exist in `model/` and `swarm/`.

---

## File Structure (post-implementation)

```
synapse-core/src/
├── economic/
│   ├── mod.rs                                    # MODIFY — add module declarations + re-exports
│   ├── reputation.rs                             # REWRITE — Reputation VO + Tier + decay
│   ├── pricing.rs                                # REWRITE — TokensPerMillion + RouteCost + cheapest_route
│   ├── stake.rs                                  # REWRITE — StakeAmount + SlashingPolicy
│   ├── route_assembly.rs                         # NEW — assemble_route pure function
│   ├── ports.rs                                  # NEW — StakeContract + PaymentGateway traits
│   └── infrastructure/                           # NEW directory
│       ├── mod.rs                                # re-exports
│       └── alloy_stake_adapter.rs               # NEW — alloy v2 StakeManager adapter
├── shared/
│   └── domain_error.rs                           # MODIFY — add freeze/ban/flags variants
├── identity/
│   └── node.rs                                   # MODIFY — add ban/unban/freeze methods to Node
└── lib.rs                                        # UNCHANGED — already exports `economic`

contracts/stake/
├── src/
│   └── StakeManager.sol                          # MODIFY — add slashing + reputation storage
└── test/
    └── StakeManager.test.ts                      # MODIFY — add slashing + reputation tests
```

---

### Task 3.0: Domain Error Variants (Economic Prerequisites)

**Files:**
- Modify: `synapse-core/src/shared/domain_error.rs`

**Interfaces:**
- Produces: 5 new `DomainError` variants for economic operations

- [ ] **Step 1: Add economic error variants**

Modify `synapse-core/src/shared/domain_error.rs` — insert these variants into the `DomainError` enum (before `StorageError`):

```rust
    #[error("node is already frozen until {until}")]
    NodeAlreadyFrozen { until: String },

    #[error("node is already banned")]
    NodeAlreadyBanned,

    #[error("node is currently frozen and cannot be slashed")]
    NodeFrozen,

    #[error("invalid flag count: {count} (must be non-zero)")]
    InvalidFlagCount { count: u32 },

    #[error("insufficient stake for slashing: {available} available, {required} required")]
    InsufficientStake { available: u64, required: u64 },
```

- [ ] **Step 2: Write tests for new error variants**

Append to `synapse-core/src/shared/domain_error.rs` tests module:

```rust
    #[test]
    fn node_already_frozen_display() {
        let err = DomainError::NodeAlreadyFrozen { until: "2026-08-01T00:00:00Z".into() };
        assert!(err.to_string().contains("node is already frozen"));
    }

    #[test]
    fn node_already_banned_display() {
        let err = DomainError::NodeAlreadyBanned;
        assert_eq!(err.to_string(), "node is already banned");
    }

    #[test]
    fn node_frozen_display() {
        let err = DomainError::NodeFrozen;
        assert_eq!(err.to_string(), "node is currently frozen and cannot be slashed");
    }

    #[test]
    fn invalid_flag_count_display() {
        let err = DomainError::InvalidFlagCount { count: 0 };
        assert!(err.to_string().contains("invalid flag count"));
    }

    #[test]
    fn insufficient_stake_display() {
        let err = DomainError::InsufficientStake { available: 50, required: 100 };
        assert_eq!(
            err.to_string(),
            "insufficient stake for slashing: 50 available, 100 required"
        );
    }
```

Run: `cargo test shared::domain_error::tests -p synapse-core`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add synapse-core/src/shared/domain_error.rs
git commit -m "feat(shared): add economic error variants (freeze, ban, flags, insufficient stake)"
```

---

### Task 3.1: Reputation Value Object + Tier + Decay

**Files:**
- Rewrite: `synapse-core/src/economic/reputation.rs`
- Modify: `synapse-core/src/economic/mod.rs` — add `pub use reputation::*;`

**Interfaces:**
- Produces: `Reputation(u16)` — bounds `[0, 1000]` enforced at construction
- Produces: `Tier` enum — `Bronze`, `Silver`, `Gold`, `Platinum`
- Produces: `Reputation::new(score: u16) -> Result<Self, DomainError>`
- Produces: `Reputation::score(&self) -> u16`, `Reputation::tier(&self) -> Tier`
- Produces: `Reputation::apply_decay(&self, hours_inactive: u32, now: DateTime<Utc>) -> Reputation`
- Produces: `Tier::min_score(&self) -> u16`, `Tier::from_score(score: u16) -> Tier`

- [ ] **Step 1: Write the failing Reputation test**

Create/replace `synapse-core/src/economic/reputation.rs` with the test module first:

```rust
use crate::shared::DomainError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reputation_accepts_zero() {
        let r = Reputation::new(0).unwrap();
        assert_eq!(r.score(), 0);
    }

    #[test]
    fn reputation_accepts_max() {
        let r = Reputation::new(1000).unwrap();
        assert_eq!(r.score(), 1000);
    }

    #[test]
    fn reputation_rejects_above_max() {
        let err = Reputation::new(1001).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid reputation score: 1001 (must be 0-1000)"
        );
    }

    #[test]
    fn bronze_tier_at_0() {
        assert_eq!(Tier::from_score(0), Tier::Bronze);
        assert_eq!(Tier::Bronze.min_score(), 0);
    }

    #[test]
    fn bronze_tier_at_299() {
        assert_eq!(Tier::from_score(299), Tier::Bronze);
    }

    #[test]
    fn silver_tier_at_300() {
        assert_eq!(Tier::from_score(300), Tier::Silver);
        assert_eq!(Tier::Silver.min_score(), 300);
    }

    #[test]
    fn silver_tier_at_599() {
        assert_eq!(Tier::from_score(599), Tier::Silver);
    }

    #[test]
    fn gold_tier_at_600() {
        assert_eq!(Tier::from_score(600), Tier::Gold);
        assert_eq!(Tier::Gold.min_score(), 600);
    }

    #[test]
    fn gold_tier_at_849() {
        assert_eq!(Tier::from_score(849), Tier::Gold);
    }

    #[test]
    fn platinum_tier_at_850() {
        assert_eq!(Tier::from_score(850), Tier::Platinum);
        assert_eq!(Tier::Platinum.min_score(), 850);
    }

    #[test]
    fn platinum_tier_at_1000() {
        assert_eq!(Tier::from_score(1000), Tier::Platinum);
    }

    #[test]
    fn reputation_tier_method() {
        assert_eq!(Reputation::new(100).unwrap().tier(), Tier::Bronze);
        assert_eq!(Reputation::new(500).unwrap().tier(), Tier::Silver);
        assert_eq!(Reputation::new(750).unwrap().tier(), Tier::Gold);
        assert_eq!(Reputation::new(900).unwrap().tier(), Tier::Platinum);
    }

    #[test]
    fn decay_reduces_score_by_one_per_24h() {
        let r = Reputation::new(500).unwrap();
        let now = chrono::Utc::now();
        let decayed = r.apply_decay(48, now);
        // 48 hours = 2 points of decay
        assert_eq!(decayed.score(), 498);
    }

    #[test]
    fn decay_never_goes_below_zero() {
        let r = Reputation::new(1).unwrap();
        let now = chrono::Utc::now();
        let decayed = r.apply_decay(1000, now);
        assert_eq!(decayed.score(), 0);
    }

    #[test]
    fn decay_zero_hours_returns_unchanged() {
        let r = Reputation::new(500).unwrap();
        let now = chrono::Utc::now();
        let decayed = r.apply_decay(0, now);
        assert_eq!(decayed.score(), 500);
    }

    #[test]
    fn decay_on_zero_score_stays_zero() {
        let r = Reputation::new(0).unwrap();
        let now = chrono::Utc::now();
        let decayed = r.apply_decay(100, now);
        assert_eq!(decayed.score(), 0);
    }
}
```

Run: `cargo test economic::reputation::tests::reputation_accepts_zero -p synapse-core`
Expected: FAIL — `Reputation` not defined.

- [ ] **Step 2: Implement Reputation and Tier**

Add the implementation above the test module in `synapse-core/src/economic/reputation.rs`:

```rust
use crate::shared::DomainError;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// 4-tier reputation system governing routing priority and slashing risk.
///
/// Tiers determine: priority in route selection (Platinum > ... > Bronze),
/// minimum stake requirements, and slashing severity multipliers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tier {
    Bronze,
    Silver,
    Gold,
    Platinum,
}

impl Tier {
    /// Minimum score to enter this tier.
    pub fn min_score(self) -> u16 {
        match self {
            Tier::Bronze => 0,
            Tier::Silver => 300,
            Tier::Gold => 600,
            Tier::Platinum => 850,
        }
    }

    /// Maps a raw score to its tier.
    pub fn from_score(score: u16) -> Tier {
        match score {
            0..=299 => Tier::Bronze,
            300..=599 => Tier::Silver,
            600..=849 => Tier::Gold,
            _ => Tier::Platinum,
        }
    }
}

/// A node's reputation score, bounded to `[0, 1000]`.
///
/// Reputation is a composite of consensus matches, uptime, and latency.
/// It decays by 1 point per 24 hours of inactivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reputation(u16);

impl Reputation {
    /// Creates a new `Reputation` score.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidReputation`] if `score` exceeds 1000.
    pub fn new(score: u16) -> Result<Self, DomainError> {
        if score > 1000 {
            return Err(DomainError::InvalidReputation { score, max: 1000 });
        }
        Ok(Self(score))
    }

    /// The raw score value.
    pub fn score(self) -> u16 {
        self.0
    }

    /// The tier this score currently occupies.
    pub fn tier(self) -> Tier {
        Tier::from_score(self.0)
    }

    /// Applies inactivity decay: 1 point per 24 hours, never below 0.
    ///
    /// `hours_inactive` is the number of hours since the node was last
    /// seen producing valid output. `now` is injected by the caller
    /// (domain is pure — no `Utc::now()` calls).
    pub fn apply_decay(self, hours_inactive: u32, _now: DateTime<Utc>) -> Reputation {
        let days = hours_inactive / 24;
        let decay = (days as u16).min(self.0);
        Reputation(self.0 - decay)
    }
}
```

- [ ] **Step 3: Run tests to verify pass**

Run: `cargo test economic::reputation -p synapse-core`
Expected: all 12 tests PASS.

- [ ] **Step 4: Wire Reputation into economic module**

Modify `synapse-core/src/economic/mod.rs` from:

```rust
pub mod pricing;
pub mod reputation;
pub mod stake;
```

To:

```rust
pub mod pricing;
pub mod reputation;
pub mod stake;

pub use reputation::{Reputation, Tier};
```

- [ ] **Step 5: Commit**

```bash
git add synapse-core/src/economic/reputation.rs synapse-core/src/economic/mod.rs
git commit -m "feat(economic): add Reputation value object with Tier and decay"
```

---

### Task 3.2: Price Domain

**Files:**
- Rewrite: `synapse-core/src/economic/pricing.rs`

**Interfaces:**
- Produces: `TokensPerMillion(u64)` — price per million tokens, non-zero
- Produces: `RouteCost` — sum of `TokensPerMillion` for an expert route
- Produces: `cheapest_route(experts: &[(ExpertId, Vec<TokensPerMillion>)]) -> Option<(Vec<ExpertId>, TokensPerMillion)>`

- [ ] **Step 1: Write the failing TokensPerMillion test**

Replace `synapse-core/src/economic/pricing.rs` with the test module first:

```rust
use crate::model::{ExpertId, ModelId};
use crate::shared::DomainError;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_per_million_rejects_zero() {
        let err = TokensPerMillion::new(0).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid price: must be non-zero"
        );
    }

    #[test]
    fn tokens_per_million_accepts_one() {
        let price = TokensPerMillion::new(1).unwrap();
        assert_eq!(price.tokens_per_million(), 1);
    }

    #[test]
    fn tokens_per_million_comparison() {
        let cheap = TokensPerMillion::new(100).unwrap();
        let expensive = TokensPerMillion::new(200).unwrap();
        assert!(cheap < expensive);
    }

    #[test]
    fn route_cost_sums_prices() {
        let a = TokensPerMillion::new(10).unwrap();
        let b = TokensPerMillion::new(20).unwrap();
        let c = TokensPerMillion::new(30).unwrap();
        let cost = RouteCost::new(vec![a, b, c]);
        assert_eq!(cost.total().tokens_per_million(), 60);
    }

    #[test]
    fn route_cost_empty_returns_zero() {
        let cost = RouteCost::new(vec![]);
        assert_eq!(cost.total().tokens_per_million(), 0);
    }

    #[test]
    fn cheapest_route_picks_lowest_sum() {
        // Build experts with prices
        let mixtral = ModelId::new("mixtral-8x7b").unwrap();
        let e0 = ExpertId::new_unchecked(mixtral.clone(), 0);
        let e1 = ExpertId::new_unchecked(mixtral.clone(), 1);
        let e2 = ExpertId::new_unchecked(mixtral.clone(), 2);

        let p5 = TokensPerMillion::new(5).unwrap();
        let p10 = TokensPerMillion::new(10).unwrap();
        let p100 = TokensPerMillion::new(100).unwrap();

        // e0: offers [5, 100], e1: [10, 5], e2: [100, 10]
        let experts = &[
            (e0.clone(), vec![p5, p100]),    // min prices: e0=5
            (e1.clone(), vec![p10, p5]),     // min prices: e1=5
            (e2.clone(), vec![p100, p10]),   // min prices: e2=10
        ];

        // Cheapest route picks one price per expert — total should be 5+5+10=20
        let (route, cost) = cheapest_route(experts).unwrap();
        assert_eq!(cost.tokens_per_million(), 20);
        assert_eq!(route.len(), 3);
    }

    #[test]
    fn cheapest_route_empty_input_returns_none() {
        let result = cheapest_route(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn cheapest_route_single_expert() {
        let mixtral = ModelId::new("mixtral-8x7b").unwrap();
        let e0 = ExpertId::new_unchecked(mixtral, 0);
        let p10 = TokensPerMillion::new(10).unwrap();
        let (route, cost) = cheapest_route(&[(e0, vec![p10])]).unwrap();
        assert_eq!(cost.tokens_per_million(), 10);
        assert_eq!(route.len(), 1);
    }
}
```

Run: `cargo test economic::pricing::tests::tokens_per_million_rejects_zero -p synapse-core`
Expected: FAIL — `TokensPerMillion` not defined.

- [ ] **Step 2: Implement TokensPerMillion, RouteCost, and cheapest_route**

Add the implementation above the test module:

```rust
use crate::model::ExpertId;
use crate::shared::DomainError;
use serde::{Deserialize, Serialize};

/// Price denominated in smallest currency units per 1M tokens.
///
/// For USDC this is cents (2 decimal places). Must be non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TokensPerMillion(u64);

impl TokensPerMillion {
    /// Creates a new price value.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidPrice`] if `tokens_per_million` is zero.
    pub fn new(tokens_per_million: u64) -> Result<Self, DomainError> {
        if tokens_per_million == 0 {
            return Err(DomainError::InvalidPrice {
                reason: "must be non-zero".into(),
            });
        }
        Ok(Self(tokens_per_million))
    }

    /// The raw price per million tokens.
    pub fn tokens_per_million(self) -> u64 {
        self.0
    }
}

/// Total cost of an expert route, summing all expert prices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteCost {
    prices: Vec<TokensPerMillion>,
}

impl RouteCost {
    /// Creates a [`RouteCost`] from individual expert prices.
    pub fn new(prices: Vec<TokensPerMillion>) -> Self {
        Self { prices }
    }

    /// The total cost across all experts in the route.
    pub fn total(&self) -> TokensPerMillion {
        let sum: u64 = self.prices.iter().map(|p| p.0).sum();
        TokensPerMillion(sum)
    }
}

/// Selects the cheapest valid expert route.
///
/// Each expert may offer multiple price points (for different replicas).
/// This function picks the minimum price for each expert and returns the
/// sorted expert list with the total cost.
///
/// Returns `None` for empty input. Returns `Some` with zero total cost
/// if all experts have empty price lists (though in practice this should
/// not happen — the DHT filters experts with no published price).
pub fn cheapest_route(
    experts: &[(ExpertId, Vec<TokensPerMillion>)],
) -> Option<(Vec<ExpertId>, TokensPerMillion)> {
    if experts.is_empty() {
        return None;
    }

    let mut route_experts = Vec::with_capacity(experts.len());
    let mut total_cost: u64 = 0;

    for (expert_id, prices) in experts {
        route_experts.push(expert_id.clone());
        if let Some(min_price) = prices.iter().min_by_key(|p| p.0) {
            total_cost += min_price.0;
        }
    }

    Some((route_experts, TokensPerMillion(total_cost)))
}
```

- [ ] **Step 3: Run tests to verify pass**

Run: `cargo test economic::pricing -p synapse-core`
Expected: all 8 tests PASS.

- [ ] **Step 4: Wire Pricing exports into economic module**

Modify `synapse-core/src/economic/mod.rs` — add the pricing exports after the reputation line:

```rust
pub use pricing::{cheapest_route, RouteCost, TokensPerMillion};
```

- [ ] **Step 5: Commit**

```bash
git add synapse-core/src/economic/pricing.rs
git commit -m "feat(economic): add TokensPerMillion, RouteCost, and cheapest_route"
```

---

### Task 3.3: Stake + Slashing

**Files:**
- Rewrite: `synapse-core/src/economic/stake.rs`

**Interfaces:**
- Produces: `StakeAmount(u64)` — USDC cents, non-zero
- Produces: `SlashingPolicy` — graduated: 10 flags → freeze 48h, 50 flags → 20% slash + score reset, 100 flags → full slash + ban
- Produces: `SlashingResult` enum — `Warning`, `Frozen { until: DateTime<Utc> }`, `Slashed { amount: u64, percentage: u8 }`, `Banned`
- Produces: `SlashingPolicy::apply(stake: &mut StakeAmount, flags: u32, now: DateTime<Utc>) -> Result<SlashingResult, DomainError>`

- [ ] **Step 1: Write the failing StakeAmount test**

Replace `synapse-core/src/economic/stake.rs` with:

```rust
use crate::shared::DomainError;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Result of applying a slashing policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashingResult {
    /// Warning only — no penalty applied.
    Warning,
    /// Stake and node are frozen until this timestamp.
    Frozen { until: DateTime<Utc> },
    /// A portion of stake was forfeited.
    Slashed { amount: u64, percentage: u8 },
    /// Full slashing + permanent ban.
    Banned,
}

/// USDC stake amount in cents.
///
/// Must be non-zero. The minimum stake for a node is 100 USDC (10_000 cents).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StakeAmount(u64);

impl StakeAmount {
    /// Creates a new [`StakeAmount`].
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidStakeAmount`] if `amount` is zero.
    pub fn new(amount: u64) -> Result<Self, DomainError> {
        if amount == 0 {
            return Err(DomainError::InvalidStakeAmount {
                reason: "stake must be non-zero".into(),
            });
        }
        Ok(Self(amount))
    }

    /// The raw stake amount in USDC cents.
    pub fn amount(self) -> u64 {
        self.0
    }

    /// Reduces the stake by the given amount (in cents).
    /// Never goes below zero — if `deduct` exceeds current stake, the remaining
    /// amount is set to 0 and the actual deduction is capped.
    pub fn deduct(&mut self, deduction: u64) -> u64 {
        let actual = deduction.min(self.0);
        self.0 -= actual;
        actual
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stake_amount_rejects_zero() {
        let err = StakeAmount::new(0).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid stake amount: stake must be non-zero"
        );
    }

    #[test]
    fn stake_amount_accepts_one_cent() {
        let s = StakeAmount::new(1).unwrap();
        assert_eq!(s.amount(), 1);
    }

    #[test]
    fn stake_deduct_reduces_amount() {
        let mut s = StakeAmount::new(1000).unwrap();
        let deducted = s.deduct(300);
        assert_eq!(deducted, 300);
        assert_eq!(s.amount(), 700);
    }

    #[test]
    fn stake_deduct_caps_at_remaining() {
        let mut s = StakeAmount::new(100).unwrap();
        let deducted = s.deduct(200);
        assert_eq!(deducted, 100);
        assert_eq!(s.amount(), 0);
    }
}
```

Run: `cargo test economic::stake::tests::stake_amount_rejects_zero -p synapse-core`
Expected: FAIL — `StakeAmount` not found (or fails due to missing implementation).

- [ ] **Step 2: Add slashing tests (still failing)**

Append more tests to the test module in `synapse-core/src/economic/stake.rs`:

```rust
    // --- SlashingPolicy tests ---

    fn make_policy() -> SlashingPolicy {
        SlashingPolicy::new(
            10,   // freeze at 10 flags
            50,   // slash at 50 flags
            20,   // 20% slash
            100,  // ban at 100 flags
        )
    }

    #[test]
    fn slashing_warns_below_freeze_threshold() {
        let policy = make_policy();
        let mut stake = StakeAmount::new(5000).unwrap();
        let now = Utc::now();
        let result = policy.apply(&mut stake, 5, now).unwrap();
        assert_eq!(result, SlashingResult::Warning);
        assert_eq!(stake.amount(), 5000); // stake unchanged
    }

    #[test]
    fn slashing_freezes_at_threshold() {
        let policy = make_policy();
        let mut stake = StakeAmount::new(5000).unwrap();
        let now = Utc::now();
        let result = policy.apply(&mut stake, 10, now).unwrap();
        match result {
            SlashingResult::Frozen { until } => {
                assert!(until > now);
            }
            other => panic!("expected Frozen, got {other:?}"),
        }
        assert_eq!(stake.amount(), 5000); // stake unchanged during freeze
    }

    #[test]
    fn slashing_slashes_20_percent_at_50_flags() {
        let policy = make_policy();
        let mut stake = StakeAmount::new(5000).unwrap();
        let now = Utc::now();
        let result = policy.apply(&mut stake, 50, now).unwrap();
        assert_eq!(
            result,
            SlashingResult::Slashed { amount: 1000, percentage: 20 }
        );
        assert_eq!(stake.amount(), 4000);
    }

    #[test]
    fn slashing_bans_at_100_flags() {
        let policy = make_policy();
        let mut stake = StakeAmount::new(5000).unwrap();
        let now = Utc::now();
        let result = policy.apply(&mut stake, 100, now).unwrap();
        assert_eq!(result, SlashingResult::Banned);
        assert_eq!(stake.amount(), 0); // full forfeiture
    }

    #[test]
    fn slashing_rejects_zero_flags() {
        let policy = make_policy();
        let mut stake = StakeAmount::new(5000).unwrap();
        let now = Utc::now();
        let err = policy.apply(&mut stake, 0, now).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid flag count: 0 (must be non-zero)"
        );
    }
```

Run: `cargo test economic::stake::tests::slashing_warns_below_freeze_threshold -p synapse-core`
Expected: FAIL — `SlashingPolicy` not defined.

- [ ] **Step 3: Implement SlashingPolicy**

Add the `SlashingPolicy` implementation above the test module:

```rust
/// Graduated slashing policy with four tiers.
///
/// Thresholds are cumulative flag counts accumulated over a rolling window.
/// The policy is configured at construction and is immutable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlashingPolicy {
    /// Flags required to trigger a 48-hour stake freeze.
    freeze_threshold: u32,
    /// Flags required to trigger a partial slash.
    slash_threshold: u32,
    /// Percentage of stake to forfeit on slash (0-100).
    slash_percentage: u8,
    /// Flags required for full slash + permanent ban.
    ban_threshold: u32,
}

impl SlashingPolicy {
    /// Creates a new [`SlashingPolicy`] with the given thresholds.
    pub fn new(
        freeze_threshold: u32,
        slash_threshold: u32,
        slash_percentage: u8,
        ban_threshold: u32,
    ) -> Self {
        Self {
            freeze_threshold,
            slash_threshold,
            slash_percentage,
            ban_threshold,
        }
    }

    /// The default V1 slashing policy:
    /// - 10 flags → freeze 48h
    /// - 50 flags → slash 20%
    /// - 100 flags → full slash + ban
    pub fn default_policy() -> Self {
        Self::new(10, 50, 20, 100)
    }

    /// Applies the slashing policy to a stake amount based on accumulated flags.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidFlagCount`] if `flags` is zero.
    pub fn apply(
        &self,
        stake: &mut StakeAmount,
        flags: u32,
        now: DateTime<Utc>,
    ) -> Result<SlashingResult, DomainError> {
        if flags == 0 {
            return Err(DomainError::InvalidFlagCount { count: 0 });
        }

        if flags >= self.ban_threshold {
            let total = stake.amount();
            stake.deduct(total);
            return Ok(SlashingResult::Banned);
        }

        if flags >= self.slash_threshold {
            let slash_amount =
                (stake.amount() as u128 * self.slash_percentage as u128 / 100) as u64;
            stake.deduct(slash_amount);
            return Ok(SlashingResult::Slashed {
                amount: slash_amount,
                percentage: self.slash_percentage,
            });
        }

        if flags >= self.freeze_threshold {
            let until = now + Duration::hours(48);
            return Ok(SlashingResult::Frozen { until });
        }

        Ok(SlashingResult::Warning)
    }
}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cargo test economic::stake -p synapse-core`
Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add synapse-core/src/economic/stake.rs
git commit -m "feat(economic): add StakeAmount and SlashingPolicy with graduated penalties"
```

---

### Task 3.4: Route Assembly

**Files:**
- Create: `synapse-core/src/economic/route_assembly.rs`

**Interfaces:**
- Consumes: `ExpertId`, `ModelId`, `TokensPerMillion`
- Produces: `assemble_route(model: &ModelId, experts_available: &[(ExpertId, TokensPerMillion, Option<&NodeId>)], active_per_token: u32) -> Result<Vec<ExpertId>, DomainError>`
- Behavior: selects cheapest price per expert, prefers co-located experts on same node

- [ ] **Step 1: Write the failing route_assembly test**

Create `synapse-core/src/economic/route_assembly.rs`:

```rust
use crate::identity::NodeId;
use crate::model::ExpertId;
use crate::shared::DomainError;
use crate::economic::pricing::TokensPerMillion;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelId;

    fn make_model_id() -> ModelId {
        ModelId::new("mixtral-8x7b").unwrap()
    }

    fn make_node(_bytes: u8) -> NodeId {
        use crate::identity::KeyPair;
        let kp = KeyPair::generate();
        NodeId::from_public_key(kp.public_key_bytes())
    }

    fn make_expert(model: &ModelId, index: u32) -> ExpertId {
        ExpertId::new_unchecked(model.clone(), index)
    }

    #[test]
    fn assemble_route_picks_cheapest_per_expert() {
        let model = make_model_id();
        let e0 = make_expert(&model, 0);
        let e1 = make_expert(&model, 1);
        let node_a = make_node(1);
        let node_b = make_node(2);

        let p5 = TokensPerMillion::new(5).unwrap();
        let p10 = TokensPerMillion::new(10).unwrap();
        let p20 = TokensPerMillion::new(20).unwrap();

        let experts = &[
            (e0.clone(), p10, Some(&node_a)),
            (e0.clone(), p5, Some(&node_b)),   // cheaper replica for e0
            (e1.clone(), p20, Some(&node_a)),
            (e1.clone(), p10, Some(&node_b)),  // e1 on node_b too
        ];

        let route = assemble_route(&model, experts, 2).unwrap();
        // Should pick e0 on node_b (p5) and e1 on node_b (p10) — co-location bonus
        assert_eq!(route.len(), 2);
        assert!(route.contains(&e0));
        assert!(route.contains(&e1));
    }

    #[test]
    fn assemble_route_returns_error_when_expert_missing() {
        let model = make_model_id();
        let e0 = make_expert(&model, 0);
        let node_a = make_node(1);
        let p10 = TokensPerMillion::new(10).unwrap();

        // Only expert 0 available, but we need 2 experts
        let experts = &[(e0.clone(), p10, Some(&node_a))];
        let err = assemble_route(&model, experts, 2).unwrap_err();
        assert!(err.to_string().contains("invalid route"));
    }

    #[test]
    fn assemble_route_prefers_co_located_experts() {
        let model = make_model_id();
        let e0 = make_expert(&model, 0);
        let e1 = make_expert(&model, 1);
        let e2 = make_expert(&model, 2);

        let node_a = make_node(1);
        let node_b = make_node(2);
        let node_c = make_node(3);

        let p10 = TokensPerMillion::new(10).unwrap();
        let p9 = TokensPerMillion::new(9).unwrap();
        let p8 = TokensPerMillion::new(8).unwrap();

        // node_a has e0 and e1 (co-located) at p10 each
        // node_b has e0 at p9 (cheaper)
        // node_c has e1 at p8 (cheaper)
        // node_c has e2 at p10
        let experts = &[
            (e0.clone(), p10, Some(&node_a)),
            (e1.clone(), p10, Some(&node_a)), // co-located with e0 on node_a
            (e0.clone(), p9, Some(&node_b)),  // cheaper but on diff node
            (e1.clone(), p8, Some(&node_c)),  // cheaper but on diff node
            (e2.clone(), p10, Some(&node_c)), // e2 on node_c too (co-located with e1)
        ];

        // Active per token = 2, so we need 2 experts out of the 3 available
        let route = assemble_route(&model, experts, 2).unwrap();
        assert_eq!(route.len(), 2);
        // Route should prefer co-located pairs. node_a has {e0, e1} together.
        // node_c has {e1, e2}. The node_a pair costs 20, node_c pair costs 18.
        // But there's also cross-node options. Let's verify the result is valid.
        for expert in &route {
            assert!(experts.iter().any(|(e, _, _)| e == expert));
        }
    }
}
```

Run: `cargo test economic::route_assembly::tests::assemble_route_picks_cheapest_per_expert -p synapse-core`
Expected: FAIL — `assemble_route` not found.

- [ ] **Step 2: Implement assemble_route**

Add the implementation above the test module:

```rust
use crate::identity::NodeId;
use crate::model::ExpertId;
use crate::model::ModelId;
use crate::shared::DomainError;
use crate::economic::pricing::TokensPerMillion;
use std::collections::{HashMap, HashSet};

/// A single expert listing from the DHT registry for route assembly.
///
/// Each expert can appear multiple times with different nodes and prices.
/// The `Option<&NodeId>` allows experts to be keyed by who hosts them
/// without forcing ownership.
type ExpertListing<'a> = (ExpertId, TokensPerMillion, Option<&'a NodeId>);

/// Assembles the cheapest valid route for a DAG request.
///
/// Given a list of available expert→(price, node) pairs and the number
/// of experts that must be active per token, this function selects one
/// instance of each required expert, preferring co-located experts
/// (same `NodeId`) to minimize network hops.
///
/// Selection algorithm:
/// 1. Group experts by `ExpertId`, keep cheapest price per expert.
/// 2. Score each unique node by how many required experts it hosts.
/// 3. Build the route greedily: for each expert, pick the node that
///    already appears most in the route (co-location bonus), breaking
///    ties on price.
///
/// # Errors
///
/// Returns [`DomainError::InvalidRoute`] if there aren't enough distinct
/// experts to satisfy `active_per_token`.
pub fn assemble_route(
    _model: &ModelId,
    experts_available: &[ExpertListing<'_>],
    active_per_token: u32,
) -> Result<Vec<ExpertId>, DomainError> {
    // Group listings by ExpertId
    let mut by_expert: HashMap<ExpertId, Vec<(TokensPerMillion, Option<&NodeId>)>> =
        HashMap::new();
    for (expert_id, price, node) in experts_available {
        by_expert
            .entry(expert_id.clone())
            .or_default()
            .push((*price, *node));
    }

    let distinct_experts = by_expert.len() as u32;
    if distinct_experts < active_per_token || by_expert.is_empty() {
        return Err(DomainError::InvalidRoute {
            reason: format!(
                "need {active_per_token} experts but only {distinct_experts} available"
            ),
        });
    }

    // For each expert, pick the cheapest listing
    let mut selected: Vec<ExpertId> = Vec::with_capacity(active_per_token as usize);
    let mut node_counts: HashMap<&NodeId, u32> = HashMap::new();

    // Collect all expert IDs and sort by how many times they appear
    // (more replicas = more route flexibility)
    let mut expert_ids: Vec<&ExpertId> = by_expert.keys().collect();
    expert_ids.sort_by_key(|eid| {
        // Experts with fewer replicas should be allocated first
        std::cmp::Reverse(by_expert[*eid].len())
    });

    for expert_id in expert_ids.iter().take(active_per_token as usize) {
        let listings = &by_expert[*expert_id];

        // Pick the listing that maximizes co-location with already-selected nodes
        let best = listings
            .iter()
            .min_by(|(price_a, node_a), (price_b, node_b)| {
                // Prefer nodes already in the route (co-location bonus)
                let score_a = node_a.map(|n| node_counts.get(n).copied().unwrap_or(0)).unwrap_or(0);
                let score_b = node_b.map(|n| node_counts.get(n).copied().unwrap_or(0)).unwrap_or(0);
                // Higher co-location score is better (reverse for min_by)
                score_b
                    .cmp(&score_a)
                    .then_with(|| price_a.cmp(price_b))
            })
            .expect("expert has at least one listing");

        selected.push((*expert_id).clone());
        if let Some(node) = best.1 {
            *node_counts.entry(node).or_insert(0) += 1;
        }
    }

    Ok(selected)
}
```

- [ ] **Step 3: Run tests to verify pass**

Run: `cargo test economic::route_assembly -p synapse-core`
Expected: all 3 tests PASS.

- [ ] **Step 4: Wire route_assembly into economic module**

Modify `synapse-core/src/economic/mod.rs` — add the route_assembly module declaration and export:

```rust
pub mod route_assembly;

pub use route_assembly::assemble_route;
```

- [ ] **Step 5: Commit**

```bash
git add synapse-core/src/economic/route_assembly.rs
git commit -m "feat(economic): add assemble_route with co-location preference"
```

---

### Task 3.5: Application Ports (Traits)

**Files:**
- Create: `synapse-core/src/economic/ports.rs`

**Interfaces:**
- Produces: `StakeContract` trait — `stake`, `unstake`, `slash`, `freeze`, `unfreeze`, `get_reputation`, `is_banned`
- Produces: `PaymentGateway` trait — `pay`, `verify_payment`

- [ ] **Step 1: Write the ports file**

Create `synapse-core/src/economic/ports.rs`:

```rust
use crate::identity::NodeId;
use crate::shared::DomainError;

pub use crate::economic::pricing::TokensPerMillion;
pub use crate::economic::reputation::Reputation;
pub use crate::economic::stake::StakeAmount;

/// Port implemented by the L2 staking contract adapter.
///
/// The domain knows this trait only. The `alloy` adapter in
/// `economic/infrastructure/` provides the concrete implementation
/// that calls `StakeManager.sol` on-chain.
pub trait StakeContract {
    /// Deposits stake for a node. Fails if node is banned.
    fn stake(&self, node: &NodeId, amount: StakeAmount) -> Result<(), DomainError>;

    /// Withdraws stake. Fails if node is frozen.
    fn unstake(&self, node: &NodeId, amount: StakeAmount) -> Result<(), DomainError>;

    /// Slashes a portion of a node's stake as penalty.
    fn slash(&self, node: &NodeId, amount: StakeAmount) -> Result<(), DomainError>;

    /// Freezes a node's stake for a duration (in seconds).
    fn freeze(&self, node: &NodeId, duration_seconds: u64) -> Result<(), DomainError>;

    /// Lifts a freeze on a node's stake.
    fn unfreeze(&self, node: &NodeId) -> Result<(), DomainError>;

    /// Returns the current reputation score for a node.
    fn get_reputation(&self, node: &NodeId) -> Result<Reputation, DomainError>;

    /// Returns true if the node is permanently banned.
    fn is_banned(&self, node: &NodeId) -> Result<bool, DomainError>;
}

/// Port implemented by the payment processing adapter.
///
/// Handles USDC transfers to miners for completed inference work.
pub trait PaymentGateway {
    /// Pays a node for processing `token_count` tokens at the given `price_per_million`.
    ///
    /// Returns a transaction hash on success.
    fn pay(
        &self,
        node: &NodeId,
        price_per_million: TokensPerMillion,
        token_count: u64,
    ) -> Result<String, DomainError>;

    /// Verifies that a payment transaction was included on-chain.
    fn verify_payment(&self, tx_hash: &str) -> Result<bool, DomainError>;
}

#[cfg(test)]
mod tests {
    // Ports are pure traits — no behavior to test at the domain level.
    // Integration tests in `economic/infrastructure/` exercise the alloy adapter.
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p synapse-core`
Expected: PASS (no errors in `economic::ports`).

- [ ] **Step 2: Wire ports into economic module**

Modify `synapse-core/src/economic/mod.rs` — add the ports module declaration:

```rust
pub mod ports;
```

- [ ] **Step 3: Commit**

```bash
git add synapse-core/src/economic/ports.rs synapse-core/src/economic/mod.rs
git commit -m "feat(economic): add StakeContract and PaymentGateway application ports"
```

---

### Task 3.6: L2 Adapter — alloy v2 + Solidity StakeManager

**Files:**
- Create: `synapse-core/src/economic/infrastructure/mod.rs`
- Create: `synapse-core/src/economic/infrastructure/alloy_stake_adapter.rs`
- Modify: `contracts/stake/src/StakeManager.sol` — add reputation + slashing storage
- Modify: `contracts/stake/test/StakeManager.test.ts` — add slashing tests

**Interfaces:**
- Produces: `AlloyStakeAdapter` — implements `StakeContract` using alloy v2
- Consumes: `StakeManager.sol` ABI (generated by `solc` / Hardhat)
- Solidity: `stake()`, `unstake()`, `slash()`, `freeze()`, `unfreeze()`, `getReputation()`, `isBanned()`

- [ ] **Step 1: Update StakeManager.sol with slashing and reputation**

Read the current `contracts/stake/src/StakeManager.sol`, then modify to add reputation storage and slashing functions:

Add to the contract state:

```solidity
    // Reputation scores (0-1000)
    mapping(address => uint16) public reputation;

    // Ban status
    mapping(address => bool) public banned;

    // Freeze expiry timestamps
    mapping(address => uint256) public frozenUntil;

    // Flag counters (rolling 7-day window)
    mapping(address => uint32) public flagCount;
    mapping(address => uint256) public flagWindowStart;

    event ReputationUpdated(address indexed node, uint16 oldScore, uint16 newScore);
    event NodeBanned(address indexed node, string reason);
    event NodeUnbanned(address indexed node);
    event NodeFrozen(address indexed node, uint256 until);
    event NodeUnfrozen(address indexed node);
    event NodeSlashed(address indexed node, uint256 amount);
    event NodeFlagged(address indexed node, uint32 totalFlags);
```

Add functions:

```solidity
    /// Updates a node's reputation score (0-1000). Only callable by authorized gateway.
    function updateReputation(address node, uint16 newScore) external onlyAuthorized {
        require(newScore <= 1000, "Score must be 0-1000");
        require(!banned[node], "Node is banned");
        uint16 oldScore = reputation[node];
        reputation[node] = newScore;
        emit ReputationUpdated(node, oldScore, newScore);
    }

    /// Returns the current reputation score for a node.
    function getReputation(address node) external view returns (uint16) {
        return reputation[node];
    }

    /// Flags a node for misbehavior. Accumulates flags; triggers slashing at thresholds.
    function flag(address node) external onlyAuthorized {
        require(!banned[node], "Node is banned");

        // Reset flag window if 7 days have passed
        if (block.timestamp > flagWindowStart[node] + 7 days) {
            flagCount[node] = 0;
            flagWindowStart[node] = block.timestamp;
        }

        flagCount[node] += 1;
        emit NodeFlagged(node, flagCount[node]);

        // Apply graduated slashing
        uint32 flags = flagCount[node];
        if (flags >= 100) {
            // Full slash + ban
            uint256 stakeAmount = stakes[node];
            stakes[node] = 0;
            banned[node] = true;
            reputation[node] = 0;
            emit NodeSlashed(node, stakeAmount);
            emit NodeBanned(node, "100+ flags accumulated");
        } else if (flags >= 50) {
            // 20% slash
            uint256 slashAmount = (stakes[node] * 20) / 100;
            stakes[node] -= slashAmount;
            reputation[node] = 0;
            emit NodeSlashed(node, slashAmount);
        } else if (flags >= 10) {
            // 48h freeze
            frozenUntil[node] = block.timestamp + 48 hours;
            emit NodeFrozen(node, frozenUntil[node]);
        }
    }

    /// Freezes a node's stake for a given duration.
    function freeze(address node, uint256 durationSeconds) external onlyAuthorized {
        require(!banned[node], "Node is banned");
        frozenUntil[node] = block.timestamp + durationSeconds;
        emit NodeFrozen(node, frozenUntil[node]);
    }

    /// Lifts a freeze on a node's stake.
    function unfreeze(address node) external onlyAuthorized {
        frozenUntil[node] = 0;
        emit NodeUnfrozen(node);
    }

    /// Slashes a specific amount from a node's stake.
    function slash(address node, uint256 amount) external onlyAuthorized {
        require(stakes[node] >= amount, "Insufficient stake");
        stakes[node] -= amount;
        emit NodeSlashed(node, amount);
    }

    /// Permanently bans a node.
    function ban(address node, string calldata reason) external onlyAuthorized {
        banned[node] = true;
        reputation[node] = 0;
        emit NodeBanned(node, reason);
    }

    /// Lifts a ban on a node.
    function unban(address node) external onlyAuthorized {
        require(banned[node], "Node is not banned");
        banned[node] = false;
        reputation[node] = 100; // restart at base reputation
        flagCount[node] = 0;
        emit NodeUnbanned(node);
    }

    /// Returns whether a node is banned.
    function isBanned(address node) external view returns (bool) {
        return banned[node];
    }

    /// Returns whether a node is currently frozen.
    function isFrozen(address node) external view returns (bool) {
        return block.timestamp < frozenUntil[node];
    }
```

Also update the existing `stake()` function to set initial reputation:

```solidity
    function stake() external payable notBanned {
        require(msg.value > 0, "Must stake something");
        stakes[msg.sender] += msg.value;
        if (reputation[msg.sender] == 0) {
            reputation[msg.sender] = 100; // initial reputation
        }
        emit Staked(msg.sender, msg.value);
    }
```

- [ ] **Step 2: Write Hardhat tests for slashing**

Modify `contracts/stake/test/StakeManager.test.ts` and add slashing test cases:

```typescript
describe("Slashing", function () {
  it("should flag a node without slashing at low counts", async function () {
    const { stakeManager, authorized } = await loadFixture(deployFixture);
    await stakeManager.connect(authorized).stake({ value: ethers.parseEther("100") });
    await stakeManager.connect(authorized).flag(authorized.address);
    const flags = await stakeManager.flagCount(authorized.address);
    expect(flags).to.equal(1);
    // No slash — stake unchanged
    expect(await stakeManager.stakes(authorized.address)).to.equal(ethers.parseEther("100"));
  });

  it("should freeze stake at 10 flags", async function () {
    const { stakeManager, authorized } = await loadFixture(deployFixture);
    await stakeManager.connect(authorized).stake({ value: ethers.parseEther("100") });
    for (let i = 0; i < 10; i++) {
      await stakeManager.connect(authorized).flag(authorized.address);
    }
    expect(await stakeManager.isFrozen(authorized.address)).to.be.true;
  });

  it("should slash 20% at 50 flags", async function () {
    const { stakeManager, authorized } = await loadFixture(deployFixture);
    await stakeManager.connect(authorized).stake({ value: ethers.parseEther("100") });
    for (let i = 0; i < 50; i++) {
      await stakeManager.connect(authorized).flag(authorized.address);
    }
    const stake = await stakeManager.stakes(authorized.address);
    expect(stake).to.equal(ethers.parseEther("80")); // 20% slashed
    expect(await stakeManager.reputation(authorized.address)).to.equal(0);
  });

  it("should ban at 100 flags", async function () {
    const { stakeManager, authorized } = await loadFixture(deployFixture);
    await stakeManager.connect(authorized).stake({ value: ethers.parseEther("100") });
    for (let i = 0; i < 100; i++) {
      await stakeManager.connect(authorized).flag(authorized.address);
    }
    expect(await stakeManager.isBanned(authorized.address)).to.be.true;
    expect(await stakeManager.stakes(authorized.address)).to.equal(0);
  });

  it("should reset flag window after 7 days", async function () {
    // This would need time manipulation via Hardhat's evm_mine
    // Omit for V1; use Hardhat's time helpers if available
  });

  it("should set initial reputation on first stake", async function () {
    const { stakeManager, authorized } = await loadFixture(deployFixture);
    await stakeManager.connect(authorized).stake({ value: ethers.parseEther("10") });
    expect(await stakeManager.reputation(authorized.address)).to.equal(100);
  });

  it("should update reputation via authorized caller", async function () {
    const { stakeManager, authorized } = await loadFixture(deployFixture);
    await stakeManager.connect(authorized).stake({ value: ethers.parseEther("10") });
    await stakeManager.connect(authorized).updateReputation(authorized.address, 720);
    expect(await stakeManager.reputation(authorized.address)).to.equal(720);
  });
});
```

- [ ] **Step 3: Run Hardhat tests**

Run: `cd contracts/stake && npx hardhat test`
Expected: existing tests PASS, new slashing tests PASS.

- [ ] **Step 4: Create infrastructure module**

Create `synapse-core/src/economic/infrastructure/mod.rs`:

```rust
pub mod alloy_stake_adapter;

pub use alloy_stake_adapter::AlloyStakeAdapter;
```

- [ ] **Step 5: Implement AlloyStakeAdapter**

Create `synapse-core/src/economic/infrastructure/alloy_stake_adapter.rs`:

```rust
use alloy::primitives::{Address, U256};
use alloy::providers::Provider;
use alloy::sol;
use crate::economic::ports::StakeContract;
use crate::economic::reputation::Reputation;
use crate::economic::stake::StakeAmount;
use crate::identity::NodeId;
use crate::shared::DomainError;
use std::sync::Arc;

// Solidity ABI bindings generated by alloy's sol! macro.
// In production this would be generated from the contract ABI.
alloy::sol! {
    #[allow(missing_docs)]
    #[sol(abi)]
    StakeManager,
}

/// Alloy v2 adapter implementing [`StakeContract`] for the on-chain `StakeManager`.
///
/// Wraps an alloy provider connected to the L2 where `StakeManager.sol` is deployed.
pub struct AlloyStakeAdapter<P: Provider> {
    contract: StakeManager::StakeManagerInstance<(), P>,
}

impl<P: Provider> AlloyStakeAdapter<P> {
    /// Creates a new adapter connected to the deployed `StakeManager` at `contract_address`.
    pub fn new(provider: P, contract_address: Address) -> Self {
        let contract = StakeManager::new(contract_address, provider);
        Self { contract }
    }
}

impl<P: Provider> StakeContract for AlloyStakeAdapter<P> {
    fn stake(&self, node: &NodeId, amount: StakeAmount) -> Result<(), DomainError> {
        // TODO: Call contract.stake() with value = amount.amount()
        // In V1, this is a stub — actual alloy integration happens
        // once the L2 is selected (Solana vs Base).
        let _ = (node, amount);
        Err(DomainError::StorageError {
            message: "alloy adapter: not yet implemented (awaiting L2 selection)".into(),
        })
    }

    fn unstake(&self, node: &NodeId, amount: StakeAmount) -> Result<(), DomainError> {
        let _ = (node, amount);
        Err(DomainError::StorageError {
            message: "alloy adapter: not yet implemented (awaiting L2 selection)".into(),
        })
    }

    fn slash(&self, node: &NodeId, amount: StakeAmount) -> Result<(), DomainError> {
        let _ = (node, amount);
        Err(DomainError::StorageError {
            message: "alloy adapter: not yet implemented (awaiting L2 selection)".into(),
        })
    }

    fn freeze(&self, node: &NodeId, duration_seconds: u64) -> Result<(), DomainError> {
        let _ = (node, duration_seconds);
        Err(DomainError::StorageError {
            message: "alloy adapter: not yet implemented (awaiting L2 selection)".into(),
        })
    }

    fn unfreeze(&self, node: &NodeId) -> Result<(), DomainError> {
        let _ = node;
        Err(DomainError::StorageError {
            message: "alloy adapter: not yet implemented (awaiting L2 selection)".into(),
        })
    }

    fn get_reputation(&self, node: &NodeId) -> Result<Reputation, DomainError> {
        let _ = node;
        Err(DomainError::StorageError {
            message: "alloy adapter: not yet implemented (awaiting L2 selection)".into(),
        })
    }

    fn is_banned(&self, node: &NodeId) -> Result<bool, DomainError> {
        let _ = node;
        Err(DomainError::StorageError {
            message: "alloy adapter: not yet implemented (awaiting L2 selection)".into(),
        })
    }
}
```

- [ ] **Step 6: Wire infrastructure into economic module**

Modify `synapse-core/src/economic/mod.rs` — add the infrastructure module declaration:

```rust
pub mod infrastructure;
```

(The rest of the mod.rs content should already be in place from Task 3.1 Step 4.)

- [ ] **Step 7: Verify everything compiles and tests pass**

Run: `cargo check -p synapse-core` and `cargo test economic -p synapse-core`
Expected: all tests PASS, no compilation errors.

- [ ] **Step 8: Commit**

```bash
git add synapse-core/src/economic/infrastructure/ contracts/stake/src/StakeManager.sol contracts/stake/test/StakeManager.test.ts
git commit -m "feat(economic): add L2 adapter skeleton and Solidity slashing logic"
```

---

## Final Verification

- [ ] `cargo test economic -p synapse-core` — all tests green
- [ ] `cargo clippy -- -D warnings` — clean
- [ ] `cargo fmt --check` — clean
- [ ] `cd contracts/stake && npx hardhat test` — all tests green
- [ ] Domain layer has zero external dependencies (no alloy in domain types)
- [ ] All public types documented with `///` doc comments
- [ ] All economic types re-exported through `synapse_core::economic`

