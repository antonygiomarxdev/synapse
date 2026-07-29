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
    ///
    /// Never goes below zero — if `deduct` exceeds current stake, the remaining
    /// amount is set to 0 and the actual deduction is capped.
    pub fn deduct(&mut self, deduction: u64) -> u64 {
        let actual = deduction.min(self.0);
        self.0 -= actual;
        actual
    }
}

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
        Self { freeze_threshold, slash_threshold, slash_percentage, ban_threshold }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stake_amount_rejects_zero() {
        let err = StakeAmount::new(0).unwrap_err();
        assert_eq!(err.to_string(), "invalid stake amount: stake must be non-zero");
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
    // --- SlashingPolicy tests ---

    fn make_policy() -> SlashingPolicy {
        SlashingPolicy::new(
            10,  // freeze at 10 flags
            50,  // slash at 50 flags
            20,  // 20% slash
            100, // ban at 100 flags
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
        assert_eq!(result, SlashingResult::Slashed { amount: 1000, percentage: 20 });
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
        assert_eq!(err.to_string(), "invalid flag count: 0 (must be non-zero)");
    }
}
