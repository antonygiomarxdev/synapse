use crate::shared::DomainError;
use serde::{Deserialize, Serialize};

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
}
