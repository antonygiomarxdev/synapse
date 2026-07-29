use crate::shared::DomainError;
use serde::{Deserialize, Serialize};

/// Maximum possible reputation score.
pub const MAX_REPUTATION: u16 = 1000;
/// Minimum score to enter Silver tier.
pub const SILVER_THRESHOLD: u16 = 300;
/// Minimum score to enter Gold tier.
pub const GOLD_THRESHOLD: u16 = 600;
/// Minimum score to enter Platinum tier.
pub const PLATINUM_THRESHOLD: u16 = 850;
/// Number of hours of inactivity before 1 reputation point decays.
pub const DECAY_HOURS_PER_POINT: u32 = 24;
/// Maximum score that qualifies as Bronze (`SILVER_THRESHOLD - 1`).
pub const BRONZE_MAX: u16 = SILVER_THRESHOLD - 1;
/// Maximum score that qualifies as Silver (`GOLD_THRESHOLD - 1`).
pub const SILVER_MAX: u16 = GOLD_THRESHOLD - 1;
/// Maximum score that qualifies as Gold (`PLATINUM_THRESHOLD - 1`).
pub const GOLD_MAX: u16 = PLATINUM_THRESHOLD - 1;

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
            Tier::Silver => SILVER_THRESHOLD,
            Tier::Gold => GOLD_THRESHOLD,
            Tier::Platinum => PLATINUM_THRESHOLD,
        }
    }

    /// Maps a raw score to its tier.
    pub fn from_score(score: u16) -> Tier {
        match score {
            0..=BRONZE_MAX => Tier::Bronze,
            SILVER_THRESHOLD..=SILVER_MAX => Tier::Silver,
            GOLD_THRESHOLD..=GOLD_MAX => Tier::Gold,
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
        if score > MAX_REPUTATION {
            return Err(DomainError::InvalidReputation { score, max: MAX_REPUTATION });
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
    /// seen producing valid output.
    pub fn apply_decay(self, hours_inactive: u32) -> Reputation {
        let days = hours_inactive / DECAY_HOURS_PER_POINT;
        let decay = (days as u16).min(self.0);
        Reputation(self.0 - decay)
    }
}

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
        let r = Reputation::new(MAX_REPUTATION).unwrap();
        assert_eq!(r.score(), MAX_REPUTATION);
    }

    #[test]
    fn reputation_rejects_above_max() {
        let err = Reputation::new(MAX_REPUTATION + 1).unwrap_err();
        assert_eq!(err.to_string(), "invalid reputation score: 1001 (must be 0-1000)");
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
        let decayed = r.apply_decay(48);
        assert_eq!(decayed.score(), 498);
    }

    #[test]
    fn decay_never_goes_below_zero() {
        let r = Reputation::new(1).unwrap();
        let decayed = r.apply_decay(1000);
        assert_eq!(decayed.score(), 0);
    }

    #[test]
    fn decay_zero_hours_returns_unchanged() {
        let r = Reputation::new(500).unwrap();
        let decayed = r.apply_decay(0);
        assert_eq!(decayed.score(), 500);
    }

    #[test]
    fn decay_on_zero_score_stays_zero() {
        let r = Reputation::new(0).unwrap();
        let decayed = r.apply_decay(100);
        assert_eq!(decayed.score(), 0);
    }
}
