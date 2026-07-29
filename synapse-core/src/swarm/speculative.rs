use crate::model::ModelId;
use crate::shared::DomainError;
use serde::{Deserialize, Serialize};

const MIN_SWARM_SIZE: u32 = 2;
const MAX_SWARM_SIZE: u32 = 32;

/// Configuration for the speculative (realtime) swarm.
///
/// Each node in the swarm runs the full model with a different seed so
/// ensemble voting can detect malicious or buggy outputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecSwarmConfig {
    model: ModelId,
    swarm_size: u32,
    seeds: Vec<u32>,
}

impl SpecSwarmConfig {
    /// Creates a speculative swarm configuration.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidSwarmSize`] if `swarm_size` is
    /// outside `[2, 32]`.
    pub fn new(model: ModelId, swarm_size: u32) -> Result<Self, DomainError> {
        if !(MIN_SWARM_SIZE..=MAX_SWARM_SIZE).contains(&swarm_size) {
            return Err(DomainError::InvalidSwarmSize { size: swarm_size });
        }
        let seeds = (1..=swarm_size).collect();
        Ok(Self { model, swarm_size, seeds })
    }

    /// The model served by the swarm.
    pub fn model(&self) -> &ModelId {
        &self.model
    }

    /// Number of nodes in the swarm.
    pub fn swarm_size(&self) -> u32 {
        self.swarm_size
    }

    /// Unique seeds assigned to each node (1-based).
    pub fn seeds(&self) -> &[u32] {
        &self.seeds
    }

    /// Minimum agreeing nodes needed for consensus.
    ///
    /// Majority is `floor(swarm_size / 2) + 1`.
    pub fn quorum(&self) -> usize {
        (self.swarm_size as usize / 2) + 1
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelId;

    fn model() -> ModelId {
        ModelId::new("kimi-k3").unwrap()
    }

    #[test]
    fn rejects_swarm_size_below_minimum() {
        let result = SpecSwarmConfig::new(model(), 1);
        assert!(result.is_err());
    }

    #[test]
    fn rejects_swarm_size_above_maximum() {
        let result = SpecSwarmConfig::new(model(), 33);
        assert!(result.is_err());
    }

    #[test]
    fn valid_size_5_has_unique_seeds() {
        let config = SpecSwarmConfig::new(model(), 5).unwrap();
        assert_eq!(config.seeds().len(), 5);
        let unique: std::collections::HashSet<_> = config.seeds().iter().collect();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn quorum_for_size_5_is_3() {
        let config = SpecSwarmConfig::new(model(), 5).unwrap();
        assert_eq!(config.quorum(), 3);
    }

    #[test]
    fn quorum_for_size_8_is_5() {
        let config = SpecSwarmConfig::new(model(), 8).unwrap();
        assert_eq!(config.quorum(), 5);
    }

    #[test]
    fn quorum_for_size_3_is_2() {
        let config = SpecSwarmConfig::new(model(), 3).unwrap();
        assert_eq!(config.quorum(), 2);
    }

    #[test]
    fn swarm_size_returns_configured_value() {
        let config = SpecSwarmConfig::new(model(), 5).unwrap();
        assert_eq!(config.swarm_size(), 5);
    }
}
