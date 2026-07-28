### Task 2.3: Speculative Swarm Domain

**Files:**
- Create: `synapse-core/src/swarm/speculative.rs`
- Modify: `synapse-core/src/swarm/mod.rs` — re-export `SpecSwarmConfig`

**Interfaces:**
- Produces: `SpecSwarmConfig { swarm_size: u32, seeds: Vec<u32>, model: ModelId }`
- Produces: `SpecSwarmConfig::new(model: ModelId, swarm_size: u32) -> Result<Self, DomainError>`
- Produces: `SpecSwarmConfig::seeds(&self) -> &[u32]`
- Produces: `SpecSwarmConfig::quorum(&self) -> usize` — majority threshold

- [ ] **Step 1: Write failing SpecSwarmConfig tests**

Create `synapse-core/src/swarm/speculative.rs` with this test module:

```rust
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
}
```

Run: `cargo test swarm::speculative::tests::rejects_swarm_size_below_minimum -p synapse-core`
Expected: FAIL with `SpecSwarmConfig` not found.

- [ ] **Step 2: Implement SpecSwarmConfig**

Add the implementation to `synapse-core/src/swarm/speculative.rs`:

```rust
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
        if swarm_size < MIN_SWARM_SIZE || swarm_size > MAX_SWARM_SIZE {
            return Err(DomainError::InvalidSwarmSize { size: swarm_size });
        }
        let seeds = (1..=swarm_size).map(|i| i as u32).collect();
        Ok(Self {
            model,
            swarm_size,
            seeds,
        })
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
```

- [ ] **Step 3: Re-export SpecSwarmConfig**

Modify `synapse-core/src/swarm/mod.rs`:

```rust
pub use speculative::SpecSwarmConfig;
```

Run: `cargo test swarm::speculative -p synapse-core`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add synapse-core/src/swarm/speculative.rs synapse-core/src/swarm/mod.rs
git commit -m "feat(swarm): add speculative swarm config with seed distribution"
```

---

