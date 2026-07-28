### Task 2.5: Re-Sync Policy

**Files:**
- Create: `synapse-core/src/swarm/resync.rs`
- Modify: `synapse-core/src/swarm/mod.rs` — re-export `ReSyncPolicy`

**Interfaces:**
- Consumes: `NodeId`, `Token`
- Produces: `ReSyncPolicy { divergence_limit: u32, expulsion_limit: u32, chronic_window_hours: u32, chronic_flag_threshold: u32 }`
- Produces: `ReSyncPolicy::default()`
- Produces: `ReSyncPolicy::record_divergence(&mut self, node_id: NodeId, token: &Token)`
- Produces: `ReSyncPolicy::should_expel(&self, node_id: &NodeId) -> bool`
- Produces: `ReSyncPolicy::record_chronic_flag(&mut self, node_id: NodeId)`
- Produces: `ReSyncPolicy::is_chronic(&self, node_id: &NodeId) -> bool`

- [ ] **Step 1: Write failing ReSyncPolicy tests**

Create `synapse-core/src/swarm/resync.rs` with this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::NodeId;
    use crate::swarm::Token;

    fn node(id: u8) -> NodeId {
        let bytes = [id; 32];
        NodeId::from_public_key(&bytes)
    }

    fn token(text: &str) -> Token {
        Token::new(text, -0.5).unwrap()
    }

    #[test]
    fn node_expelled_after_three_divergences() {
        let mut policy = ReSyncPolicy::default();
        let n = node(1);
        policy.record_divergence(n, &token("a"));
        policy.record_divergence(n, &token("b"));
        assert!(!policy.should_expel(&n));
        policy.record_divergence(n, &token("c"));
        assert!(policy.should_expel(&n));
    }

    #[test]
    fn expulsion_resets_after_request_boundary() {
        let mut policy = ReSyncPolicy::default();
        let n = node(1);
        policy.record_divergence(n, &token("a"));
        policy.record_divergence(n, &token("b"));
        policy.record_divergence(n, &token("c"));
        assert!(policy.should_expel(&n));
        policy.reset_request();
        assert!(!policy.should_expel(&n));
    }

    #[test]
    fn chronic_flag_after_ten_flags() {
        let mut policy = ReSyncPolicy::default();
        let n = node(1);
        for i in 0..9 {
            policy.record_chronic_flag(n);
            assert!(!policy.is_chronic(&n), "flag {} should not be chronic", i + 1);
        }
        policy.record_chronic_flag(n);
        assert!(policy.is_chronic(&n));
    }

    #[test]
    fn different_nodes_tracked_independently() {
        let mut policy = ReSyncPolicy::default();
        let a = node(1);
        let b = node(2);
        for _ in 0..3 {
            policy.record_divergence(a, &token("x"));
        }
        assert!(policy.should_expel(&a));
        assert!(!policy.should_expel(&b));
    }
}
```

Run: `cargo test swarm::resync::tests::node_expelled_after_three_divergences -p synapse-core`
Expected: FAIL with `ReSyncPolicy` not found.

- [ ] **Step 2: Implement ReSyncPolicy**

Add the implementation to `synapse-core/src/swarm/resync.rs`:

```rust
use crate::identity::NodeId;
use crate::swarm::Token;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-request and chronic divergence tracking.
///
/// Nodes that diverge too often in a single request are expelled.
/// Chronic divergers (10+ flags in 24 hours) are flagged for slashing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReSyncPolicy {
    divergence_limit: u32,
    expulsion_limit: u32,
    chronic_flag_threshold: u32,
    per_request_divergences: HashMap<NodeId, u32>,
    chronic_flags: HashMap<NodeId, u32>,
}

impl Default for ReSyncPolicy {
    fn default() -> Self {
        Self {
            divergence_limit: 3,
            expulsion_limit: 3,
            chronic_flag_threshold: 10,
            per_request_divergences: HashMap::new(),
            chronic_flags: HashMap::new(),
        }
    }
}

impl ReSyncPolicy {
    /// Creates a policy with custom limits.
    pub fn new(divergence_limit: u32, expulsion_limit: u32, chronic_flag_threshold: u32) -> Self {
        Self {
            divergence_limit,
            expulsion_limit,
            chronic_flag_threshold,
            per_request_divergences: HashMap::new(),
            chronic_flags: HashMap::new(),
        }
    }

    /// Records a divergence for a node in the current request.
    pub fn record_divergence(&mut self, node_id: NodeId, _token: &Token) {
        *self.per_request_divergences.entry(node_id).or_insert(0) += 1;
    }

    /// True if the node has reached the expulsion threshold.
    pub fn should_expel(&self, node_id: &NodeId) -> bool {
        self.per_request_divergences
            .get(node_id)
            .copied()
            .unwrap_or(0)
            >= self.expulsion_limit
    }

    /// Records a chronic flag (one per request where the node was expelled
    /// or audited as malicious).
    pub fn record_chronic_flag(&mut self, node_id: NodeId) {
        *self.chronic_flags.entry(node_id).or_insert(0) += 1;
    }

    /// True if the node has crossed the chronic flag threshold.
    pub fn is_chronic(&self, node_id: &NodeId) -> bool {
        self.chronic_flags
            .get(node_id)
            .copied()
            .unwrap_or(0)
            >= self.chronic_flag_threshold
    }

    /// Resets per-request counters at the end of a request.
    pub fn reset_request(&mut self) {
        self.per_request_divergences.clear();
    }

    /// Divergence threshold before expulsion.
    pub fn divergence_limit(&self) -> u32 {
        self.divergence_limit
    }

    /// Exact number of divergences that triggers expulsion.
    pub fn expulsion_limit(&self) -> u32 {
        self.expulsion_limit
    }

    /// Number of chronic flags required for a slashing freeze.
    pub fn chronic_flag_threshold(&self) -> u32 {
        self.chronic_flag_threshold
    }
}
```

- [ ] **Step 3: Re-export ReSyncPolicy**

Modify `synapse-core/src/swarm/mod.rs`:

```rust
pub use resync::ReSyncPolicy;
```

Run: `cargo test swarm::resync -p synapse-core`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add synapse-core/src/swarm/resync.rs synapse-core/src/swarm/mod.rs
git commit -m "feat(swarm): add re-sync policy with divergence and chronic flag tracking"
```

---

