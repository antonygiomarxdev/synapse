use crate::identity::NodeId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Seconds in one hour, used for chronic-flag window calculation.
const SECONDS_PER_HOUR: i64 = 3600;

/// Per-request and chronic divergence tracking.
///
/// Nodes that diverge too often in a single request are expelled.
/// Chronic divergers (10+ flags within `chronic_window_hours`) are flagged for slashing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReSyncPolicy {
    divergence_limit: u32,
    expulsion_limit: u32,
    chronic_flag_threshold: u32,
    chronic_window_hours: u32,
    per_request_divergences: HashMap<NodeId, u32>,
    chronic_flags: HashMap<NodeId, Vec<i64>>,
}

impl Default for ReSyncPolicy {
    fn default() -> Self {
        Self {
            divergence_limit: 3,
            expulsion_limit: 3,
            chronic_flag_threshold: 10,
            chronic_window_hours: 24,
            per_request_divergences: HashMap::new(),
            chronic_flags: HashMap::new(),
        }
    }
}

impl ReSyncPolicy {
    /// Creates a policy with custom limits.
    pub fn new(
        divergence_limit: u32,
        expulsion_limit: u32,
        chronic_flag_threshold: u32,
        chronic_window_hours: u32,
    ) -> Self {
        Self {
            divergence_limit,
            expulsion_limit,
            chronic_flag_threshold,
            chronic_window_hours,
            per_request_divergences: HashMap::new(),
            chronic_flags: HashMap::new(),
        }
    }

    /// Records a divergence for a node in the current request.
    pub fn record_divergence(&mut self, node_id: NodeId) {
        *self.per_request_divergences.entry(node_id).or_insert(0) += 1;
    }

    /// True if the node has reached the expulsion threshold.
    pub fn should_expel(&self, node_id: &NodeId) -> bool {
        self.per_request_divergences.get(node_id).copied().unwrap_or(0) >= self.expulsion_limit
    }

    /// Records a chronic flag with the current epoch timestamp.
    /// One per request where the node was expelled or audited as malicious.
    pub fn record_chronic_flag(&mut self, node_id: NodeId, now_epoch_secs: i64) {
        self.chronic_flags.entry(node_id).or_default().push(now_epoch_secs);
    }

    /// True if the node has crossed the chronic flag threshold
    /// within the `chronic_window_hours` window.
    pub fn is_chronic(&self, node_id: &NodeId, now_epoch_secs: i64) -> bool {
        let cutoff = now_epoch_secs - (self.chronic_window_hours as i64) * SECONDS_PER_HOUR;
        let count = self
            .chronic_flags
            .get(node_id)
            .map(|timestamps| timestamps.iter().filter(|&&ts| ts > cutoff).count() as u32)
            .unwrap_or(0);
        count >= self.chronic_flag_threshold
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

    /// Number of hours in the chronic-flag window.
    pub fn chronic_window_hours(&self) -> u32 {
        self.chronic_window_hours
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::NodeId;

    fn node(id: u8) -> NodeId {
        let bytes = [id; 32];
        NodeId::from_public_key(&bytes)
    }

    #[test]
    fn node_expelled_after_three_divergences() {
        let mut policy = ReSyncPolicy::default();
        let n = node(1);
        policy.record_divergence(n);
        policy.record_divergence(n);
        assert!(!policy.should_expel(&n));
        policy.record_divergence(n);
        assert!(policy.should_expel(&n));
    }

    #[test]
    fn expulsion_resets_after_request_boundary() {
        let mut policy = ReSyncPolicy::default();
        let n = node(1);
        policy.record_divergence(n);
        policy.record_divergence(n);
        policy.record_divergence(n);
        assert!(policy.should_expel(&n));
        policy.reset_request();
        assert!(!policy.should_expel(&n));
    }

    #[test]
    fn chronic_flag_after_ten_flags() {
        let mut policy = ReSyncPolicy::default();
        let n = node(1);
        let now = 1_000_000_000i64;
        for i in 0..9 {
            policy.record_chronic_flag(n, now);
            assert!(!policy.is_chronic(&n, now), "flag {} should not be chronic", i + 1);
        }
        policy.record_chronic_flag(n, now);
        assert!(policy.is_chronic(&n, now));
    }

    #[test]
    fn chronic_window_filters_old_flags() {
        let mut policy = ReSyncPolicy::new(3, 3, 5, 24);
        let n = node(1);
        let now = 1_000_000_000i64;
        // 5 flags within window (recent)
        for _ in 0..5 {
            policy.record_chronic_flag(n, now);
        }
        // 5 flags outside window (2 days ago)
        let old = now - 48 * SECONDS_PER_HOUR;
        for _ in 0..5 {
            policy.record_chronic_flag(n, old);
        }
        // Only the 5 recent flags count, which is >= threshold of 5
        assert!(policy.is_chronic(&n, now));
        // But if we look from a later time, the recent ones become old too
        let later = now + 48 * SECONDS_PER_HOUR;
        assert!(!policy.is_chronic(&n, later));
    }

    #[test]
    fn different_nodes_tracked_independently() {
        let mut policy = ReSyncPolicy::default();
        let a = node(1);
        let b = node(2);
        for _ in 0..3 {
            policy.record_divergence(a);
        }
        assert!(policy.should_expel(&a));
        assert!(!policy.should_expel(&b));
    }

    #[test]
    fn accessors_reflect_configured_values() {
        let policy = ReSyncPolicy::new(5, 7, 15, 48);
        assert_eq!(policy.divergence_limit(), 5);
        assert_eq!(policy.expulsion_limit(), 7);
        assert_eq!(policy.chronic_flag_threshold(), 15);
        assert_eq!(policy.chronic_window_hours(), 48);
    }

    #[test]
    fn is_chronic_at_exact_threshold() {
        let mut policy = ReSyncPolicy::new(3, 3, 5, 24);
        let n = node(1);
        let now = 1_000_000_000i64;
        for _ in 0..5 {
            policy.record_chronic_flag(n, now);
        }
        assert!(policy.is_chronic(&n, now));
        // The 5 flags at `now` remain, even after resetting request divergences
        assert_eq!(policy.chronic_window_hours(), 24);
        assert!(policy.is_chronic(&n, now));
    }

    #[test]
    fn is_chronic_uses_strict_greater_than_cutoff() {
        let mut policy = ReSyncPolicy::new(3, 3, 3, 24);
        let n = node(1);
        let now = 1_000_000_000i64;
        let cutoff = now - 24 * SECONDS_PER_HOUR;
        // 2 flags inside the window
        policy.record_chronic_flag(n, now);
        policy.record_chronic_flag(n, now);
        // 1 flag exactly at the cutoff boundary
        policy.record_chronic_flag(n, cutoff);
        // threshold is 3, but only 2 flags are > cutoff, so NOT chronic
        assert!(!policy.is_chronic(&n, now));
    }
}
