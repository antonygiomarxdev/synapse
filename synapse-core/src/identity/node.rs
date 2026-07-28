use super::{KeyPair, NodeId};
use crate::shared::{DomainError, DomainEvent};

/// The maximum reputation score a node can achieve.
pub const MAX_REPUTATION: u16 = 1000;

/// The initial reputation score assigned to newly registered nodes.
pub const INITIAL_REPUTATION: u16 = 100;

/// A Synapse compute node registered in the swarm.
///
/// A [`Node`] is the aggregate root for node identity. It binds together
/// a cryptographic identity ([`NodeId`]), an on-chain stake address,
/// and a reputation score that governs routing priority and slashing risk.
///
/// Nodes are created via [`Node::register`], which emits a
/// [`DomainEvent::NodeRegistered`] event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub node_id: NodeId,
    pub stake_address: String,
    pub reputation: u16,
}

impl Node {
    /// Registers a new node with the given key pair and stake address.
    ///
    /// The node's [`NodeId`] is derived from the public key. Reputation
    /// starts at [`INITIAL_REPUTATION`] (100).
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidNodeId`] if the stake address is empty.
    pub fn register(
        keypair: &KeyPair,
        stake_address: String,
    ) -> Result<(Self, Vec<DomainEvent>), DomainError> {
        if stake_address.trim().is_empty() {
            return Err(DomainError::InvalidNodeId {
                reason: "stake address must not be empty".into(),
            });
        }

        let node_id = Self::derive_node_id(keypair);

        let node =
            Self { node_id, stake_address: stake_address.clone(), reputation: INITIAL_REPUTATION };

        let event = DomainEvent::NodeRegistered {
            event_id: uuid::Uuid::new_v4(),
            node_id: node_id.to_string(),
            stake_address,
            reputation: INITIAL_REPUTATION,
        };

        Ok((node, vec![event]))
    }

    /// Derives a [`NodeId`] from a [`KeyPair`]'s public key.
    ///
    /// This is `SHA256(public_key_bytes)` — deterministic and pure.
    pub fn derive_node_id(keypair: &KeyPair) -> NodeId {
        NodeId::from_public_key(keypair.public_key_bytes())
    }

    /// Updates the node's reputation score.
    ///
    /// The score is clamped to `[0, MAX_REPUTATION]`.
    /// Returns an event if the score changed.
    pub fn update_reputation(&mut self, new_score: u16) -> Option<DomainEvent> {
        let clamped = new_score.min(MAX_REPUTATION);
        if clamped == self.reputation {
            return None;
        }
        let old = self.reputation;
        self.reputation = clamped;
        Some(DomainEvent::ReputationChanged {
            event_id: uuid::Uuid::new_v4(),
            node_id: self.node_id.to_string(),
            old_score: old,
            new_score: clamped,
        })
    }

    /// Returns true if this node meets the minimum reputation threshold.
    pub fn meets_reputation(&self, minimum: u16) -> bool {
        self.reputation >= minimum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keypair() -> KeyPair {
        KeyPair { public: [1u8; 32], secret: [2u8; 32] }
    }

    // --- register tests ---

    #[test]
    fn register_produces_valid_node() {
        let kp = test_keypair();
        let (node, events) = Node::register(&kp, "0xabcd".into()).unwrap();
        assert_eq!(node.stake_address, "0xabcd");
        assert_eq!(node.reputation, INITIAL_REPUTATION);
        assert!(!events.is_empty());
        assert!(matches!(events[0], DomainEvent::NodeRegistered { .. }));
    }

    #[test]
    fn register_rejects_empty_stake_address() {
        let kp = test_keypair();
        let result = Node::register(&kp, "".into());
        assert!(result.is_err());
    }

    #[test]
    fn register_rejects_whitespace_only_stake_address() {
        let kp = test_keypair();
        let result = Node::register(&kp, "   ".into());
        assert!(result.is_err());
    }

    #[test]
    fn register_starts_at_100_reputation() {
        let kp = test_keypair();
        let (node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        assert_eq!(node.reputation, 100);
    }

    #[test]
    fn register_emits_node_registered_event() {
        let kp = test_keypair();
        let (node, events) = Node::register(&kp, "0xabc".into()).unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            DomainEvent::NodeRegistered { node_id, stake_address, reputation, .. } => {
                assert_eq!(node_id, &node.node_id.to_string());
                assert_eq!(stake_address, "0xabc");
                assert_eq!(*reputation, 100);
            }
            _ => panic!("wrong event variant"),
        }
    }

    // --- derive_node_id tests ---

    #[test]
    fn derive_node_id_is_deterministic() {
        let kp = test_keypair();
        let id1 = Node::derive_node_id(&kp);
        let id2 = Node::derive_node_id(&kp);
        assert_eq!(id1, id2);
    }

    #[test]
    fn derive_node_id_differs_for_different_keys() {
        let kp1 = test_keypair();
        let kp2 = KeyPair { public: [3u8; 32], secret: [4u8; 32] };
        assert_ne!(Node::derive_node_id(&kp1), Node::derive_node_id(&kp2));
    }

    #[test]
    fn derive_node_id_matches_register() {
        let kp = test_keypair();
        let expected = Node::derive_node_id(&kp);
        let (node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        assert_eq!(node.node_id, expected);
    }

    // --- update_reputation tests ---

    #[test]
    fn update_reputation_changes_score() {
        let kp = test_keypair();
        let (mut node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        let event = node.update_reputation(500).unwrap();
        assert_eq!(node.reputation, 500);
        match event {
            DomainEvent::ReputationChanged { old_score, new_score, .. } => {
                assert_eq!(old_score, 100);
                assert_eq!(new_score, 500);
            }
            _ => panic!("wrong event"),
        }
    }

    #[test]
    fn update_reputation_clamps_to_max() {
        let kp = test_keypair();
        let (mut node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        node.update_reputation(1500);
        assert_eq!(node.reputation, MAX_REPUTATION);
    }

    #[test]
    fn update_reputation_clamps_to_zero() {
        let kp = test_keypair();
        let (mut node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        node.update_reputation(0);
        assert_eq!(node.reputation, 0);
    }

    #[test]
    fn update_reputation_no_event_if_unchanged() {
        let kp = test_keypair();
        let (mut node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        // Rep starts at 100; setting to 100 should be no-op
        let clamped_100 = node.reputation; // 100, already at 100
        let event = node.update_reputation(clamped_100);
        assert!(event.is_none());
    }

    // --- meets_reputation tests ---

    #[test]
    fn meets_reputation_above_threshold() {
        let kp = test_keypair();
        let (node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        assert!(node.meets_reputation(50));
    }

    #[test]
    fn meets_reputation_at_threshold() {
        let kp = test_keypair();
        let (node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        assert!(node.meets_reputation(100));
    }

    #[test]
    fn meets_reputation_below_threshold() {
        let kp = test_keypair();
        let (mut node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        node.update_reputation(50);
        assert!(!node.meets_reputation(300));
    }
}
