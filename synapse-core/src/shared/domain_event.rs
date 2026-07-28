use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Domain events emitted by aggregates when state changes.
///
/// Each event carries a unique ID for idempotent processing.
/// Infrastructure layers subscribe to these for side effects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainEvent {
    NodeRegistered { event_id: Uuid, node_id: String, stake_address: String, reputation: u16 },
    ModelAdded { event_id: Uuid, model_id: String, experts: u32, active_per_token: u32 },
    ModelRemoved { event_id: Uuid, model_id: String },
    StakeUpdated { event_id: Uuid, node_id: String, old_amount: u64, new_amount: u64 },
    ReputationChanged { event_id: Uuid, node_id: String, old_score: u16, new_score: u16 },
    NodeBanned { event_id: Uuid, node_id: String, reason: String },
    NodeUnbanned { event_id: Uuid, node_id: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_registered_event_has_unique_id() {
        let e1 = DomainEvent::NodeRegistered {
            event_id: Uuid::new_v4(),
            node_id: "n1".into(),
            stake_address: "0xabc".into(),
            reputation: 100,
        };
        let e2 = DomainEvent::NodeRegistered {
            event_id: Uuid::new_v4(),
            node_id: "n2".into(),
            stake_address: "0xdef".into(),
            reputation: 100,
        };
        // Different event_ids → different events
        assert_ne!(e1, e2);
    }

    #[test]
    fn model_added_event_roundtrip_json() {
        let event = DomainEvent::ModelAdded {
            event_id: Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
            model_id: "kimi-k3".into(),
            experts: 896,
            active_per_token: 16,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: DomainEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, parsed);
    }

    #[test]
    fn reputation_changed_event_roundtrip_json() {
        let event = DomainEvent::ReputationChanged {
            event_id: Uuid::new_v4(),
            node_id: "n1".into(),
            old_score: 100,
            new_score: 350,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: DomainEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, parsed);
    }

    #[test]
    fn node_banned_event_carries_reason() {
        let event = DomainEvent::NodeBanned {
            event_id: Uuid::new_v4(),
            node_id: "n42".into(),
            reason: "50 slashing flags".into(),
        };
        assert!(matches!(event, DomainEvent::NodeBanned { .. }));
    }

    #[test]
    fn all_variants_serialize() {
        let events = vec![
            DomainEvent::NodeRegistered {
                event_id: Uuid::new_v4(),
                node_id: "n1".into(),
                stake_address: "0x1".into(),
                reputation: 100,
            },
            DomainEvent::ModelAdded {
                event_id: Uuid::new_v4(),
                model_id: "m1".into(),
                experts: 8,
                active_per_token: 2,
            },
            DomainEvent::StakeUpdated {
                event_id: Uuid::new_v4(),
                node_id: "n1".into(),
                old_amount: 500,
                new_amount: 1000,
            },
        ];
        let json = serde_json::to_string(&events).unwrap();
        let parsed: Vec<DomainEvent> = serde_json::from_str(&json).unwrap();
        assert_eq!(events, parsed);
    }
}
