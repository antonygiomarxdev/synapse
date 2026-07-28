use std::sync::Arc;
use synapse_core::model::ModelId;
use synapse_core::shared::DomainError;
use synapse_core::swarm::infrastructure::Libp2pSwarmCoordinator;
use synapse_core::swarm::ports::{
    InferenceEngine, InferenceOutput, InferenceRequest, Priority, SwarmCoordinator,
};
use synapse_core::swarm::{SpecSwarmConfig, Token};
use uuid::Uuid;

/// A deterministic inference engine that returns the same tokens on every call.
///
/// Used to verify that the coordinator reaches consensus when all nodes
/// produce identical output. Divergence simulation will be added in a
/// later phase when the coordinator accepts per-node engines.
struct DeterministicEngine {
    tokens: Vec<Token>,
}

impl InferenceEngine for DeterministicEngine {
    fn generate(&self, _request: &InferenceRequest) -> Result<InferenceOutput, DomainError> {
        Ok(InferenceOutput { request_id: Uuid::new_v4(), tokens: self.tokens.clone() })
    }
}

#[test]
fn coordinator_reaches_consensus_with_unanimous_engine() {
    let model = ModelId::new("kimi-k3").unwrap();
    let tokens = vec![Token::new("def", -0.5).unwrap(), Token::new(" fibo", -0.2).unwrap()];
    let engine = Arc::new(DeterministicEngine { tokens });
    let mut coordinator = Libp2pSwarmCoordinator::new(engine);
    let request = InferenceRequest {
        id: Uuid::new_v4(),
        model: model.clone(),
        priority: Priority::Realtime,
        swarm: Some(SpecSwarmConfig::new(model, 5).unwrap()),
        max_tokens: 10,
    };
    let result = coordinator.coordinate(&request).unwrap();
    assert_eq!(result.consensus_tokens.len(), 2);
    assert_eq!(result.consensus_tokens[0].text(), "def");
    assert_eq!(result.consensus_tokens[1].text(), " fibo");
    assert!(result.divergent_nodes.is_empty());
}
