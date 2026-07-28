use crate::model::ModelId;
use crate::shared::DomainError;
use crate::swarm::consensus::{ConsensusResult, NodeOutput};
use crate::swarm::speculative::SpecSwarmConfig;
use crate::swarm::token::Token;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Request priority selects the swarm execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Priority {
    Realtime,
    Batch,
}

/// A request sent to an inference engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub id: Uuid,
    pub model: ModelId,
    pub priority: Priority,
    pub swarm: Option<SpecSwarmConfig>,
    pub max_tokens: u32,
}

/// Output from a single inference engine invocation.
///
/// Does not derive `Eq` because [`Token`] only implements `PartialEq`
/// (it contains an `f64` log-probability field).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceOutput {
    pub request_id: Uuid,
    pub tokens: Vec<Token>,
}

/// Port implemented by concrete inference runtimes (vLLM, llama.cpp, ...).
///
/// The domain knows this trait only; infrastructure adapters provide
/// the actual model execution. No async, no I/O in this trait.
pub trait InferenceEngine {
    /// Generates tokens for a single request.
    fn generate(&self, request: &InferenceRequest) -> Result<InferenceOutput, DomainError>;
}

/// Port implemented by swarm coordinators.
///
/// A coordinator takes a request, dispatches it to multiple nodes
/// through an `InferenceEngine`, and applies consensus to produce a
/// trusted result.
pub trait SwarmCoordinator {
    /// Coordinates a request across the swarm and returns the consensus.
    fn coordinate(&self, request: &InferenceRequest) -> Result<ConsensusResult, DomainError>;

    /// Returns the raw node outputs for the last coordinated request.
    ///
    /// Useful for audit, debugging, and re-sync.
    fn node_outputs(&self) -> Vec<NodeOutput>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelId;
    use crate::swarm::{SpecSwarmConfig, Token};

    struct DummyEngine;

    impl InferenceEngine for DummyEngine {
        fn generate(
            &self,
            request: &InferenceRequest,
        ) -> Result<InferenceOutput, crate::shared::DomainError> {
            Ok(InferenceOutput {
                request_id: request.id,
                tokens: vec![Token::new("ok", -0.1).unwrap()],
            })
        }
    }

    #[test]
    fn dummy_engine_implements_trait() {
        let req = InferenceRequest {
            id: uuid::Uuid::new_v4(),
            model: ModelId::new("kimi-k3").unwrap(),
            priority: Priority::Realtime,
            swarm: Some(SpecSwarmConfig::new(ModelId::new("kimi-k3").unwrap(), 5).unwrap()),
            max_tokens: 10,
        };
        let engine = DummyEngine;
        let out = engine.generate(&req).unwrap();
        assert_eq!(out.tokens.len(), 1);
        assert_eq!(out.tokens[0].text(), "ok");
    }
}
