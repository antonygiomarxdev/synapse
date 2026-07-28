use crate::identity::NodeId;
use crate::shared::DomainError;
use crate::swarm::consensus::{ConsensusResult, NodeOutput, vote};
use crate::swarm::ports::{InferenceEngine, InferenceRequest, SwarmCoordinator};
use std::sync::Arc;

/// libp2p-based coordinator adapter.
///
/// V1 uses the adapter as a local trait bridge. The full network
/// transport will be added in a later phase; for now it simulates
/// multi-node coordination by invoking the provided `InferenceEngine`
/// multiple times with different swarm seeds.
#[derive(Clone)]
pub struct Libp2pSwarmCoordinator {
    engine: Arc<dyn InferenceEngine>,
    last_outputs: Vec<NodeOutput>,
}

impl std::fmt::Debug for Libp2pSwarmCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Libp2pSwarmCoordinator")
            .field("engine", &"<inference engine>")
            .field("last_outputs", &self.last_outputs)
            .finish()
    }
}

impl Libp2pSwarmCoordinator {
    /// Creates a new coordinator backed by the given inference engine.
    pub fn new(engine: Arc<dyn InferenceEngine>) -> Self {
        Self { engine, last_outputs: Vec::new() }
    }
}

impl SwarmCoordinator for Libp2pSwarmCoordinator {
    fn coordinate(&self, request: &InferenceRequest) -> Result<ConsensusResult, DomainError> {
        // Simulated multi-node coordination for V1.
        let swarm = request.swarm.clone().ok_or(DomainError::InvalidSwarmSize { size: 0 })?;
        let mut outputs = Vec::with_capacity(swarm.swarm_size() as usize);
        for i in 0..swarm.swarm_size() {
            let node_id = NodeId::from_public_key(&[i as u8; 32]);
            let output = self.engine.generate(request)?;
            outputs.push(NodeOutput { node_id, tokens: output.tokens });
        }
        let result = vote(request.id, &outputs, swarm.quorum())?;
        Ok(result)
    }

    fn node_outputs(&self) -> Vec<NodeOutput> {
        self.last_outputs.clone()
    }
}
