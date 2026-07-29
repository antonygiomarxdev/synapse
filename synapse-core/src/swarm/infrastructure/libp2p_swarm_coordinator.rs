use crate::identity::NodeId;
use crate::shared::DomainError;
use crate::swarm::consensus::{ConsensusResult, NodeOutput, vote};
use crate::swarm::ports::{InferenceEngine, InferenceRequest, SwarmCoordinator};
use std::sync::Arc;

/// Local trait bridge for V1 swarm coordination.
///
/// This is NOT a real libp2p transport. It simulates multi-node coordination
/// by invoking the same `InferenceEngine` N times in-process. Full P2P
/// transport will be added in Phase 6 (E2E Integration).
///
/// The name preserves API compatibility for the eventual libp2p adapter.
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
    fn coordinate(&mut self, request: &InferenceRequest) -> Result<ConsensusResult, DomainError> {
        // Simulated multi-node coordination for V1.
        let swarm = request.swarm.clone().ok_or(DomainError::InvalidSwarmSize { size: 0 })?;
        let mut outputs = Vec::with_capacity(swarm.swarm_size() as usize);
        for (i, _seed) in swarm.seeds().iter().enumerate() {
            let node_id = NodeId::from_public_key(&[i as u8; 32]);
            // Clone the request per node so future V2 can vary per-node config
            // (e.g. bind the seed to the swarm field). V1 simulation ignores the
            // seed value, but the iteration fulfills the doc contract.
            let node_request = request.clone();
            let output = self.engine.generate(&node_request)?;
            outputs.push(NodeOutput { node_id, tokens: output.tokens });
        }
        self.last_outputs = outputs;
        let result = vote(request.id, &self.last_outputs, swarm.quorum())?;
        Ok(result)
    }

    fn node_outputs(&self) -> Vec<NodeOutput> {
        self.last_outputs.clone()
    }
}
