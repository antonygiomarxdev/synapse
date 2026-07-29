/// MoE Coordinator domain types.
///
/// The coordinator runs shared layers (attention, norms) and the router,
/// then dispatches expert computation to workers. It only needs `gate_inp`
/// weights (~384 KB per shard) and `SpecSwarmConfig` from the existing
/// swarm module.
use crate::identity::NodeId;
use crate::shared::DomainError;
use crate::swarm::ports::InferenceRequest;
use serde::{Deserialize, Serialize};

/// Routing decision for a single token: which experts to activate
/// and through which worker nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertRoute {
    /// Expert IDs assigned to each worker node.
    pub assignments: Vec<WorkerAssignment>,
    /// Gate weights for combining outputs.
    pub gate_weights: Vec<f32>,
}

/// A set of expert indices assigned to one worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerAssignment {
    pub node_id: NodeId,
    /// Local expert indices within that worker's shard.
    pub expert_ids: Vec<u32>,
}

/// One layer of gate_inp weights: [n_experts × d_model] in expert-major order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateInpLayer {
    pub weights: Vec<f32>, // flat: n_experts * d_model
    pub n_experts: usize,
    pub d_model: usize,
}

impl GateInpLayer {
    /// Load from a buffer of f32 values in expert-major order.
    pub fn from_slice(data: &[f32], n_experts: usize, d_model: usize) -> Result<Self, DomainError> {
        let expected = n_experts * d_model;
        if data.len() != expected {
            return Err(DomainError::InvalidExpertCount {
                expected: expected as u64,
                actual: data.len() as u64,
            });
        }
        Ok(Self {
            weights: data.to_vec(),
            n_experts,
            d_model,
        })
    }

    /// Compute expert scores: hidden_state @ gate_inp.T
    /// hidden: [d_model], returns [n_experts] scores.
    pub fn score_experts(&self, hidden: &[f32]) -> Vec<f32> {
        let n = self.n_experts;
        let d = self.d_model;
        let mut scores = vec![0.0_f32; n];
        for expert in 0..n {
            let mut acc = 0.0;
            for dim in 0..d {
                acc += hidden[dim] * self.weights[expert * d + dim];
            }
            scores[expert] = acc;
        }
        scores
    }

    /// Top-k expert indices by score.
    pub fn top_k(&self, hidden: &[f32], k: usize) -> Vec<u32> {
        let scores = self.score_experts(hidden);
        let mut indexed: Vec<(usize, f32)> =
            scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .into_iter()
            .take(k.min(self.n_experts))
            .map(|(i, _)| i as u32)
            .collect()
    }
}

/// Trait for MoE routing: decides which worker handles which experts.
pub trait ExpertRouter {
    fn route(
        &self,
        layer: usize,
        hidden_state: &[f32],
        request: &InferenceRequest,
    ) -> Result<ExpertRoute, DomainError>;
}

/// Simple round-robin router that partitions experts evenly across workers.
pub struct RoundRobinRouter {
    pub layers: Vec<GateInpLayer>,
    pub worker_count: usize,
}

impl ExpertRouter for RoundRobinRouter {
    fn route(
        &self,
        layer: usize,
        hidden_state: &[f32],
        _request: &InferenceRequest,
    ) -> Result<ExpertRoute, DomainError> {
        let gate = self.layers.get(layer).ok_or(DomainError::InvalidExpertCount {
            expected: self.layers.len() as u64,
            actual: layer as u64,
        })?;

        let k = 8; // top-k experts per token
        let expert_ids = gate.top_k(hidden_state, k);
        let scores = gate.score_experts(hidden_state);

        let per_worker = (expert_ids.len() + self.worker_count - 1) / self.worker_count;
        let mut assignments = Vec::new();
        let mut gate_weights = Vec::new();

        for w in 0..self.worker_count {
            let start = w * per_worker;
            let end = (start + per_worker).min(expert_ids.len());
            if start >= end {
                break;
            }
            let local_ids: Vec<u32> = expert_ids[start..end].to_vec();
            for &eid in &local_ids {
                gate_weights.push(scores[eid as usize]);
            }
            assignments.push(WorkerAssignment {
                node_id: NodeId::from_hex(
                    &format!("000000000000000000000000000000000000000000000000000000000000000{w}")
                )
                .unwrap(),
                expert_ids: local_ids,
            });
        }

        Ok(ExpertRoute {
            assignments,
            gate_weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_inp_loads_and_scores() {
        // 2 experts, 4 dims, expert-major: [e0_dim0, e0_dim1, ..., e1_dim0, ...]
        let data = vec![
            1.0, 0.0, 0.0, 0.0, // expert 0
            0.0, 2.0, 0.0, 0.0, // expert 1
        ];
        let gate = GateInpLayer::from_slice(&data, 2, 4).unwrap();
        let hidden = vec![1.0, 0.0, 0.0, 0.0];
        let scores = gate.score_experts(&hidden);
        assert!((scores[0] - 1.0).abs() < 1e-6);
        assert!((scores[1] - 0.0).abs() < 1e-6);

        let top = gate.top_k(&hidden, 1);
        assert_eq!(top, vec![0]);
    }

    #[test]
    fn gate_inp_rejects_wrong_size() {
        let result = GateInpLayer::from_slice(&[1.0, 2.0, 3.0], 2, 4);
        assert!(result.is_err());
    }

    #[test]
    fn round_robin_routes_experts() {
        let data: Vec<f32> = (0..(4 * 8)).map(|x| x as f32 * 0.1).collect();
        let gate = GateInpLayer::from_slice(&data, 4, 8).unwrap();
        let router = RoundRobinRouter {
            layers: vec![gate],
            worker_count: 2,
        };

        let hidden: Vec<f32> = (0..8).map(|_| 0.5).collect();
        use crate::model::ModelId;
        use crate::swarm::ports::Priority;
        let req = InferenceRequest::new(
            uuid::Uuid::new_v4(),
            ModelId::new("test").unwrap(),
            Priority::Batch,
            None,
            10,
            vec![],
        );
        let route = router.route(0, &hidden, &req).unwrap();
        assert_eq!(route.assignments.len(), 2);
        assert!(!route.gate_weights.is_empty());
    }
}
