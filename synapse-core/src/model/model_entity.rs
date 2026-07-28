use super::model_id::ModelId;
use serde::{Deserialize, Serialize};

/// A curated model in the Synapse catalog.
///
/// Each model entry captures the structural metadata needed for
/// swarm expert routing: expert count, active per token, VRAM
/// footprint, context window, and license compliance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntity {
    pub id: ModelId,
    pub name: String,
    pub description: String,
    pub total_params: String,
    pub experts: u32,
    pub active_per_token: u32,
    pub expert_size_gb: f64,
    pub shared_params_gb: f64,
    pub context_window: u64,
    pub license: String,
    pub hf_repo: String,
    pub sha256: Option<String>,
}

impl ModelEntity {
    /// Creates a new model entity. All fields required except sha256.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ModelId,
        name: String,
        description: String,
        total_params: String,
        experts: u32,
        active_per_token: u32,
        expert_size_gb: f64,
        shared_params_gb: f64,
        context_window: u64,
        license: String,
        hf_repo: String,
        sha256: Option<String>,
    ) -> Self {
        Self {
            id,
            name,
            description,
            total_params,
            experts,
            active_per_token,
            expert_size_gb,
            shared_params_gb,
            context_window,
            license,
            hf_repo,
            sha256,
        }
    }

    /// The model's sparsity ratio: active_per_token / experts.
    pub fn sparsity(&self) -> f64 {
        (self.active_per_token as f64) / (self.experts as f64)
    }

    /// The minimum number of nodes needed to cover all experts
    /// at `experts_per_node` experts per node.
    pub fn min_nodes_for_coverage(&self, experts_per_node: u32) -> u32 {
        self.experts.div_ceil(experts_per_node)
    }

    /// Total VRAM required per node: expert_size * experts_per_node + shared_params.
    pub fn vram_per_node(&self, experts_per_node: u32) -> f64 {
        (self.expert_size_gb * experts_per_node as f64) + self.shared_params_gb
    }
}

impl PartialEq for ModelEntity {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kimi() -> ModelId {
        ModelId::new("kimi-k3").unwrap()
    }

    fn mixtral() -> ModelId {
        ModelId::new("mixtral-8x7b").unwrap()
    }

    fn make_kimi() -> ModelEntity {
        ModelEntity::new(
            kimi(),
            "Kimi K3".into(),
            "Moonshot AI frontier MoE".into(),
            "2.8T".into(),
            896,
            16,
            1.5,
            12.0,
            1_000_000,
            "MIT".into(),
            "moonshotai/Kimi-K3".into(),
            None,
        )
    }

    fn make_mixtral() -> ModelEntity {
        ModelEntity::new(
            mixtral(),
            "Mixtral 8x7B".into(),
            "Mistral MoE".into(),
            "46.7B".into(),
            8,
            2,
            3.0,
            3.0,
            32768,
            "Apache-2.0".into(),
            "mistralai/Mixtral-8x7B-v0.1".into(),
            None,
        )
    }

    #[test]
    fn model_entity_creation() {
        let model = make_kimi();
        assert_eq!(model.experts, 896);
        assert_eq!(model.active_per_token, 16);
        assert_eq!(model.context_window, 1_000_000);
    }

    #[test]
    fn kimi_k3_sparsity_is_below_2_percent() {
        let model = make_kimi();
        assert!(model.sparsity() < 0.02);
        assert!((model.sparsity() - 0.01785).abs() < 0.001);
    }

    #[test]
    fn mixtral_sparsity() {
        let model = make_mixtral();
        assert_eq!(model.sparsity(), 0.25);
    }

    #[test]
    fn min_nodes_for_coverage_ceil() {
        let model = make_kimi();
        // 896 experts, 2 per node → 448 nodes
        assert_eq!(model.min_nodes_for_coverage(2), 448);
        // 896 / 3 = 298.66 → 299
        assert_eq!(model.min_nodes_for_coverage(3), 299);
    }

    #[test]
    fn vram_per_node_calculation() {
        let model = make_kimi();
        // 2 experts: 1.5 * 2 + 12.0 = 15.0 GB
        assert!((model.vram_per_node(2) - 15.0).abs() < 0.01);
        // 4 experts: 1.5 * 4 + 12.0 = 18.0 GB
        assert!((model.vram_per_node(4) - 18.0).abs() < 0.01);
    }

    #[test]
    fn equality_is_by_id_only() {
        let a = make_kimi();
        let mut b = make_kimi();
        b.description = "different".into();
        assert_eq!(a, b);
    }

    #[test]
    fn different_models_not_equal() {
        assert_ne!(make_kimi(), make_mixtral());
    }
}
