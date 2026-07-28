use super::model_id::ModelId;

/// A model entity in the catalog.
#[derive(Debug, Clone)]
pub struct ModelEntity {
    pub id: ModelId,
    pub experts: u32,
    pub active_per_token: u32,
}

impl ModelEntity {
    pub fn new(id: ModelId, experts: u32, active_per_token: u32) -> Self {
        Self { id, experts, active_per_token }
    }

    pub fn sparsity(&self) -> f64 {
        (self.active_per_token as f64) / (self.experts as f64)
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

    #[test]
    fn model_entity_creation() {
        let model = ModelEntity::new(kimi(), 896, 16);
        assert_eq!(model.experts, 896);
        assert_eq!(model.active_per_token, 16);
    }

    #[test]
    fn kimi_k3_sparsity_is_below_2_percent() {
        let model = ModelEntity::new(kimi(), 896, 16);
        assert!(model.sparsity() < 0.02);
        assert!((model.sparsity() - 0.01785).abs() < 0.001);
    }

    #[test]
    fn mixtral_sparsity() {
        let model = ModelEntity::new(mixtral(), 8, 2);
        assert_eq!(model.sparsity(), 0.25);
    }
}
