use super::model_id::ModelId;
use crate::shared::DomainError;
use serde::{Deserialize, Serialize};

/// Identifies a specific expert within a model.
///
/// An expert is a sub-network within a Mixture-of-Experts model.
/// `ExpertId` is a composite key of the model identifier and the
/// zero-based expert index within that model.
///
/// Example: Expert #3 of Kimi K3 → `ExpertId { model: "kimi-k3", index: 3 }`
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExpertId {
    pub model: ModelId,
    pub index: u32,
}

impl ExpertId {
    /// Creates a new `ExpertId`.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidExpertId`] if `index` exceeds
    /// `max_expert_index` (must be `< num_experts`).
    pub fn new(model: ModelId, index: u32, num_experts: u32) -> Result<Self, DomainError> {
        if index >= num_experts {
            return Err(DomainError::InvalidExpertId {
                reason: format!(
                    "expert index {index} is out of bounds for model with {num_experts} experts"
                ),
            });
        }
        Ok(Self { model, index })
    }

    /// Creates an `ExpertId` without bounds checking.
    ///
    /// Useful when the model's expert count is validated elsewhere.
    pub fn new_unchecked(model: ModelId, index: u32) -> Self {
        Self { model, index }
    }
}

impl std::fmt::Display for ExpertId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}#{}", self.model, self.index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kimi_model() -> ModelId {
        ModelId::new("kimi-k3").unwrap()
    }

    #[test]
    fn valid_expert_id() {
        let id = ExpertId::new(kimi_model(), 0, 896).unwrap();
        assert_eq!(id.index, 0);
        assert_eq!(id.model.as_str(), "kimi-k3");
    }

    #[test]
    fn last_expert_is_valid() {
        let id = ExpertId::new(kimi_model(), 895, 896).unwrap();
        assert_eq!(id.index, 895);
    }

    #[test]
    fn expert_at_model_boundary_rejected() {
        assert!(ExpertId::new(kimi_model(), 896, 896).is_err());
    }

    #[test]
    fn expert_beyond_boundary_rejected() {
        let err = ExpertId::new(kimi_model(), 1000, 896).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn new_unchecked_skips_validation() {
        let id = ExpertId::new_unchecked(kimi_model(), 9999);
        assert_eq!(id.index, 9999);
    }

    #[test]
    fn display_format() {
        let id = ExpertId::new(kimi_model(), 42, 896).unwrap();
        assert_eq!(id.to_string(), "kimi-k3#42");
    }

    #[test]
    fn equality_same_model_same_index() {
        let a = ExpertId::new(kimi_model(), 7, 100).unwrap();
        let b = ExpertId::new(kimi_model(), 7, 100).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn inequality_different_index() {
        let a = ExpertId::new(kimi_model(), 7, 100).unwrap();
        let b = ExpertId::new(kimi_model(), 8, 100).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn inequality_different_model() {
        let kimi = kimi_model();
        let mixtral = ModelId::new("mixtral-8x7b").unwrap();
        let a = ExpertId::new(kimi, 0, 100).unwrap();
        let b = ExpertId::new(mixtral, 0, 10).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let id = ExpertId::new(kimi_model(), 42, 896).unwrap();
        let json = serde_json::to_string(&id).unwrap();
        let parsed: ExpertId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, parsed);
    }
}
