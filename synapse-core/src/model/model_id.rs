use serde::{Deserialize, Serialize};

/// Unique identifier for an AI model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(String);

impl ModelId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_equality() {
        let a = ModelId::new("kimi-k3");
        let b = ModelId::new("kimi-k3");
        assert_eq!(a, b);
    }

    #[test]
    fn model_id_inequality() {
        let a = ModelId::new("kimi-k3");
        let b = ModelId::new("mixtral-8x7b");
        assert_ne!(a, b);
    }

    #[test]
    fn model_id_as_str() {
        let id = ModelId::new("kimi-k3");
        assert_eq!(id.as_str(), "kimi-k3");
    }
}
