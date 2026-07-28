use thiserror::Error;

/// Cross-cutting domain errors for the Synapse protocol.
///
/// Every domain operation that can fail returns a [`DomainError`].
/// Infrastructure adapters translate these into their own error types.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DomainError {
    #[error("invalid NodeId: {reason}")]
    InvalidNodeId { reason: String },

    #[error("invalid ModelId: {reason}")]
    InvalidModelId { reason: String },

    #[error("invalid ExpertId: {reason}")]
    InvalidExpertId { reason: String },

    #[error("duplicate model: {model_id}")]
    DuplicateModel { model_id: String },

    #[error("model not found: {model_id}")]
    ModelNotFound { model_id: String },

    #[error("duplicate stake address: {address}")]
    DuplicateStakeAddress { address: String },

    #[error("node not found: {node_id}")]
    NodeNotFound { node_id: String },

    #[error("insufficient reputation: current {current}, required {required}")]
    InsufficientReputation { current: u16, required: u16 },

    #[error("invalid reputation score: {score} (must be 0-{max})")]
    InvalidReputation { score: u16, max: u16 },

    #[error("invalid stake amount: {reason}")]
    InvalidStakeAmount { reason: String },

    #[error("invalid swarm size: {size}")]
    InvalidSwarmSize { size: u32 },

    #[error("invalid price: {reason}")]
    InvalidPrice { reason: String },

    #[error("invalid token: {reason}")]
    InvalidToken { reason: String },

    #[error("storage error: {message}")]
    StorageError { message: String },

    #[error("signature verification failed")]
    SignatureVerificationFailed,

    #[error("catalog load failed: {reason}")]
    CatalogLoadFailed { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_node_id_display() {
        let err = DomainError::InvalidNodeId { reason: "too short".into() };
        assert_eq!(err.to_string(), "invalid NodeId: too short");
    }

    #[test]
    fn invalid_model_id_display() {
        let err = DomainError::InvalidModelId { reason: "empty string".into() };
        assert_eq!(err.to_string(), "invalid ModelId: empty string");
    }

    #[test]
    fn duplicate_model_display() {
        let err = DomainError::DuplicateModel { model_id: "kimi-k3".into() };
        assert_eq!(err.to_string(), "duplicate model: kimi-k3");
    }

    #[test]
    fn node_not_found_display() {
        let err = DomainError::NodeNotFound { node_id: "abc123".into() };
        assert_eq!(err.to_string(), "node not found: abc123");
    }

    #[test]
    fn insufficient_reputation_display() {
        let err = DomainError::InsufficientReputation { current: 150, required: 300 };
        assert_eq!(err.to_string(), "insufficient reputation: current 150, required 300");
    }

    #[test]
    fn invalid_reputation_display() {
        let err = DomainError::InvalidReputation { score: 1500, max: 1000 };
        assert_eq!(err.to_string(), "invalid reputation score: 1500 (must be 0-1000)");
    }

    #[test]
    fn signature_failed_display() {
        let err = DomainError::SignatureVerificationFailed;
        assert_eq!(err.to_string(), "signature verification failed");
    }

    #[test]
    fn error_equality() {
        let a = DomainError::InvalidNodeId { reason: "bad".into() };
        let b = DomainError::InvalidNodeId { reason: "bad".into() };
        assert_eq!(a, b);
    }

    #[test]
    fn error_inequality_different_variant() {
        let a = DomainError::InvalidNodeId { reason: "bad".into() };
        let b = DomainError::ModelNotFound { model_id: "x".into() };
        assert_ne!(a, b);
    }
}
