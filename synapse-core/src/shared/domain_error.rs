use thiserror::Error;

/// Cross-cutting domain errors for the Synapse protocol.
///
/// Every domain operation that can fail returns a [`DomainError`].
/// Infrastructure adapters translate these into their own error types.
#[derive(Debug, Clone, PartialEq, Error)]
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

    #[error("invalid token log_prob: {value} (must be finite)")]
    InvalidTokenLogProb { value: f64 },

    #[error("invalid token text: {reason}")]
    InvalidTokenText { reason: String },

    #[error("invalid consensus quorum: {quorum} for swarm_size {swarm_size}")]
    InvalidConsensusQuorum { quorum: usize, swarm_size: usize },

    #[error("no consensus reached at token index {token_index}")]
    NoConsensus { token_index: usize },

    #[error("storage error: {message}")]
    StorageError { message: String },

    #[error("signature verification failed")]
    SignatureVerificationFailed,

    #[error("invalid route: {reason}")]
    InvalidRoute { reason: String },

    #[error("catalog load failed: {reason}")]
    CatalogLoadFailed { reason: String },
}

// SAFETY: DomainError contains f64 only in InvalidTokenLogProb, and
// Token::new() rejects non-finite values, so Eq is sound despite the
// derived PartialEq on f64 (NaN comparisons can never occur).
impl Eq for DomainError {}

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

    #[test]
    fn invalid_token_log_prob_display() {
        let err = DomainError::InvalidTokenLogProb { value: f64::NAN };
        assert_eq!(err.to_string(), "invalid token log_prob: NaN (must be finite)");
    }

    #[test]
    fn invalid_token_text_display() {
        let err = DomainError::InvalidTokenText { reason: "too long".into() };
        assert_eq!(err.to_string(), "invalid token text: too long");
    }
    #[test]
    fn invalid_route_display() {
        let err = DomainError::InvalidRoute { reason: "empty steps".into() };
        assert_eq!(err.to_string(), "invalid route: empty steps");
    }

    #[test]
    fn invalid_consensus_quorum_display() {
        let err = DomainError::InvalidConsensusQuorum { quorum: 0, swarm_size: 5 };
        assert_eq!(err.to_string(), "invalid consensus quorum: 0 for swarm_size 5");
    }

    #[test]
    fn no_consensus_display() {
        let err = DomainError::NoConsensus { token_index: 7 };
        assert_eq!(err.to_string(), "no consensus reached at token index 7");
    }
}
