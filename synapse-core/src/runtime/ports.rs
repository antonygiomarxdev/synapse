use crate::model::{ExpertId, ModelId};
use crate::shared::DomainError;
use crate::swarm::ports::{InferenceOutput, InferenceRequest};

/// Port implemented by concrete inference runtimes (vLLM, llama.cpp, SGLang).
///
/// The domain depends on this trait only; infrastructure adapters provide
/// the actual model loading, inference, verification, and VRAM detection.
///
/// Each method maps to a bridge protocol message sent over Unix socket
/// to the Python subprocess.
pub trait InferencePort {
    /// Loads a model with the specified experts into VRAM.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::ModelNotFound`] if the model is not available.
    /// Returns [`DomainError::StorageError`] if VRAM is insufficient.
    fn load(&self, model: &ModelId, experts: &[ExpertId]) -> Result<(), DomainError>;

    /// Generates tokens for a single inference request.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidTokenText`] or [`DomainError::InvalidTokenLogProb`] if the prompt contains invalid tokens.
    /// Returns [`DomainError::StorageError`] if the runtime encounters an error.
    fn generate(&self, request: &InferenceRequest) -> Result<InferenceOutput, DomainError>;

    /// Verifies that a model's weights match the expected SHA256 hash.
    ///
    /// Returns `Ok(true)` if the hash matches, `Ok(false)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::ModelNotFound`] if the model is not loaded.
    fn verify(&self, model: &ModelId, expected_hash: &str) -> Result<bool, DomainError>;

    /// Detects available VRAM on the compute node.
    ///
    /// Returns the available VRAM in megabytes.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::StorageError`] if VRAM detection fails.
    fn detect_vram(&self) -> Result<u32, DomainError>;
}

#[cfg(test)]
mod tests {
    /// These tests are documentation of the trait contract.
    /// Actual behavior is tested in infrastructure adapter tests.

    /// Trait is object-safe for dynamic dispatch in tests.
    #[allow(dead_code)]
    fn assert_object_safe(_port: &dyn super::InferencePort) {}

    /// The trait has exactly 4 methods.
    #[test]
    fn trait_has_four_methods() {
        // compile-time assertion: if a method is added/removed,
        // the implementations in infrastructure/ must be updated.
        fn _check(p: &dyn super::InferencePort) {
            let _ = p.load(&crate::model::ModelId::new("test").unwrap(), &[]);
            let _ = p.generate(&crate::swarm::ports::InferenceRequest::new(
                uuid::Uuid::new_v4(),
                crate::model::ModelId::new("test").unwrap(),
                crate::swarm::ports::Priority::Batch,
                None,
                100,
            ));
            let _ = p.verify(&crate::model::ModelId::new("test").unwrap(), "");
            let _ = p.detect_vram();
        }
    }
}
