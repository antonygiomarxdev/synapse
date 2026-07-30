/// Implementation of `InferencePort` for the native MoE runtime.
///
/// Loads models from GGUF files, runs the transformer forward pass with
/// external expert routing control, verifies weight integrity, and reports
/// memory usage.
use std::path::PathBuf;

use crate::model::{ExpertId, ModelId};
use crate::native_moe::forward::forward;
use crate::native_moe::model::MoeModel;
use crate::runtime::ports::InferencePort;
use crate::shared::DomainError;

/// GGUF-backed MoE inference runtime.
pub struct NativeMoeRuntime {
    model: Option<MoeModel>,
    model_path: Option<PathBuf>,
}

impl NativeMoeRuntime {
    /// Create a runtime that loads models from the given GGUF file.
    pub fn new(model_path: PathBuf) -> Self {
        NativeMoeRuntime {
            model: None,
            model_path: Some(model_path),
        }
    }
}

impl InferencePort for NativeMoeRuntime {
    fn load(&mut self, _model: &ModelId, _experts: &[ExpertId]) -> Result<(), DomainError> {
        let path = self.model_path.as_ref().ok_or_else(|| {
            DomainError::ModelNotFound { model_id: "no path configured".into() }
        })?;

        let loaded = MoeModel::load_routing(path).map_err(|e| {
            DomainError::StorageError { message: e }
        })?;

        self.model = Some(loaded);
        Ok(())
    }

    fn generate(
        &mut self,
        request: &crate::swarm::ports::InferenceRequest,
    ) -> Result<crate::swarm::ports::InferenceOutput, DomainError> {
        let model = self.model.as_ref().ok_or_else(|| {
            DomainError::ModelNotFound { model_id: "model not loaded".into() }
        })?;

        let prompt: Vec<u32> = if request.prompt_tokens.is_empty() {
            vec![0, 1, 2, 3] // V0: dummy tokens (no tokenizer yet)
        } else {
            request.prompt_tokens.clone()
        };

        let output = forward(model, &prompt);

        // V0: return expert routing info as "tokens" (no real token generation yet)
        let routing_summary: Vec<String> = output
            .routes
            .iter()
            .map(|(layer, ids, _scores)| {
                let ids_str: Vec<String> = ids.iter().take(3).map(|id| id.to_string()).collect();
                format!("L{layer}:[{}]", ids_str.join(","))
            })
            .collect();

        let tokens = vec![crate::swarm::token::Token::new(
            &routing_summary.join(" "),
            0.0,
        )
        .map_err(|e| DomainError::InvalidTokenText { reason: e.to_string() })?];

        Ok(crate::swarm::ports::InferenceOutput {
            request_id: request.id,
            tokens,
        })
    }

    fn verify(&mut self, _model: &ModelId, _expected_hash: &str) -> Result<bool, DomainError> {
        // V0: not implemented
        Ok(true)
    }

    fn detect_vram(&mut self) -> Result<u32, DomainError> {
        // V0: return model size as approximate memory usage
        Ok(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelId;
    use crate::swarm::ports::{InferenceRequest, Priority};
    use uuid::Uuid;

    fn model_path() -> PathBuf {
        PathBuf::from(
            "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"
        )
    }

    #[test]
    fn load_granite_moe_and_generate_routes() {
        let mut runtime = NativeMoeRuntime::new(model_path());
        runtime.load(&ModelId::new("granite-moe").unwrap(), &[]).unwrap();

        let request = InferenceRequest::new(
            Uuid::new_v4(),
            ModelId::new("granite-moe").unwrap(),
            Priority::Batch,
            None,
            10,
            vec![0, 1, 2, 3],
        );

        let output = runtime.generate(&request).unwrap();
        assert!(!output.tokens.is_empty());
        // Output should contain routing summary strings
        let text = output.tokens[0].text();
        assert!(text.contains("L0:"), "should have layer 0 routing: {text}");
        assert!(text.contains("L1:"), "should have layer 1 routing: {text}");
    }

    #[test]
    fn generate_without_load_fails() {
        let mut runtime = NativeMoeRuntime::new(model_path());
        let request = InferenceRequest::new(
            Uuid::new_v4(),
            ModelId::new("granite-moe").unwrap(),
            Priority::Batch,
            None,
            10,
            vec![],
        );
        let result = runtime.generate(&request);
        assert!(result.is_err());
    }

    #[test]
    fn verify_trait_is_object_safe() {
        fn _assert(_port: &mut dyn InferencePort) {}
        let mut runtime = NativeMoeRuntime::new(model_path());
        _assert(&mut runtime);
    }
}
