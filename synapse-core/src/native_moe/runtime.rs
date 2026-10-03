/// Implementation of `InferencePort` for the native MoE runtime.
///
/// Loads models from GGUF files, runs the transformer forward pass with
/// external expert routing control, verifies weight integrity, and reports
/// memory usage.
use std::io::BufReader;
use std::path::PathBuf;

use crate::model::{ExpertId, ModelId};
use crate::native_moe::generate::{GenerateOutput, SamplingConfig, generate};
use crate::native_moe::model::MoeModel;
use crate::native_moe::model::Tensor;
use crate::runtime::ports::InferencePort;
use crate::shared::DomainError;
use crate::swarm::ports::{InferenceOutput, InferenceRequest};
use crate::swarm::token::Token;
use sha2::Digest;

/// GGUF-backed MoE inference runtime.
pub struct NativeMoeRuntime {
    model: Option<MoeModel>,
    model_path: Option<PathBuf>,
}

/// Calculate byte size of an optional tensor, accounting for f32 element size.
fn tensor_bytes(t: Option<&Tensor>) -> u64 {
    t.map(|tensor| (tensor.data.len() * std::mem::size_of::<f32>()) as u64).unwrap_or(0)
}

impl NativeMoeRuntime {
    /// Create a runtime that loads models from the given GGUF file.
    pub fn new(model_path: PathBuf) -> Self {
        NativeMoeRuntime { model: None, model_path: Some(model_path) }
    }
}

impl InferencePort for NativeMoeRuntime {
    fn load(&mut self, _model: &ModelId, _experts: &[ExpertId]) -> Result<(), DomainError> {
        let path = self
            .model_path
            .as_ref()
            .ok_or_else(|| DomainError::ModelNotFound { model_id: "no path configured".into() })?;

        let loaded =
            MoeModel::load_all(path).map_err(|e| DomainError::StorageError { message: e })?;

        self.model = Some(loaded);
        Ok(())
    }

    fn generate(&mut self, request: &InferenceRequest) -> Result<InferenceOutput, DomainError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| DomainError::ModelNotFound { model_id: "model not loaded".into() })?;

        let prompt_tokens: Vec<u32> = if request.prompt_tokens.is_empty() {
            vec![0, 1, 2, 3] // V0: dummy tokens (no tokenizer yet)
        } else {
            request.prompt_tokens.clone()
        };

        let config = SamplingConfig {
            max_tokens: request.max_tokens as usize,
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
            eos_token_id: None,
        };

        let output: GenerateOutput = generate(model, &prompt_tokens, &config)
            .map_err(|e| DomainError::StorageError { message: e.to_string() })?;

        // Convert generated token IDs to Token objects
        // Note: without a tokenizer, we use the token ID as text representation
        let tokens: Vec<Token> = output
            .tokens
            .into_iter()
            .enumerate()
            .map(|(i, tid)| {
                let logit =
                    output.logits.get(i).and_then(|v| v.get(tid as usize)).copied().unwrap_or(0.0);
                Token::new(tid.to_string(), logit as f64)
                    .map_err(|e| DomainError::InvalidTokenText { reason: e.to_string() })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(InferenceOutput { request_id: request.id, tokens })
    }

    fn verify(&mut self, model: &ModelId, expected_hash: &str) -> Result<bool, DomainError> {
        let _model = self
            .model
            .as_ref()
            .ok_or_else(|| DomainError::ModelNotFound { model_id: model.to_string() })?;

        let path = self
            .model_path
            .as_ref()
            .ok_or_else(|| DomainError::ModelNotFound { model_id: "no path configured".into() })?;

        let file = std::fs::File::open(path)
            .map_err(|e| DomainError::StorageError { message: e.to_string() })?;
        let mut hasher = sha2::Sha256::new();
        std::io::copy(&mut BufReader::new(file), &mut hasher)
            .map_err(|e| DomainError::StorageError { message: e.to_string() })?;
        let actual_hash = format!("{:x}", hasher.finalize());

        Ok(actual_hash == expected_hash)
    }

    fn detect_vram(&mut self) -> Result<u32, DomainError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| DomainError::ModelNotFound { model_id: "model not loaded".into() })?;

        // Estimate VRAM: sum of all tensor sizes in bytes / (1024*1024)
        let mut total_bytes: u64 = 0;
        total_bytes += tensor_bytes(model.token_embd.as_ref());
        total_bytes += tensor_bytes(model.output_norm.as_ref());
        total_bytes += tensor_bytes(model.output.as_ref());
        for layer in &model.layers {
            total_bytes += tensor_bytes(layer.attn_norm.as_ref());
            total_bytes += tensor_bytes(layer.attn_q.as_ref());
            total_bytes += tensor_bytes(layer.attn_k.as_ref());
            total_bytes += tensor_bytes(layer.attn_v.as_ref());
            total_bytes += tensor_bytes(layer.attn_output.as_ref());
            total_bytes += tensor_bytes(layer.ffn_norm.as_ref());
            total_bytes += tensor_bytes(Some(&layer.gate_inp));
            total_bytes += tensor_bytes(layer.gate_exps.as_ref());
            total_bytes += tensor_bytes(layer.up_exps.as_ref());
            total_bytes += tensor_bytes(layer.down_exps.as_ref());
        }

        Ok((total_bytes / (1024 * 1024)) as u32)
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
            "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b",
        )
    }

    #[test]
    #[ignore] // Requires GGUF model
    fn load_granite_moe_and_generate_tokens() {
        let mut runtime = NativeMoeRuntime::new(model_path());
        runtime.load(&ModelId::new("granite-moe").unwrap(), &[]).unwrap();

        let request = InferenceRequest::new(
            Uuid::new_v4(),
            ModelId::new("granite-moe").unwrap(),
            Priority::Batch,
            None,
            5,
            vec![0, 1, 2, 3],
        );

        let output = runtime.generate(&request).unwrap();
        assert!(!output.tokens.is_empty());
        // Should generate max_tokens tokens
        assert_eq!(output.tokens.len(), 5);
        // Each token should be a valid token ID string
        for token in &output.tokens {
            let _tid: u32 = token.text().parse().expect("token text should be numeric ID");
        }
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

    #[test]
    #[ignore] // Requires GGUF model
    fn verify_matches_correct_hash() {
        let mut runtime = NativeMoeRuntime::new(model_path());
        runtime.load(&ModelId::new("granite-moe").unwrap(), &[]).unwrap();

        let file = std::fs::File::open(model_path()).unwrap();
        let mut hasher = sha2::Sha256::new();
        std::io::copy(&mut BufReader::new(file), &mut hasher).unwrap();
        let expected_hash = format!("{:x}", hasher.finalize());

        let result = runtime.verify(&ModelId::new("granite-moe").unwrap(), &expected_hash).unwrap();
        assert!(result);
    }

    #[test]
    #[ignore] // Requires GGUF model
    fn verify_rejects_wrong_hash() {
        let mut runtime = NativeMoeRuntime::new(model_path());
        runtime.load(&ModelId::new("granite-moe").unwrap(), &[]).unwrap();

        let result = runtime.verify(&ModelId::new("granite-moe").unwrap(), "deadbeef").unwrap();
        assert!(!result);
    }

    #[test]
    #[ignore] // Requires GGUF model
    fn detect_vram_returns_positive() {
        let mut runtime = NativeMoeRuntime::new(model_path());
        runtime.load(&ModelId::new("granite-moe").unwrap(), &[]).unwrap();

        let vram = runtime.detect_vram().unwrap();
        assert!(vram > 0, "VRAM estimate should be positive, got {vram}");
    }
}
