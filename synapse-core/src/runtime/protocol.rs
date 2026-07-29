use crate::model::ModelId;

/// Request to load a model with specific experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadModelRequest {
    pub model_id: ModelId,
    pub expert_indices: Vec<u32>,
}

/// Response after model load attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadModelResponse {
    pub success: bool,
    pub error: String,
    pub loaded_experts: u32,
}

/// Token generation request (bridge-level).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateBridgeRequest {
    pub request_id: Vec<u8>,
    pub token_ids: Vec<u32>,
    pub seed: u32,
    pub max_tokens: u32,
}

/// Token generation response (bridge-level).
#[derive(Debug, Clone)]
pub struct GenerateBridgeResponse {
    pub request_id: Vec<u8>,
    pub token_ids: Vec<u32>,
    pub log_probs: Vec<f32>,
    pub finished: bool,
}

/// SHA256 verification request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyBridgeRequest {
    pub model_id: ModelId,
    pub expected_sha256: String,
}

/// SHA256 verification response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyBridgeResponse {
    pub matches: bool,
    pub actual_sha256: String,
}

/// VRAM query request (unit struct — no fields).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VramBridgeRequest;

/// VRAM query response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VramBridgeResponse {
    pub total_mb: u32,
    pub available_mb: u32,
}

impl LoadModelRequest {
    /// Creates a new `LoadModelRequest`.
    pub fn new(model_id: ModelId, expert_indices: Vec<u32>) -> Self {
        Self { model_id, expert_indices }
    }
}

impl LoadModelResponse {
    /// Creates a successful response.
    pub fn ok(loaded_experts: u32) -> Self {
        Self { success: true, error: String::new(), loaded_experts }
    }

    /// Creates a failure response.
    pub fn err(error: impl Into<String>) -> Self {
        Self { success: false, error: error.into(), loaded_experts: 0 }
    }
}

impl GenerateBridgeRequest {
    /// Creates a new `GenerateBridgeRequest`.
    pub fn new(request_id: Vec<u8>, token_ids: Vec<u32>, seed: u32, max_tokens: u32) -> Self {
        Self { request_id, token_ids, seed, max_tokens }
    }
}

impl GenerateBridgeResponse {
    /// Creates a new `GenerateBridgeResponse`.
    pub fn new(
        request_id: Vec<u8>,
        token_ids: Vec<u32>,
        log_probs: Vec<f32>,
        finished: bool,
    ) -> Self {
        Self { request_id, token_ids, log_probs, finished }
    }
}

impl VerifyBridgeRequest {
    /// Creates a new `VerifyBridgeRequest`.
    pub fn new(model_id: ModelId, expected_sha256: String) -> Self {
        Self { model_id, expected_sha256 }
    }
}

impl VerifyBridgeResponse {
    /// Creates a successful match response.
    pub fn matched(actual_sha256: String) -> Self {
        Self { matches: true, actual_sha256 }
    }

    /// Creates a mismatch response.
    pub fn mismatched(expected: String, actual: String) -> Self {
        Self { matches: false, actual_sha256: format!("expected {expected}, got {actual}") }
    }
}

impl VramBridgeResponse {
    /// Creates a new `VramBridgeResponse`.
    pub fn new(total_mb: u32, available_mb: u32) -> Self {
        Self { total_mb, available_mb }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_id() -> ModelId {
        ModelId::new("mixtral-8x7b").unwrap()
    }

    #[test]
    fn load_model_request_construction() {
        let req = LoadModelRequest::new(model_id(), vec![0, 3, 7]);
        assert_eq!(req.model_id.as_str(), "mixtral-8x7b");
        assert_eq!(req.expert_indices, vec![0, 3, 7]);
    }

    #[test]
    fn load_model_request_empty_experts() {
        let req = LoadModelRequest::new(model_id(), vec![]);
        assert!(req.expert_indices.is_empty());
    }

    #[test]
    fn load_model_response_ok() {
        let resp = LoadModelResponse::ok(3);
        assert!(resp.success);
        assert_eq!(resp.loaded_experts, 3);
        assert!(resp.error.is_empty());
    }

    #[test]
    fn load_model_response_err() {
        let resp = LoadModelResponse::err("OOM");
        assert!(!resp.success);
        assert_eq!(resp.error, "OOM");
        assert_eq!(resp.loaded_experts, 0);
    }

    #[test]
    fn generate_bridge_request_seed_zero() {
        let req = GenerateBridgeRequest::new(b"r1".to_vec(), vec![1, 2], 0, 100);
        assert_eq!(req.seed, 0);
        assert_eq!(req.max_tokens, 100);
    }

    #[test]
    fn generate_bridge_response_unfinished() {
        let resp = GenerateBridgeResponse::new(b"r1".to_vec(), vec![42], vec![-0.5], false);
        assert!(!resp.finished);
        assert_eq!(resp.token_ids, vec![42]);
    }

    #[test]
    fn verify_bridge_response_matched() {
        let resp = VerifyBridgeResponse::matched("abc123".into());
        assert!(resp.matches);
        assert_eq!(resp.actual_sha256, "abc123");
    }

    #[test]
    fn verify_bridge_response_mismatched() {
        let resp = VerifyBridgeResponse::mismatched("abc".into(), "def".into());
        assert!(!resp.matches);
        assert!(resp.actual_sha256.contains("expected"));
    }

    #[test]
    fn vram_bridge_response() {
        let resp = VramBridgeResponse::new(16384, 8192);
        assert_eq!(resp.total_mb, 16384);
        assert_eq!(resp.available_mb, 8192);
    }
}
