//! Unix socket bridge to Python vLLM subprocess.
//!
//! Implements [`InferencePort`] by encoding requests as protobuf
//! messages and sending them over a Unix domain socket to the
//! Python runtime process.

use crate::model::{ExpertId, ModelId};
use crate::runtime::ports::InferencePort;
use crate::shared::DomainError;
use crate::swarm::ports::{InferenceOutput, InferenceRequest};
use crate::swarm::token::Token;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

/// Bridge to a Python vLLM subprocess over a Unix domain socket.
pub struct UnixSocketBridge {
    socket_path: String,
}

impl UnixSocketBridge {
    /// Creates a new bridge connected to the given socket path.
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self { socket_path: socket_path.into() }
    }

    /// Sends a length-prefixed request and reads the length-prefixed response.
    ///
    /// Wire format: 4-byte big-endian payload length + payload bytes.
    fn send_request(&self, data: &[u8]) -> Result<Vec<u8>, DomainError> {
        let mut stream =
            UnixStream::connect(&self.socket_path).map_err(|e| DomainError::StorageError {
                message: format!("Failed to connect to runtime socket: {e}"),
            })?;

        // Write 4-byte big-endian length + payload
        let len = data.len();
        let len_bytes = (len as u32).to_be_bytes();
        stream.write_all(&len_bytes).map_err(|e| DomainError::StorageError {
            message: format!("Failed to write request length: {e}"),
        })?;
        stream.write_all(data).map_err(|e| DomainError::StorageError {
            message: format!("Failed to write request data: {e}"),
        })?;

        // Read 4-byte big-endian response length
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).map_err(|e| DomainError::StorageError {
            message: format!("Failed to read response length: {e}"),
        })?;
        let resp_len = u32::from_be_bytes(len_buf) as usize;

        // Read response payload
        let mut resp_data = vec![0u8; resp_len];
        if resp_len > 0 {
            stream.read_exact(&mut resp_data).map_err(|e| DomainError::StorageError {
                message: format!("Failed to read response data: {e}"),
            })?;
        }

        Ok(resp_data)
    }
}

impl InferencePort for UnixSocketBridge {
    fn load(&self, model: &ModelId, experts: &[ExpertId]) -> Result<(), DomainError> {
        let expert_indices: Vec<u32> = experts.iter().map(|e| e.index).collect();
        let req = crate::runtime::protocol::LoadModelRequest::new(model.clone(), expert_indices);
        let req_data =
            crate::runtime::infrastructure::proto::runtime::encode_load_model_request(&req);

        let mut framed = vec![1u8];
        framed.extend_from_slice(&req_data);

        let resp_data = self.send_request(&framed)?;
        let actual_data = if resp_data.len() > 1 { &resp_data[1..] } else { &resp_data };
        let resp =
            crate::runtime::infrastructure::proto::runtime::decode_load_model_response(actual_data);

        if resp.success { Ok(()) } else { Err(DomainError::StorageError { message: resp.error }) }
    }

    fn generate(&self, request: &InferenceRequest) -> Result<InferenceOutput, DomainError> {
        let bridge_req = crate::runtime::protocol::GenerateBridgeRequest::new(
            request.id.as_bytes().to_vec(),
            vec![], // token_ids populated by tokenizer in full impl
            request.max_tokens,
            0,
        );
        let req_data =
            crate::runtime::infrastructure::proto::runtime::encode_generate_request(&bridge_req);

        let mut framed = vec![3u8];
        framed.extend_from_slice(&req_data);

        let resp_data = self.send_request(&framed)?;
        let actual_data = if resp_data.len() > 1 { &resp_data[1..] } else { &resp_data };
        let resp =
            crate::runtime::infrastructure::proto::runtime::decode_generate_response(actual_data);

        let tokens: Vec<Token> = resp
            .token_ids
            .iter()
            .zip(resp.log_probs.iter())
            .map(|(_, &lp)| Token::new(String::new(), lp as f64))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(InferenceOutput { request_id: request.id, tokens })
    }

    fn verify(&self, model: &ModelId, expected_hash: &str) -> Result<bool, DomainError> {
        let req = crate::runtime::protocol::VerifyBridgeRequest::new(
            model.clone(),
            expected_hash.to_string(),
        );
        let req_data = crate::runtime::infrastructure::proto::runtime::encode_verify_request(&req);

        let mut framed = vec![5u8];
        framed.extend_from_slice(&req_data);

        let resp_data = self.send_request(&framed)?;
        let actual_data = if resp_data.len() > 1 { &resp_data[1..] } else { &resp_data };
        let resp =
            crate::runtime::infrastructure::proto::runtime::decode_verify_response(actual_data);

        Ok(resp.matches)
    }

    fn detect_vram(&self) -> Result<u32, DomainError> {
        let req = crate::runtime::protocol::VramBridgeRequest;
        let req_data = crate::runtime::infrastructure::proto::runtime::encode_vram_request(&req);

        let mut framed = vec![7u8];
        framed.extend_from_slice(&req_data);

        let resp_data = self.send_request(&framed)?;
        let actual_data = if resp_data.len() > 1 { &resp_data[1..] } else { &resp_data };
        let resp =
            crate::runtime::infrastructure::proto::runtime::decode_vram_response(actual_data);

        Ok(resp.available_mb)
    }
}
