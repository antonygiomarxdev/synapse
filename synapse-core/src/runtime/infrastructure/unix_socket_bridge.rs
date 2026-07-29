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
    /// Retries up to `MAX_RETRIES` times with exponential backoff.
    fn send_request(&self, request_data: &[u8]) -> Result<Vec<u8>, DomainError> {
        const MAX_RETRIES: u32 = 3;
        let mut last_error = String::new();

        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                // Exponential backoff: 200ms (attempt 1), 400ms (attempt 2)
                std::thread::sleep(std::time::Duration::from_millis(100 * (1 << attempt) as u64));
            }

            match Self::try_send(&self.socket_path, request_data) {
                Ok(data) => return Ok(data),
                Err(e) => last_error = e,
            }
        }

        Err(DomainError::StorageError {
            message: format!("Runtime request failed after {MAX_RETRIES} retries: {last_error}"),
        })
    }

    /// Attempt a single send/receive transaction over the Unix socket.
    fn try_send(socket_path: &str, request_data: &[u8]) -> Result<Vec<u8>, String> {
        let mut stream = UnixStream::connect(socket_path).map_err(|e| format!("connect: {e}"))?;
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(30)))
            .map_err(|e| format!("set timeout: {e}"))?;

        let len_bytes = (request_data.len() as u32).to_be_bytes();
        stream.write_all(&len_bytes).map_err(|e| format!("write len: {e}"))?;
        stream.write_all(request_data).map_err(|e| format!("write data: {e}"))?;

        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).map_err(|e| format!("read len: {e}"))?;
        let resp_len = u32::from_be_bytes(len_buf) as usize;

        let mut resp_data = vec![0u8; resp_len];
        stream.read_exact(&mut resp_data).map_err(|e| format!("read data: {e}"))?;

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

        if resp.token_ids.len() != resp.log_probs.len() {
            return Err(DomainError::InvalidTokenText {
                reason: format!(
                    "token_ids length ({}) != log_probs length ({})",
                    resp.token_ids.len(),
                    resp.log_probs.len()
                ),
            });
        }

        let tokens: Vec<Token> = resp
            .token_ids
            .iter()
            .zip(resp.log_probs.iter())
            .map(|(&tid, &lp)| Token::new(tid.to_string(), lp as f64))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_request_retries_on_failure() {
        // When socket doesn't exist, send_request should return an error
        let bridge = UnixSocketBridge::new("/tmp/nonexistent-synapse-socket.sock");
        let result = bridge.send_request(&[1, 2, 3]);
        assert!(result.is_err());
        match result {
            Err(DomainError::StorageError { .. }) => {} // expected
            other => panic!("Expected StorageError, got {other:?}"),
        }
    }
}
