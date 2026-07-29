//! Protobuf message encoding for the runtime bridge.
//!
//! Hand-coded to avoid build-time protoc dependency and to stay
//! dependency-free — no prost or other protobuf crate needed.

use crate::runtime::protocol::{
    GenerateBridgeRequest, LoadModelRequest, VerifyBridgeRequest, VramBridgeRequest,
};

// ── Wire-type constants ──────────────────────────────────────────────

/// Wire type for varint-encoded fields (int32, int64, uint32, uint64, bool, enum).
pub(crate) const WIRE_VARINT: u32 = 0;

/// Wire type for length-delimited fields (bytes, string, embedded messages, packed repeated).
pub(crate) const WIRE_LEN: u32 = 2;

// ── Public encoding functions ────────────────────────────────────────

/// Encode a `LoadModelRequest` to protobuf bytes.
///
/// Wire format: field 1 (string model_id), field 2 (packed repeated uint32 expert_indices).
pub fn encode_load_model_request(req: &LoadModelRequest) -> Vec<u8> {
    let mut buf = Vec::new();
    // field 1: model_id (string)
    encode_string(&mut buf, 1, req.model_id.as_str());
    // field 2: expert_indices (packed repeated uint32)
    encode_packed_uint32(&mut buf, 2, &req.expert_indices);
    buf
}

/// Encode a `GenerateBridgeRequest` to protobuf bytes.
pub fn encode_generate_request(req: &GenerateBridgeRequest) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_bytes(&mut buf, 1, &req.request_id);
    encode_packed_uint32(&mut buf, 2, &req.token_ids);
    encode_uint32(&mut buf, 3, req.seed);
    encode_uint32(&mut buf, 4, req.max_tokens);
    buf
}

/// Encode a `VerifyBridgeRequest` to protobuf bytes.
pub fn encode_verify_request(req: &VerifyBridgeRequest) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_string(&mut buf, 1, req.model_id.as_str());
    encode_string(&mut buf, 2, &req.expected_sha256);
    buf
}

/// Encode a `VramBridgeRequest` to protobuf bytes (empty message).
pub fn encode_vram_request(_req: &VramBridgeRequest) -> Vec<u8> {
    Vec::new()
}

// ── Protobuf encoding helpers ────────────────────────────────────────

/// Encode an unsigned 64-bit integer as a protobuf varint.
pub(crate) fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value > 0x7F {
        buf.push((value as u8 & 0x7F) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

/// Encode a field tag (field_number << 3 | wire_type) as a varint.
pub(crate) fn encode_tag(buf: &mut Vec<u8>, field_number: u32, wire_type: u32) {
    encode_varint(buf, ((field_number << 3) | wire_type) as u64);
}

/// Encode a `uint32` field.
pub(crate) fn encode_uint32(buf: &mut Vec<u8>, field_number: u32, value: u32) {
    encode_tag(buf, field_number, WIRE_VARINT);
    encode_varint(buf, value as u64);
}

/// Encode a length-delimited bytes field.
pub(crate) fn encode_bytes(buf: &mut Vec<u8>, field_number: u32, data: &[u8]) {
    encode_tag(buf, field_number, WIRE_LEN);
    encode_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

/// Encode a string field (delegates to `encode_bytes`).
pub(crate) fn encode_string(buf: &mut Vec<u8>, field_number: u32, value: &str) {
    encode_bytes(buf, field_number, value.as_bytes());
}

/// Encode a packed repeated uint32 field.
///
/// Skips encoding when the value slice is empty — omitting the field
/// is semantically equivalent to an empty packed list in protobuf.
pub(crate) fn encode_packed_uint32(buf: &mut Vec<u8>, field_number: u32, values: &[u32]) {
    if values.is_empty() {
        return;
    }
    let mut packed = Vec::with_capacity(values.len() * 5);
    for &v in values {
        encode_varint(&mut packed, v as u64);
    }
    encode_bytes(buf, field_number, &packed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelId;

    fn model_id() -> ModelId {
        ModelId::new("mixtral-8x7b").unwrap()
    }

    #[test]
    fn encode_empty_load_model_request() {
        let req = LoadModelRequest::new(model_id(), vec![]);
        let data = encode_load_model_request(&req);
        // field 1: string "mixtral-8x7b" = 0x0a + len(12) + "mixtral-8x7b"
        assert!(!data.is_empty());
        assert_eq!(data[0], 0x0a); // tag for field 1, wire type 2
        assert_eq!(data[1], 12); // string length
    }

    #[test]
    fn encode_load_model_with_experts() {
        let req = LoadModelRequest::new(model_id(), vec![0, 3, 7]);
        let data = encode_load_model_request(&req);
        assert!(data.len() > 14); // has the expert indices packed field
    }

    #[test]
    fn encode_generate_req() {
        let req = GenerateBridgeRequest::new(b"rid".to_vec(), vec![1, 2, 3], 0, 100);
        let data = encode_generate_request(&req);
        assert!(!data.is_empty());
    }

    #[test]
    fn encode_vram_request_empty() {
        let req = VramBridgeRequest;
        let data = encode_vram_request(&req);
        assert!(data.is_empty());
    }

    #[test]
    fn encode_verify_req() {
        let req = VerifyBridgeRequest::new(model_id(), "abc123".into());
        let data = encode_verify_request(&req);
        assert!(!data.is_empty());
    }
}
