//! Protobuf message encoding/decoding for the runtime bridge.
//!
//! Hand-coded to avoid build-time protoc dependency and to stay
//! dependency-free — no prost or other protobuf crate needed.
use crate::runtime::protocol::{
    GenerateBridgeRequest, GenerateBridgeResponse, LoadModelRequest, LoadModelResponse,
    VerifyBridgeRequest, VerifyBridgeResponse, VramBridgeRequest, VramBridgeResponse,
};

// ── Wire-type constants ──────────────────────────────────────────────

/// Wire type for varint-encoded fields (int32, int64, uint32, uint64, bool, enum).
const WIRE_VARINT: u32 = 0;

/// Wire type for length-delimited fields (bytes, string, embedded messages, packed repeated).
const WIRE_LEN: u32 = 2;

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
fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value > 0x7F {
        buf.push((value as u8 & 0x7F) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

/// Encode a field tag (field_number << 3 | wire_type) as a varint.
fn encode_tag(buf: &mut Vec<u8>, field_number: u32, wire_type: u32) {
    encode_varint(buf, ((field_number << 3) | wire_type) as u64);
}

/// Encode a `uint32` field.
fn encode_uint32(buf: &mut Vec<u8>, field_number: u32, value: u32) {
    encode_tag(buf, field_number, WIRE_VARINT);
    encode_varint(buf, value as u64);
}

/// Encode a length-delimited bytes field.
fn encode_bytes(buf: &mut Vec<u8>, field_number: u32, data: &[u8]) {
    encode_tag(buf, field_number, WIRE_LEN);
    encode_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

/// Encode a string field (delegates to `encode_bytes`).
fn encode_string(buf: &mut Vec<u8>, field_number: u32, value: &str) {
    encode_bytes(buf, field_number, value.as_bytes());
}

/// Encode a packed repeated uint32 field.
///
/// Skips encoding when the value slice is empty — omitting the field
/// is semantically equivalent to an empty packed list in protobuf.
fn encode_packed_uint32(buf: &mut Vec<u8>, field_number: u32, values: &[u32]) {
    if values.is_empty() {
        return;
    }
    let mut packed = Vec::with_capacity(values.len() * 5);
    for &v in values {
        encode_varint(&mut packed, v as u64);
    }
    encode_bytes(buf, field_number, &packed);
}

type ParsedFields<'a> = Vec<(u32, u32, &'a [u8])>;

// ── Protobuf decoding helpers ──────────────────────────────────────

fn decode_varint(data: &[u8], offset: &mut usize) -> Result<u64, String> {
    let mut value: u64 = 0;
    let mut shift = 0;
    while *offset < data.len() {
        let byte = data[*offset];
        *offset += 1;
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 64 {
            return Err("Varint too long".into());
        }
    }

    Err("Truncated varint".into())
}

fn decode_uint32(fields: &ParsedFields, field_num: u32, default: u32) -> u32 {
    for &(num, wire_type, payload) in fields {
        if num == field_num && wire_type == WIRE_VARINT {
            let mut off = 0;
            if let Ok(v) = decode_varint(payload, &mut off) {
                return v as u32;
            }
        }
    }
    default
}

fn decode_bool(fields: &ParsedFields, field_num: u32, default: bool) -> bool {
    decode_uint32(fields, field_num, if default { 1 } else { 0 }) != 0
}

fn decode_string_field(fields: &ParsedFields, field_num: u32, default: &str) -> String {
    for &(num, wire_type, payload) in fields {
        if num == field_num && wire_type == WIRE_LEN {
            return String::from_utf8_lossy(payload).to_string();
        }
    }
    default.to_string()
}

fn decode_repeated_uint32(fields: &ParsedFields, field_num: u32) -> Vec<u32> {
    for &(num, wire_type, payload) in fields {
        if num == field_num && wire_type == WIRE_LEN {
            let mut result = Vec::new();
            let mut off = 0;
            while off < payload.len() {
                if let Ok(v) = decode_varint(payload, &mut off) {
                    result.push(v as u32);
                } else {
                    break;
                }
            }
            return result;
        }
    }
    Vec::new()
}

fn decode_repeated_float(fields: &ParsedFields, field_num: u32) -> Vec<f32> {
    for &(num, wire_type, payload) in fields {
        if num == field_num && wire_type == WIRE_LEN {
            let count = payload.len() / 4;
            let mut result = Vec::with_capacity(count);
            for i in 0..count {
                let bytes: [u8; 4] = payload[i * 4..(i + 1) * 4].try_into().unwrap_or([0; 4]);
                result.push(f32::from_le_bytes(bytes));
            }
            return result;
        }
    }
    Vec::new()
}

/// Parse a protobuf message into field tuples: (field_number, wire_type, payload).
fn parse_message(data: &[u8]) -> Result<ParsedFields<'_>, String> {
    let mut fields = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let tag = decode_varint(data, &mut offset)?;
        let field_number = (tag >> 3) as u32;
        let wire_type = (tag & 0x07) as u32;

        match wire_type {
            WIRE_VARINT => {
                // Varint — payload is the encoded varint bytes
                let start = offset;
                let _value = decode_varint(data, &mut offset)?;
                fields.push((field_number, wire_type, &data[start..offset]));
            }
            WIRE_LEN => {
                // Length-delimited
                let length = decode_varint(data, &mut offset)? as usize;
                if offset + length > data.len() {
                    return Err(format!(
                        "Truncated field {field_number}: declared {length} bytes, available {}",
                        data.len() - offset,
                    ));
                }
                let payload = &data[offset..offset + length];
                offset += length;
                fields.push((field_number, wire_type, payload));
            }
            _ => return Err(format!("Unknown wire type: {wire_type}")),
        }
    }
    Ok(fields)
}

/// Decode a `LoadModelResponse` from protobuf bytes.
pub fn decode_load_model_response(data: &[u8]) -> LoadModelResponse {
    if let Ok(fields) = parse_message(data) {
        return LoadModelResponse {
            success: decode_bool(&fields, 1, false),
            error: decode_string_field(&fields, 2, ""),
            loaded_experts: decode_uint32(&fields, 3, 0),
        };
    }
    LoadModelResponse {
        success: false,
        error: "Failed to parse response".into(),
        loaded_experts: 0,
    }
}

/// Decode a `GenerateBridgeResponse` from protobuf bytes.
pub fn decode_generate_response(data: &[u8]) -> GenerateBridgeResponse {
    if let Ok(fields) = parse_message(data) {
        return GenerateBridgeResponse {
            request_id: decode_bytes_field(&fields, 1),
            token_ids: decode_repeated_uint32(&fields, 2),
            log_probs: decode_repeated_float(&fields, 3),
            finished: decode_bool(&fields, 4, false),
        };
    }
    GenerateBridgeResponse {
        request_id: vec![],
        token_ids: vec![],
        log_probs: vec![],
        finished: false,
    }
}

fn decode_bytes_field(fields: &ParsedFields, field_num: u32) -> Vec<u8> {
    for &(num, wire_type, payload) in fields {
        if num == field_num && wire_type == WIRE_LEN {
            return payload.to_vec();
        }
    }
    vec![]
}

/// Decode a `VerifyBridgeResponse` from protobuf bytes.
pub fn decode_verify_response(data: &[u8]) -> VerifyBridgeResponse {
    if let Ok(fields) = parse_message(data) {
        return VerifyBridgeResponse {
            matches: decode_bool(&fields, 1, false),
            actual_sha256: decode_string_field(&fields, 2, ""),
        };
    }
    VerifyBridgeResponse { matches: false, actual_sha256: "Failed to parse response".into() }
}

/// Decode a `VramBridgeResponse` from protobuf bytes.
pub fn decode_vram_response(data: &[u8]) -> VramBridgeResponse {
    if let Ok(fields) = parse_message(data) {
        return VramBridgeResponse {
            total_mb: decode_uint32(&fields, 1, 0),
            available_mb: decode_uint32(&fields, 2, 0),
        };
    }
    VramBridgeResponse { total_mb: 0, available_mb: 0 }
}

#[cfg(test)]
mod decode_tests {
    use super::*;

    #[test]
    fn decode_load_model_response_ok() {
        // Encode a success response manually
        let resp = LoadModelResponse::ok(3);
        let encoded = encode_load_model_response_for_test(&resp);
        let decoded = decode_load_model_response(&encoded);
        assert!(decoded.success);
        assert_eq!(decoded.loaded_experts, 3);
    }

    #[test]
    fn decode_load_model_response_err() {
        let resp = LoadModelResponse::err("OOM");
        let encoded = encode_load_model_response_for_test(&resp);
        let decoded = decode_load_model_response(&encoded);
        assert!(!decoded.success);
        assert_eq!(decoded.error, "OOM");
    }

    #[test]
    fn decode_generate_response_with_tokens() {
        let resp =
            GenerateBridgeResponse::new(b"r1".to_vec(), vec![7, 8, 9], vec![-0.1, -0.2], true);
        let encoded = encode_generate_response_for_test(&resp);
        let decoded = decode_generate_response(&encoded);
        assert_eq!(decoded.token_ids, vec![7, 8, 9]);
        assert_eq!(decoded.log_probs.len(), 2);
        assert!(decoded.finished);
    }

    #[test]
    fn test_decode_vram_response() {
        let resp = VramBridgeResponse::new(16384, 8192);
        let encoded = encode_vram_response_for_test(&resp);
        let decoded = super::decode_vram_response(&encoded);
        assert_eq!(decoded.total_mb, 16384);
        assert_eq!(decoded.available_mb, 8192);
    }

    // Helper: manual encoding for test roundtrips
    fn encode_load_model_response_for_test(resp: &LoadModelResponse) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_varint_field(&mut buf, 1, if resp.success { 1 } else { 0 });
        if !resp.error.is_empty() {
            encode_string(&mut buf, 2, &resp.error);
        }
        encode_varint_field(&mut buf, 3, resp.loaded_experts as u64);
        buf
    }

    fn encode_generate_response_for_test(resp: &GenerateBridgeResponse) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_bytes(&mut buf, 1, &resp.request_id);
        encode_packed_uint32(&mut buf, 2, &resp.token_ids);
        encode_packed_float(&mut buf, 3, &resp.log_probs);
        encode_varint_field(&mut buf, 4, if resp.finished { 1 } else { 0 });
        buf
    }

    fn encode_vram_response_for_test(resp: &VramBridgeResponse) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_varint_field(&mut buf, 1, resp.total_mb as u64);
        encode_varint_field(&mut buf, 2, resp.available_mb as u64);
        buf
    }

    fn encode_varint_field(buf: &mut Vec<u8>, field_number: u32, value: u64) {
        encode_tag(buf, field_number, 0);
        encode_varint(buf, value);
    }

    fn encode_packed_float(buf: &mut Vec<u8>, field_number: u32, values: &[f32]) {
        if values.is_empty() {
            return;
        }
        let mut packed = Vec::with_capacity(values.len() * 4);
        for &v in values {
            packed.extend_from_slice(&v.to_le_bytes());
        }
        encode_bytes(buf, field_number, &packed);
    }
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
