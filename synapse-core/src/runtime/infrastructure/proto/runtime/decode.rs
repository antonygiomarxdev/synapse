//! Protobuf message decoding for the runtime bridge.
//!
//! Hand-coded to avoid build-time protoc dependency and to stay
//! dependency-free — no prost or other protobuf crate needed.

use super::encode::{WIRE_LEN, WIRE_VARINT};
use crate::runtime::protocol::{
    GenerateBridgeResponse, LoadModelResponse, VerifyBridgeResponse, VramBridgeResponse,
};

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
    let mut result = Vec::new();
    for &(num, wire_type, payload) in fields {
        if num == field_num && wire_type == WIRE_LEN {
            let mut off = 0;
            while off < payload.len() {
                if let Ok(v) = decode_varint(payload, &mut off) {
                    result.push(v as u32);
                } else {
                    break;
                }
            }
        }
    }
    result
}

fn decode_repeated_float(fields: &ParsedFields, field_num: u32) -> Vec<f32> {
    let mut result = Vec::new();
    for &(num, wire_type, payload) in fields {
        if num == field_num && wire_type == WIRE_LEN {
            if payload.len() % 4 != 0 {
                continue;
            }
            let count = payload.len() / 4;
            for i in 0..count {
                let start = i * 4;
                let end = (i + 1) * 4;
                let bytes: [u8; 4] = payload[start..end].try_into().unwrap();
                result.push(f32::from_le_bytes(bytes));
            }
        }
    }
    result
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

// ── Public decoding functions ────────────────────────────────────────

/// Decode a `LoadModelResponse` from protobuf bytes.
pub fn decode_load_model_response(data: &[u8]) -> Result<LoadModelResponse, String> {
    let fields = parse_message(data)?;
    Ok(LoadModelResponse {
        success: decode_bool(&fields, 1, false),
        error: decode_string_field(&fields, 2, ""),
        loaded_experts: decode_uint32(&fields, 3, 0),
    })
}

/// Decode a `GenerateBridgeResponse` from protobuf bytes.
pub fn decode_generate_response(data: &[u8]) -> Result<GenerateBridgeResponse, String> {
    let fields = parse_message(data)?;
    Ok(GenerateBridgeResponse {
        request_id: decode_bytes_field(&fields, 1),
        token_ids: decode_repeated_uint32(&fields, 2),
        log_probs: decode_repeated_float(&fields, 3),
        finished: decode_bool(&fields, 4, false),
    })
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
pub fn decode_verify_response(data: &[u8]) -> Result<VerifyBridgeResponse, String> {
    let fields = parse_message(data)?;
    Ok(VerifyBridgeResponse {
        matches: decode_bool(&fields, 1, false),
        actual_sha256: decode_string_field(&fields, 2, ""),
    })
}

/// Decode a `VramBridgeResponse` from protobuf bytes.
pub fn decode_vram_response(data: &[u8]) -> Result<VramBridgeResponse, String> {
    let fields = parse_message(data)?;
    Ok(VramBridgeResponse {
        total_mb: decode_uint32(&fields, 1, 0),
        available_mb: decode_uint32(&fields, 2, 0),
    })
}

#[cfg(test)]
mod tests {
    use super::super::encode::{
        encode_bytes, encode_packed_uint32, encode_string, encode_tag, encode_varint,
    };
    use super::*;

    #[test]
    fn decode_load_model_response_ok() {
        let resp = LoadModelResponse::ok(3);
        let encoded = encode_load_model_response_for_test(&resp);
        let decoded = decode_load_model_response(&encoded).unwrap();
        assert!(decoded.success);
        assert_eq!(decoded.loaded_experts, 3);
    }

    #[test]
    fn decode_load_model_response_err() {
        let resp = LoadModelResponse::err("OOM");
        let encoded = encode_load_model_response_for_test(&resp);
        let decoded = decode_load_model_response(&encoded).unwrap();
        assert!(!decoded.success);
        assert_eq!(decoded.error, "OOM");
    }

    #[test]
    fn decode_generate_response_with_tokens() {
        let resp =
            GenerateBridgeResponse::new(b"r1".to_vec(), vec![7, 8, 9], vec![-0.1, -0.2], true);
        let encoded = encode_generate_response_for_test(&resp);
        let decoded = decode_generate_response(&encoded).unwrap();
        assert_eq!(decoded.token_ids, vec![7, 8, 9]);
        assert_eq!(decoded.log_probs.len(), 2);
        assert!(decoded.finished);
    }

    #[test]
    fn test_decode_vram_response() {
        let resp = VramBridgeResponse::new(16384, 8192);
        let encoded = encode_vram_response_for_test(&resp);
        let decoded = decode_vram_response(&encoded).unwrap();
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
