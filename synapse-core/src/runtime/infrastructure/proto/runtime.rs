//! Protobuf message encoding/decoding for the runtime bridge.
//!
//! Hand-coded to avoid build-time protoc dependency and to stay
//! dependency-free — no prost or other protobuf crate needed.
//!
//! This module re-exports from the `encode` and `decode` submodules.

pub mod decode;
pub mod encode;

pub use decode::{
    decode_generate_response, decode_load_model_response, decode_verify_response,
    decode_vram_response,
};
pub use encode::{
    encode_generate_request, encode_load_model_request, encode_verify_request, encode_vram_request,
};
