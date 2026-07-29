//! Runtime-agnostic inference adapter domain.
//!
//! Defines the [`InferencePort`] trait that any inference runtime
//! (vLLM, llama.cpp, SGLang) implements. Protocol value objects
//! mirror the bridge protobuf schema 1:1.

pub mod infrastructure;
pub mod ports;
pub mod protocol;

pub use ports::InferencePort;
pub use protocol::{
    GenerateBridgeRequest, GenerateBridgeResponse, LoadModelRequest, LoadModelResponse,
    VerifyBridgeRequest, VerifyBridgeResponse, VramBridgeRequest, VramBridgeResponse,
};
