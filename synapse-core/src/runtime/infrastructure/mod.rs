//! Infrastructure adapters for the runtime port.
//!
//! Currently: Unix socket bridge to Python vLLM subprocess.
//! Future: llama.cpp, SGLang adapters.

pub mod config_loader;
pub mod proto;
pub mod unix_socket_bridge;
