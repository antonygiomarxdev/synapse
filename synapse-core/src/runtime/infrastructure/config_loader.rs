//! Loads configuration from config/default.toml at startup.
//!
//! Infrastructure adapter — reads filesystem, produces domain RuntimeConfig.

use std::path::Path;

use crate::runtime::protocol::RuntimeConfig;

/// Loads runtime configuration from the project config file.
pub fn load_runtime_config() -> RuntimeConfig {
    let mut cfg = RuntimeConfig::default();

    // Look for config/default.toml relative to the crate
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let config_path = manifest_dir.parent().map(|p| p.join("config").join("default.toml"));

    if let Some(path) = config_path {
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                for line in content.lines() {
                    let line = line.trim();
                    if let Some(value) = line.strip_prefix("socket_path = \"") {
                        if let Some(end) = value.rfind('"') {
                            cfg.socket_path = value[..end].to_string();
                        }
                    } else if let Some(value) = line.strip_prefix("max_retries = ") {
                        if let Ok(n) = value.trim().parse::<u32>() {
                            cfg.max_retries = n;
                        }
                    } else if let Some(value) = line.strip_prefix("read_timeout_secs = ") {
                        if let Ok(n) = value.trim().parse::<u64>() {
                            cfg.read_timeout_secs = n;
                        }
                    }
                }
            }
        }
    }

    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_runtime_config_defaults() {
        let cfg = load_runtime_config();
        // Should always have default socket_path
        assert!(cfg.socket_path.contains("synapse-runtime"));
    }
}
