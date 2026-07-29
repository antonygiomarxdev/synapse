//! Configuration loaded from config/default.toml.
//!
//! In V1, this reads the TOML file at startup.
//! Future: could be served by the gateway.

use std::path::Path;

/// Runtime bridge configuration.
pub struct RuntimeConfig {
    pub socket_path: String,
    pub max_retries: u32,
    pub read_timeout_secs: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            socket_path: "/tmp/synapse-runtime.sock".into(),
            max_retries: 3,
            read_timeout_secs: 30,
        }
    }
}

impl RuntimeConfig {
    /// Load from config/default.toml
    pub fn load() -> Self {
        let mut cfg = Self::default();
        let config_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(|p| p.join("config").join("default.toml"));

        if let Some(path) = config_path {
            if path.exists() {
                // Simple TOML parsing for V1
                let content = std::fs::read_to_string(&path).ok();
                if let Some(text) = content {
                    // Parse [runtime] section
                    for line in text.lines() {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_config_defaults() {
        let cfg = RuntimeConfig::default();
        assert_eq!(cfg.socket_path, "/tmp/synapse-runtime.sock");
        assert_eq!(cfg.max_retries, 3);
    }
}
