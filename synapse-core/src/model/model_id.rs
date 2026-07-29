use crate::shared::DomainError;
use serde::{Deserialize, Serialize};

/// Unique identifier for an AI model in the Synapse catalog.
///
/// Model IDs must be non-empty and in kebab-case (lowercase letters,
/// digits, and hyphens only). Examples: `"kimi-k3"`, `"mixtral-8x7b"`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModelId(String);

impl ModelId {
    /// Creates a new `ModelId` after validating the format.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidModelId`] if the ID is empty or
    /// contains characters other than lowercase letters, digits, and hyphens.
    pub fn new(id: impl Into<String>) -> Result<Self, DomainError> {
        let id: String = id.into();
        if id.is_empty() {
            return Err(DomainError::InvalidModelId {
                reason: "model ID must not be empty".into(),
            });
        }
        if !is_kebab_case(&id) {
            return Err(DomainError::InvalidModelId {
                reason: format!(
                    "model ID must be kebab-case (lowercase letters, digits, hyphens): '{id}'"
                ),
            });
        }
        Ok(Self(id))
    }

    /// Returns the model ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Validates that a string is kebab-case: lowercase alphanumeric or hyphen,
/// no leading/trailing hyphens, no consecutive hyphens.
fn is_kebab_case(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let bytes = s.as_bytes();
    // First and last must not be hyphens
    if bytes[0] == b'-' || bytes[bytes.len() - 1] == b'-' {
        return false;
    }
    let mut prev_hyphen = false;
    for &b in bytes {
        match b {
            b'a'..=b'z' | b'0'..=b'9' => prev_hyphen = false,
            b'-' => {
                if prev_hyphen {
                    return false; // consecutive hyphens
                }
                prev_hyphen = true;
            }
            _ => return false, // invalid character
        }
    }
    true
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- valid model IDs ---

    #[test]
    fn valid_kebab_case() {
        assert!(ModelId::new("kimi-k3").is_ok());
        assert!(ModelId::new("mixtral-8x7b").is_ok());
        assert!(ModelId::new("deepseek-v2-lite").is_ok());
    }

    #[test]
    fn single_word_is_valid() {
        assert!(ModelId::new("synapse").is_ok());
    }

    #[test]
    fn numbers_only_is_valid() {
        assert!(ModelId::new("42").is_ok());
    }

    // --- invalid model IDs ---

    #[test]
    fn empty_rejected() {
        let err = ModelId::new("").unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn uppercase_rejected() {
        assert!(ModelId::new("Kimi-K3").is_err());
    }

    #[test]
    fn spaces_rejected() {
        assert!(ModelId::new("kimi k3").is_err());
    }

    #[test]
    fn leading_hyphen_rejected() {
        assert!(ModelId::new("-kimi").is_err());
    }

    #[test]
    fn trailing_hyphen_rejected() {
        assert!(ModelId::new("kimi-").is_err());
    }

    #[test]
    fn consecutive_hyphens_rejected() {
        assert!(ModelId::new("kimi--k3").is_err());
    }

    #[test]
    fn special_chars_rejected() {
        assert!(ModelId::new("kimi_k3").is_err());
        assert!(ModelId::new("kimi.k3").is_err());
    }

    // --- Display ---

    #[test]
    fn display_returns_id_string() {
        let id = ModelId::new("kimi-k3").unwrap();
        assert_eq!(id.to_string(), "kimi-k3");
    }

    // --- Equality ---

    #[test]
    fn same_id_equal() {
        let a = ModelId::new("kimi-k3").unwrap();
        let b = ModelId::new("kimi-k3").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_id_not_equal() {
        let a = ModelId::new("kimi-k3").unwrap();
        let b = ModelId::new("mixtral-8x7b").unwrap();
        assert_ne!(a, b);
    }

    // --- as_str ---

    #[test]
    fn as_str_returns_input() {
        let id = ModelId::new("qwen2-moe").unwrap();
        assert_eq!(id.as_str(), "qwen2-moe");
    }

    // --- serialization ---

    #[test]
    fn serialize_deserialize_roundtrip() {
        let id = ModelId::new("kimi-k3").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        let parsed: ModelId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, parsed);
    }
}
