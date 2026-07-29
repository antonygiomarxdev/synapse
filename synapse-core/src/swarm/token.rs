use crate::shared::DomainError;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const MAX_TOKEN_TEXT_LEN: usize = 65_536;

/// A single generated token and its model-assigned log-probability.
///
/// Tokens are the atomic unit of consensus. Two tokens are equal if
/// their text and log_prob are equal. The `id` is a UUID for tracing
/// individual tokens through the swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    id: Uuid,
    text: String,
    log_prob: f64,
}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        self.text == other.text && self.log_prob == other.log_prob
    }
}

impl Token {
    /// Creates a new `Token`.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidTokenLogProb`] if `log_prob` is
    /// NaN or infinite. Returns [`DomainError::InvalidTokenText`] if
    /// `text` exceeds `MAX_TOKEN_TEXT_LEN`.
    pub fn new(text: impl Into<String>, log_prob: f64) -> Result<Self, DomainError> {
        if !log_prob.is_finite() {
            return Err(DomainError::InvalidTokenLogProb { value: log_prob });
        }
        let text: String = text.into();
        if text.len() > MAX_TOKEN_TEXT_LEN {
            return Err(DomainError::InvalidTokenText {
                reason: format!("text exceeds {MAX_TOKEN_TEXT_LEN} bytes"),
            });
        }
        Ok(Self { id: Uuid::new_v4(), text, log_prob })
    }

    /// Unique trace identifier for this token.
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// The generated text fragment.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The model log-probability for this token.
    pub fn log_prob(&self) -> f64 {
        self.log_prob
    }

    /// True if this token has no text (e.g., padding / end-of-stream).
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_rejects_nan_log_prob() {
        let result = Token::new("hello", f64::NAN);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().to_string(), "invalid token log_prob: NaN (must be finite)");
    }
    #[test]
    fn non_empty_token_is_not_empty() {
        let token = Token::new("hello", -1.0).unwrap();
        assert!(!token.is_empty());
    }

    #[test]
    fn token_rejects_infinite_log_prob() {
        let result = Token::new("hello", f64::INFINITY);
        assert!(result.is_err());
        let result = Token::new("hello", f64::NEG_INFINITY);
        assert!(result.is_err());
    }

    #[test]
    fn token_accepts_empty_text() {
        let token = Token::new("", -1.23).unwrap();
        assert!(token.is_empty());
        assert_eq!(token.text(), "");
    }

    #[test]
    fn token_rejects_overly_long_text() {
        let text = "a".repeat(65_537);
        let result = Token::new(text, -0.5);
        assert!(result.is_err());
    }
}
