### Task 2.1: Token Value Object

**Files:**
- Create: `synapse-core/src/swarm/token.rs`
- Modify: `synapse-core/src/swarm/mod.rs` — add `pub mod token;` and re-export

**Interfaces:**
- Produces: `Token { id: Uuid, text: String, log_prob: f64 }`
- Produces: `Token::new(text: impl Into<String>, log_prob: f64) -> Result<Self, DomainError>`
- Produces: `Token::id(&self) -> Uuid`, `Token::text(&self) -> &str`, `Token::log_prob(&self) -> f64`
- Produces: `Token::is_empty(&self) -> bool` (text is empty)

- [ ] **Step 1: Add InvalidToken error variants**

Modify `synapse-core/src/shared/domain_error.rs` to add two variants inside `DomainError`:

```rust
#[error("invalid token log_prob: {value} (must be finite)")]
InvalidTokenLogProb { value: f64 },

#[error("invalid token text: {reason}")]
InvalidTokenText { reason: String },
```

Append corresponding tests to the same file:

```rust
#[test]
fn invalid_token_log_prob_display() {
    let err = DomainError::InvalidTokenLogProb { value: f64::NAN };
    assert_eq!(err.to_string(), "invalid token log_prob: NaN (must be finite)");
}

#[test]
fn invalid_token_text_display() {
    let err = DomainError::InvalidTokenText { reason: "too long".into() };
    assert_eq!(err.to_string(), "invalid token text: too long");
}
```

Run: `cargo test shared::domain_error::tests -p synapse-core`
Expected: PASS after adding the new variants.

- [ ] **Step 2: Write the failing Token test**

Create `synapse-core/src/swarm/token.rs` with this initial test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_rejects_nan_log_prob() {
        let result = Token::new("hello", f64::NAN);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid token log_prob: NaN (must be finite)"
        );
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
```

Run: `cargo test swarm::token::tests::token_rejects_nan_log_prob -p synapse-core`
Expected: FAIL with `Token::new` not found.

- [ ] **Step 3: Implement Token value object**

Add the implementation to `synapse-core/src/swarm/token.rs`:

```rust
use crate::shared::DomainError;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const MAX_TOKEN_TEXT_LEN: usize = 65_536;

/// A single generated token and its model-assigned log-probability.
///
/// Tokens are the atomic unit of consensus. Two tokens are equal if
/// their text and log_prob are equal. The `id` is a UUID for tracing
/// individual tokens through the swarm.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Token {
    id: Uuid,
    text: String,
    log_prob: f64,
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
        Ok(Self {
            id: Uuid::new_v4(),
            text,
            log_prob,
        })
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
```

- [ ] **Step 4: Wire Token into swarm module**

Modify `synapse-core/src/swarm/mod.rs` from:

```rust
pub mod consensus;
pub mod dag;
pub mod speculative;
```

To:

```rust
pub mod consensus;
pub mod dag;
pub mod ports;
pub mod resync;
pub mod speculative;
pub mod token;

pub use token::Token;
```

Run: `cargo test swarm::token -p synapse-core`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add synapse-core/src/shared/domain_error.rs synapse-core/src/swarm/token.rs synapse-core/src/swarm/mod.rs
git commit -m "feat(swarm): add Token value object with log_prob validation"
```

---

