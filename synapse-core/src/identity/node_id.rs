use crate::identity::ED25519_KEY_BYTES;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Length of a NodeId hex string (64 hex chars for 32 bytes).
pub const NODE_ID_HEX_LEN: usize = 64;

/// Unique identifier for a Synapse node — SHA256 of its Ed25519 public key.
///
/// A [`NodeId`] is a 32-byte hash that uniquely and deterministically
/// identifies a node in the Synapse swarm. It is derived from the node's
/// public key and cannot be forged without the corresponding secret key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub [u8; ED25519_KEY_BYTES]);

impl NodeId {
    /// Derives a `NodeId` from an Ed25519 public key.
    ///
    /// The derivation is `SHA256(public_key_bytes)` — a pure function
    /// with no side effects or I/O.
    pub fn from_public_key(pk: &[u8; ED25519_KEY_BYTES]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(pk);
        let hash: [u8; ED25519_KEY_BYTES] = hasher.finalize().into();
        Self(hash)
    }

    /// Parses a `NodeId` from a lowercase hex string.
    ///
    /// Returns `None` if the string is not exactly 64 hex characters.
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != NODE_ID_HEX_LEN {
            return None;
        }
        let mut bytes = [0u8; ED25519_KEY_BYTES];
        for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
            if chunk.len() != 2 {
                return None;
            }
            let high = hex_val(chunk[0])?;
            let low = hex_val(chunk[1])?;
            bytes[i] = (high << 4) + low;
        }
        Some(Self(bytes))
    }

    /// Returns the hex-encoded `NodeId` as a 64-character string.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Returns the raw 32 bytes of this `NodeId`.
    pub fn as_bytes(&self) -> &[u8; ED25519_KEY_BYTES] {
        &self.0
    }
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::test_bytes;
    // --- from_public_key tests (keep existing) ---

    #[test]
    fn node_id_is_deterministic() {
        let pk: [u8; ED25519_KEY_BYTES] = test_bytes(1);
        let id1 = NodeId::from_public_key(&pk);
        let id2 = NodeId::from_public_key(&pk);
        assert_eq!(id1, id2);
    }

    #[test]
    fn different_keys_produce_different_ids() {
        let pk1: [u8; ED25519_KEY_BYTES] = test_bytes(1);
        let pk2: [u8; ED25519_KEY_BYTES] = [2u8; ED25519_KEY_BYTES];
        assert_ne!(NodeId::from_public_key(&pk1), NodeId::from_public_key(&pk2));
    }

    #[test]
    fn node_id_is_32_bytes() {
        let pk: [u8; ED25519_KEY_BYTES] = test_bytes(0);
        let id = NodeId::from_public_key(&pk);
        assert_eq!(id.0.len(), 32);
    }

    // --- from_hex tests ---

    #[test]
    fn from_hex_roundtrip() {
        let pk: [u8; ED25519_KEY_BYTES] = test_bytes(42);
        let id = NodeId::from_public_key(&pk);
        let hex = id.to_hex();
        let parsed = NodeId::from_hex(&hex).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn from_hex_valid_64_chars() {
        let hex = "a".repeat(NODE_ID_HEX_LEN);
        let id = NodeId::from_hex(&hex);
        assert!(id.is_some());
    }

    #[test]
    fn from_hex_rejects_short_string() {
        assert!(NodeId::from_hex("abc").is_none());
    }

    #[test]
    fn from_hex_rejects_long_string() {
        assert!(NodeId::from_hex(&"a".repeat(65)).is_none());
    }

    #[test]
    fn from_hex_rejects_empty() {
        assert!(NodeId::from_hex("").is_none());
    }

    #[test]
    fn from_hex_rejects_invalid_chars() {
        assert!(NodeId::from_hex(&"g".repeat(NODE_ID_HEX_LEN)).is_none());
        assert!(NodeId::from_hex(&"Z".repeat(NODE_ID_HEX_LEN)).is_none());
    }

    // --- Display tests ---

    #[test]
    fn display_is_64_char_hex() {
        let pk: [u8; ED25519_KEY_BYTES] = test_bytes(0);
        let id = NodeId::from_public_key(&pk);
        let display = id.to_string();
        assert_eq!(display.len(), NODE_ID_HEX_LEN);
        assert!(display.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn display_is_lowercase() {
        // Construct a NodeId with a byte whose high nibble is 0xA,
        // so we can verify the hex output uses lowercase "a" not uppercase "A".
        let id = NodeId([0xAB; 32]);
        let display = id.to_string();
        // The first hex pair should be "ab" (lowercase) not "AB".
        assert!(display.starts_with("ab"), "expected lowercase 'ab', got {display:?}");
        assert_eq!(display, display.to_lowercase());
    }

    #[test]
    fn from_hex_uses_bitwise_or() {
        // Hex "ff" decodes as (15 << 4) | 15 = 255 with OR.
        // If XOR were used instead: (15 << 4) ^ 15 = 240, which is wrong.
        let id = NodeId::from_hex("ff".repeat(32).as_str()).unwrap();
        assert_eq!(id.0, test_bytes(0xFF));
    }

    // --- as_bytes tests ---

    #[test]
    fn as_bytes_returns_32_bytes() {
        let pk: [u8; ED25519_KEY_BYTES] = test_bytes(7);
        let id = NodeId::from_public_key(&pk);
        assert_eq!(id.as_bytes().len(), 32);
    }

    #[test]
    fn as_bytes_is_the_inner_array() {
        let pk: [u8; ED25519_KEY_BYTES] = test_bytes(1);
        let id = NodeId::from_public_key(&pk);
        assert_eq!(id.as_bytes(), &id.0);
    }
}
