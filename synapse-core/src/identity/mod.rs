/// Size of an Ed25519 key in bytes (32).
pub const ED25519_KEY_BYTES: usize = 32;

/// Creates a 32-byte array filled with `val` for use in tests.
///
/// Replaces the repetitive `[val; ED25519_KEY_BYTES]` pattern.
#[cfg(test)]
pub(crate) fn test_bytes(val: u8) -> [u8; ED25519_KEY_BYTES] {
    [val; ED25519_KEY_BYTES]
}

pub mod infrastructure;
pub mod key_pair;

pub mod node;
pub mod node_id;
pub mod ports;

pub use key_pair::KeyPair;
pub use node::Node;
pub use node_id::NodeId;
pub use ports::{IdentityStore, KeySigner};
