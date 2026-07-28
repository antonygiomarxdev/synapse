use sha2::{Digest, Sha256};

/// Unique identifier for a Synapse node — SHA256 of its public key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub [u8; 32]);

impl NodeId {
    /// Derives a NodeId from an Ed25519 public key.
    pub fn from_public_key(pk: &[u8; 32]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(pk);
        let hash: [u8; 32] = hasher.finalize().into();
        Self(hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_is_deterministic() {
        let pk: [u8; 32] = [1u8; 32];
        let id1 = NodeId::from_public_key(&pk);
        let id2 = NodeId::from_public_key(&pk);
        assert_eq!(id1, id2);
    }

    #[test]
    fn different_keys_produce_different_ids() {
        let pk1: [u8; 32] = [1u8; 32];
        let pk2: [u8; 32] = [2u8; 32];
        assert_ne!(NodeId::from_public_key(&pk1), NodeId::from_public_key(&pk2));
    }

    #[test]
    fn node_id_is_32_bytes() {
        let pk: [u8; 32] = [0u8; 32];
        let id = NodeId::from_public_key(&pk);
        assert_eq!(id.0.len(), 32);
    }
}
