use super::{ED25519_KEY_BYTES, Node, NodeId};
use crate::shared::DomainError;

/// Application port for cryptographic signing and verification.
///
/// Infrastructure adapters implement this trait with specific crypto
/// libraries (e.g., ed25519-dalek). The domain layer depends only on
/// this trait, never on concrete crypto.
pub trait KeySigner: Send + Sync {
    /// Signs `data` and returns the signature bytes.
    fn sign(&self, data: &[u8]) -> Vec<u8>;

    /// Verifies that `signature` is a valid signature over `data`
    /// for this signer's public key.
    fn verify(&self, data: &[u8], signature: &[u8]) -> bool;

    /// Returns the raw 32-byte public key for this signer.
    fn public_key_bytes(&self) -> [u8; ED25519_KEY_BYTES];
}

/// Application port for persisting and retrieving node identities.
///
/// Infrastructure adapters implement this trait with concrete storage
/// (e.g., in-memory, SQLite, DHT). The domain layer depends only on
/// this trait, never on concrete storage.
pub trait IdentityStore: Send + Sync {
    /// Persists a node. Returns an error if storage fails.
    fn save(&self, node: &Node) -> Result<(), DomainError>;

    /// Finds a node by its [`NodeId`].
    fn find(&self, id: &NodeId) -> Option<Node>;

    /// Finds a node by its stake address.
    fn find_by_stake_address(&self, address: &str) -> Option<Node>;

    /// Lists all registered nodes.
    fn list_all(&self) -> Vec<Node>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{KeyPair, Node};
    use std::sync::Mutex;

    /// In-memory IdentityStore for testing — validates the trait contract.
    struct InMemoryStore {
        nodes: Mutex<Vec<Node>>,
    }

    impl InMemoryStore {
        fn new() -> Self {
            Self { nodes: Mutex::new(Vec::new()) }
        }
    }

    impl IdentityStore for InMemoryStore {
        fn save(&self, node: &Node) -> Result<(), DomainError> {
            let mut nodes = self.nodes.lock().unwrap();
            if nodes.iter().any(|n| n.stake_address == node.stake_address) {
                return Err(DomainError::DuplicateStakeAddress {
                    address: node.stake_address.clone(),
                });
            }
            nodes.push(node.clone());
            Ok(())
        }

        fn find(&self, id: &NodeId) -> Option<Node> {
            let nodes = self.nodes.lock().unwrap();
            nodes.iter().find(|n| n.node_id == *id).cloned()
        }

        fn find_by_stake_address(&self, address: &str) -> Option<Node> {
            let nodes = self.nodes.lock().unwrap();
            nodes.iter().find(|n| n.stake_address == address).cloned()
        }

        fn list_all(&self) -> Vec<Node> {
            self.nodes.lock().unwrap().clone()
        }
    }

    #[test]
    fn save_and_find_node() {
        let store = InMemoryStore::new();
        let kp = KeyPair::generate();
        let (node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        let node_id = node.node_id;

        store.save(&node).unwrap();
        let found = store.find(&node_id).unwrap();
        assert_eq!(found.node_id, node_id);
    }

    #[test]
    fn find_returns_none_for_unknown_id() {
        let store = InMemoryStore::new();
        let kp = KeyPair::generate();
        let unknown_id = Node::derive_node_id(&kp);
        assert!(store.find(&unknown_id).is_none());
    }

    #[test]
    fn find_by_stake_address() {
        let store = InMemoryStore::new();
        let kp = KeyPair::generate();
        let (node, _) = Node::register(&kp, "0xunique".into()).unwrap();
        store.save(&node).unwrap();

        let found = store.find_by_stake_address("0xunique").unwrap();
        assert_eq!(found.stake_address, "0xunique");
    }

    #[test]
    fn list_all_returns_all_nodes() {
        let store = InMemoryStore::new();
        let (n1, _) = Node::register(&KeyPair::generate(), "0x1".into()).unwrap();
        let (n2, _) = Node::register(&KeyPair::generate(), "0x2".into()).unwrap();
        store.save(&n1).unwrap();
        store.save(&n2).unwrap();

        let all = store.list_all();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn save_duplicate_stake_address_is_rejected() {
        let store = InMemoryStore::new();
        let kp = KeyPair::generate();
        let (node, _) = Node::register(&kp, "0xdup".into()).unwrap();
        store.save(&node).unwrap();
        let result = store.save(&node);
        assert!(matches!(result, Err(DomainError::DuplicateStakeAddress { .. })));
    }
}
