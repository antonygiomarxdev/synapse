use rand::RngCore;

/// A cryptographic key pair for a Synapse node.
///
/// This is a pure domain entity — it holds raw key material but has
/// no knowledge of specific crypto algorithms. Infrastructure adapters
/// (e.g., [`Ed25519Signer`]) interpret these bytes.
///
/// The public key is 32 bytes; the secret key is 32 bytes.
/// In production, these are Ed25519 keys, but the domain layer
/// treats them opaquely.
#[derive(Debug, Clone)]
pub struct KeyPair {
    pub public: [u8; 32],
    pub secret: [u8; 32],
}

impl KeyPair {
    /// Generates a fresh random key pair.
    ///
    /// Uses OS randomness via `rand`. The domain layer generates keys
    /// directly — it does not delegate to a crypto library.
    /// Infrastructure adapters validate that the key material is usable
    /// for their specific algorithm.
    pub fn generate() -> Self {
        let mut public = [0u8; 32];
        let mut secret = [0u8; 32];
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut public);
        rng.fill_bytes(&mut secret);
        Self { public, secret }
    }

    /// Returns a reference to the 32-byte public key.
    pub fn public_key_bytes(&self) -> &[u8; 32] {
        &self.public
    }

    /// Returns a reference to the 32-byte secret key.
    pub fn secret_key_bytes(&self) -> &[u8; 32] {
        &self.secret
    }
}

impl PartialEq for KeyPair {
    fn eq(&self, other: &Self) -> bool {
        self.public == other.public && self.secret == other.secret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_produces_32_byte_keys() {
        let kp = KeyPair::generate();
        assert_eq!(kp.public.len(), 32);
        assert_eq!(kp.secret.len(), 32);
    }

    #[test]
    fn generate_produces_different_keys() {
        let kp1 = KeyPair::generate();
        let kp2 = KeyPair::generate();
        // Extremely unlikely to collide with 32 random bytes each
        assert_ne!(kp1.public, kp2.public);
        assert_ne!(kp1.secret, kp2.secret);
    }

    #[test]
    fn public_key_bytes_is_32_bytes() {
        let kp = KeyPair::generate();
        assert_eq!(kp.public_key_bytes().len(), 32);
    }

    #[test]
    fn same_keys_are_equal() {
        let kp1 = KeyPair { public: [1u8; 32], secret: [2u8; 32] };
        let kp2 = KeyPair { public: [1u8; 32], secret: [2u8; 32] };
        assert_eq!(kp1, kp2);
    }

    #[test]
    fn different_public_not_equal() {
        let kp1 = KeyPair { public: [1u8; 32], secret: [2u8; 32] };
        let kp2 = KeyPair { public: [3u8; 32], secret: [2u8; 32] };
        assert_ne!(kp1, kp2);
    }

    #[test]
    fn different_secret_not_equal() {
        let kp1 = KeyPair { public: [1u8; 32], secret: [2u8; 32] };
        let kp2 = KeyPair { public: [1u8; 32], secret: [4u8; 32] };
        assert_ne!(kp1, kp2);
    }

    #[test]
    fn secret_key_bytes_returns_actual_secret() {
        let secret = [0xABu8; 32];
        let kp = KeyPair { public: [0u8; 32], secret };
        assert_eq!(*kp.secret_key_bytes(), secret);
        // Verify NOT returning a zero array or any other fixed value
        assert_ne!(*kp.secret_key_bytes(), [0u8; 32]);
    }
}
