use crate::identity::ED25519_KEY_BYTES;
use crate::identity::ports::KeySigner;
use crate::shared::DomainError;
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};

/// Ed25519 implementation of the [`KeySigner`] application port.
///
/// Wraps `ed25519_dalek` to provide production-grade cryptographic
/// signing and verification. This is the only file in the codebase
/// that depends on `ed25519_dalek` — the domain layer remains pure.
pub struct Ed25519Signer {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
}

impl Ed25519Signer {
    /// Creates a new `Ed25519Signer` from 32-byte public and secret keys.
    ///
    /// # Errors
    ///
    /// Returns `DomainError::SignatureVerificationFailed` if the public key
    /// is not a valid Ed25519 verifying key.
    /// In production, keys come from [`KeyPair::generate`] or are
    /// loaded from persistent storage.
    pub fn new(
        public: &[u8; ED25519_KEY_BYTES],
        secret: &[u8; ED25519_KEY_BYTES],
    ) -> Result<Self, DomainError> {
        let signing_key = SigningKey::from_bytes(secret);
        let verifying_key = VerifyingKey::from_bytes(public)
            .map_err(|_| DomainError::SignatureVerificationFailed)?;
        Ok(Self { signing_key, verifying_key })
    }

    /// Generates a fresh Ed25519 key pair using OS randomness.
    pub fn generate() -> Self {
        use rand::RngCore;
        let mut secret = [0u8; ED25519_KEY_BYTES];
        rand::thread_rng().fill_bytes(&mut secret);
        let signing_key = SigningKey::from_bytes(&secret);
        let verifying_key = signing_key.verifying_key();
        Self { signing_key, verifying_key }
    }

    /// Returns the raw 32-byte secret key material.
    pub fn secret_key_bytes(&self) -> [u8; ED25519_KEY_BYTES] {
        self.signing_key.to_bytes()
    }
}

impl KeySigner for Ed25519Signer {
    fn sign(&self, data: &[u8]) -> Vec<u8> {
        let signature = self.signing_key.sign(data);
        signature.to_bytes().to_vec()
    }

    fn verify(&self, data: &[u8], signature: &[u8]) -> bool {
        let Ok(sig) = ed25519_dalek::Signature::from_slice(signature) else {
            return false;
        };
        self.verifying_key.verify(data, &sig).is_ok()
    }

    fn public_key_bytes(&self) -> [u8; ED25519_KEY_BYTES] {
        self.verifying_key.to_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generate_produces_valid_signer() {
        let signer = Ed25519Signer::generate();
        assert_eq!(signer.public_key_bytes().len(), 32);
        assert_eq!(signer.secret_key_bytes().len(), 32);
    }

    #[test]
    fn sign_and_verify_roundtrip() {
        let signer = Ed25519Signer::generate();
        let message = b"synapse inference request";
        let signature = signer.sign(message);
        assert!(signer.verify(message, &signature));
    }

    #[test]
    fn wrong_signature_fails_verification() {
        let signer = Ed25519Signer::generate();
        let message = b"hello";
        let signature = signer.sign(message);

        // Tamper with one byte
        let mut bad_sig = signature.clone();
        bad_sig[0] = bad_sig[0].wrapping_add(1);
        assert!(!signer.verify(message, &bad_sig));
    }

    #[test]
    fn wrong_message_fails_verification() {
        let signer = Ed25519Signer::generate();
        let signature = signer.sign(b"original");
        assert!(!signer.verify(b"tampered", &signature));
    }

    #[test]
    fn different_signer_fails_verification() {
        let alice = Ed25519Signer::generate();
        let bob = Ed25519Signer::generate();
        let message = b"alice signed this";
        let sig = alice.sign(message);
        assert!(!bob.verify(message, &sig));
    }

    #[test]
    fn new_from_raw_keys_works() {
        let signer = Ed25519Signer::generate();
        let public = signer.public_key_bytes();
        let secret = signer.secret_key_bytes();

        let reconstructed = Ed25519Signer::new(&public, &secret).unwrap();
        let msg = b"test";
        let sig = reconstructed.sign(msg);
        assert!(reconstructed.verify(msg, &sig));
    }

    #[test]
    fn garbage_signature_slice_fails_verify() {
        let signer = Ed25519Signer::generate();
        assert!(!signer.verify(b"data", &[0u8; 3]));
    }
}
