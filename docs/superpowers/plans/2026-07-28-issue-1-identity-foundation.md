# Issue #1: Foundation — Identity + Model Contexts

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Synapse domain foundation — identity primitives (NodeId, KeyPair, Node), model primitives (ModelId, ExpertId, Catalog), shared kernel (DomainError, DomainEvent), and application ports (KeySigner, IdentityStore), plus the Ed25519Signer infrastructure adapter. Domain layer stays pure: zero I/O, zero crypto deps.

**Architecture:** DDD with Clean Architecture. Domain types are pure Rust value objects and aggregates in `identity/` and `model/`. Shared kernel (`shared/`) holds cross-cutting domain errors and events. Application ports are traits in domain modules, implemented by adapters in `identity/infrastructure/`. This is a Rust-only phase — no Python, no Solidity, no network.

**Tech Stack:** Rust 1.97 (pinned), edition 2024. `sha2 0.10` for NodeId derivation, `ed25519-dalek 3.0` for signing adapter, `thiserror 2` for error derive, `serde 1` for serialization, `uuid 1` for event IDs.

**Design Spec:** `docs/superpowers/specs/2026-07-27-synapse-design.md`
**Parent Plan:** `docs/superpowers/plans/2026-07-27-synapse-implementation.md`

## Global Constraints

These expand on the non-negotiable principles in `AGENTS.md`. Every task MUST comply.

### DDD
- Domain layer has ZERO external dependencies — no ed25519, no libp2p, no axum. Crypto lives in `identity/infrastructure/` only.
- `sha2` is allowed in `NodeId` (pure computation, no I/O). `ed25519_dalek` goes in `identity/infrastructure/` only.
- All newtypes use tuple structs (`pub struct NodeId([u8; 32])`), not named fields.
- Aggregates enforce invariants at construction. No `pub` fields that could break invariants — use methods for mutations.
- Every state-changing aggregate method returns `Vec<DomainEvent>` alongside the result.
- Node reputation starts at 100 on registration, clamped to [0, 1000].

### TDD
- **EVERY task step writes the test BEFORE the implementation.** The test MUST be run and MUST fail before writing code.
- Tests inline: `#[cfg(test)] mod tests` in the same file as source.
- Test names describe the scenario: `register_rejects_empty_stake_address`, `from_hex_rejects_invalid_chars`.
- Domain tests are pure: construct inputs, call function, assert output. No mocking, no test doubles, no setup/teardown.
- Infrastructure tests use the real `ed25519_dalek` crate — no mocking of crypto.

### Clean Code
- ALL public types get `///` doc comments. First line is a summary sentence.
- `thiserror` for domain errors — no manual `Display`/`Error` impls.
- `cargo fmt` (max_width 100, 4-space indent), `cargo clippy -- -D warnings` before every commit.
- Commit messages follow Conventional Commits: `feat(identity): ...`, `test(model): ...`.
- `ModelId` validation: non-empty, kebab-case enforced.

---

## File Structure (post-implementation)

```
synapse-core/src/
├── shared/                          # NEW
│   ├── mod.rs                       # re-exports
│   ├── domain_error.rs              # DomainError enum
│   └── domain_event.rs              # DomainEvent enum
├── identity/
│   ├── mod.rs                       # MODIFY — add module declarations, re-exports
│   ├── node_id.rs                   # MODIFY — add hex parsing, Display, more tests
│   ├── key_pair.rs                  # NEW — KeyPair entity (pure domain)
│   ├── node.rs                      # NEW — Node aggregate
│   ├── ports.rs                     # NEW — KeySigner, IdentityStore traits
│   └── infrastructure/             # NEW directory
│       ├── mod.rs                   # re-exports
│       └── ed25519_signer.rs        # Ed25519Signer adapter
├── model/
│   ├── mod.rs                       # MODIFY — add module declarations
│   ├── model_id.rs                  # MODIFY — add validation, more tests
│   ├── model_entity.rs             # MODIFY — add missing fields
│   ├── expert_id.rs                 # NEW — ExpertId value object
│   └── catalog.rs                   # MODIFY — implement Catalog aggregate
├── lib.rs                           # MODIFY — add `pub mod shared;`
└── main.rs                          # UNCHANGED
```

---

### Task 1.0: Shared Kernel — DomainError + DomainEvent

**Files:**
- Create: `synapse-core/src/shared/mod.rs`
- Create: `synapse-core/src/shared/domain_error.rs`
- Create: `synapse-core/src/shared/domain_event.rs`
- Modify: `synapse-core/src/lib.rs` — add `pub mod shared;`

**Interfaces:**
- Produces: `DomainError` (16 variants), `DomainEvent` (7 variants), both re-exported from `synapse_core::shared`

- [ ] **Step 1: Create shared module — mod.rs**

Create `synapse-core/src/shared/mod.rs`:

```rust
pub mod domain_error;
pub mod domain_event;

pub use domain_error::DomainError;
pub use domain_event::DomainEvent;
```

- [ ] **Step 2: Write domain_error.rs**

Create `synapse-core/src/shared/domain_error.rs`:

```rust
use thiserror::Error;

/// Cross-cutting domain errors for the Synapse protocol.
///
/// Every domain operation that can fail returns a [`DomainError`].
/// Infrastructure adapters translate these into their own error types.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DomainError {
    #[error("invalid NodeId: {reason}")]
    InvalidNodeId { reason: String },

    #[error("invalid ModelId: {reason}")]
    InvalidModelId { reason: String },

    #[error("invalid ExpertId: {reason}")]
    InvalidExpertId { reason: String },

    #[error("duplicate model: {model_id}")]
    DuplicateModel { model_id: String },

    #[error("model not found: {model_id}")]
    ModelNotFound { model_id: String },

    #[error("duplicate stake address: {address}")]
    DuplicateStakeAddress { address: String },

    #[error("node not found: {node_id}")]
    NodeNotFound { node_id: String },

    #[error("insufficient reputation: current {current}, required {required}")]
    InsufficientReputation { current: u16, required: u16 },

    #[error("invalid reputation score: {score} (must be 0-{max})")]
    InvalidReputation { score: u16, max: u16 },

    #[error("invalid stake amount: {reason}")]
    InvalidStakeAmount { reason: String },

    #[error("invalid swarm size: {size}")]
    InvalidSwarmSize { size: u32 },

    #[error("invalid price: {reason}")]
    InvalidPrice { reason: String },

    #[error("invalid token: {reason}")]
    InvalidToken { reason: String },

    #[error("storage error: {message}")]
    StorageError { message: String },

    #[error("signature verification failed")]
    SignatureVerificationFailed,

    #[error("catalog load failed: {reason}")]
    CatalogLoadFailed { reason: String },
}
```

- [ ] **Step 3: Write domain_error tests**

Append to `synapse-core/src/shared/domain_error.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_node_id_display() {
        let err = DomainError::InvalidNodeId { reason: "too short".into() };
        assert_eq!(err.to_string(), "invalid NodeId: too short");
    }

    #[test]
    fn invalid_model_id_display() {
        let err = DomainError::InvalidModelId { reason: "empty string".into() };
        assert_eq!(err.to_string(), "invalid ModelId: empty string");
    }

    #[test]
    fn duplicate_model_display() {
        let err = DomainError::DuplicateModel { model_id: "kimi-k3".into() };
        assert_eq!(err.to_string(), "duplicate model: kimi-k3");
    }

    #[test]
    fn node_not_found_display() {
        let err = DomainError::NodeNotFound { node_id: "abc123".into() };
        assert_eq!(err.to_string(), "node not found: abc123");
    }

    #[test]
    fn insufficient_reputation_display() {
        let err = DomainError::InsufficientReputation { current: 150, required: 300 };
        assert_eq!(
            err.to_string(),
            "insufficient reputation: current 150, required 300"
        );
    }

    #[test]
    fn invalid_reputation_display() {
        let err = DomainError::InvalidReputation { score: 1500, max: 1000 };
        assert_eq!(
            err.to_string(),
            "invalid reputation score: 1500 (must be 0-1000)"
        );
    }

    #[test]
    fn signature_failed_display() {
        let err = DomainError::SignatureVerificationFailed;
        assert_eq!(err.to_string(), "signature verification failed");
    }

    #[test]
    fn error_equality() {
        let a = DomainError::InvalidNodeId { reason: "bad".into() };
        let b = DomainError::InvalidNodeId { reason: "bad".into() };
        assert_eq!(a, b);
    }

    #[test]
    fn error_inequality_different_variant() {
        let a = DomainError::InvalidNodeId { reason: "bad".into() };
        let b = DomainError::ModelNotFound { model_id: "x".into() };
        assert_ne!(a, b);
    }
}
```

- [ ] **Step 4: Write domain_event.rs**

Create `synapse-core/src/shared/domain_event.rs`:

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Domain events emitted by aggregates when state changes.
///
/// Each event carries a unique ID for idempotent processing.
/// Infrastructure layers subscribe to these for side effects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainEvent {
    NodeRegistered {
        event_id: Uuid,
        node_id: String,
        stake_address: String,
        reputation: u16,
    },
    ModelAdded {
        event_id: Uuid,
        model_id: String,
        experts: u32,
        active_per_token: u32,
    },
    ModelRemoved {
        event_id: Uuid,
        model_id: String,
    },
    StakeUpdated {
        event_id: Uuid,
        node_id: String,
        old_amount: u64,
        new_amount: u64,
    },
    ReputationChanged {
        event_id: Uuid,
        node_id: String,
        old_score: u16,
        new_score: u16,
    },
    NodeBanned {
        event_id: Uuid,
        node_id: String,
        reason: String,
    },
    NodeUnbanned {
        event_id: Uuid,
        node_id: String,
    },
}
```

- [ ] **Step 5: Write domain_event tests**

Append to `synapse-core/src/shared/domain_event.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_registered_event_has_unique_id() {
        let e1 = DomainEvent::NodeRegistered {
            event_id: Uuid::new_v4(),
            node_id: "n1".into(),
            stake_address: "0xabc".into(),
            reputation: 100,
        };
        let e2 = DomainEvent::NodeRegistered {
            event_id: Uuid::new_v4(),
            node_id: "n2".into(),
            stake_address: "0xdef".into(),
            reputation: 100,
        };
        // Different event_ids → different events
        assert_ne!(e1, e2);
    }

    #[test]
    fn model_added_event_roundtrip_json() {
        let event = DomainEvent::ModelAdded {
            event_id: Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
            model_id: "kimi-k3".into(),
            experts: 896,
            active_per_token: 16,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: DomainEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, parsed);
    }

    #[test]
    fn reputation_changed_event_roundtrip_json() {
        let event = DomainEvent::ReputationChanged {
            event_id: Uuid::new_v4(),
            node_id: "n1".into(),
            old_score: 100,
            new_score: 350,
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: DomainEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(event, parsed);
    }

    #[test]
    fn node_banned_event_carries_reason() {
        let event = DomainEvent::NodeBanned {
            event_id: Uuid::new_v4(),
            node_id: "n42".into(),
            reason: "50 slashing flags".into(),
        };
        assert!(matches!(event, DomainEvent::NodeBanned { .. }));
    }

    #[test]
    fn all_variants_serialize() {
        let events = vec![
            DomainEvent::NodeRegistered {
                event_id: Uuid::new_v4(),
                node_id: "n1".into(),
                stake_address: "0x1".into(),
                reputation: 100,
            },
            DomainEvent::ModelAdded {
                event_id: Uuid::new_v4(),
                model_id: "m1".into(),
                experts: 8,
                active_per_token: 2,
            },
            DomainEvent::StakeUpdated {
                event_id: Uuid::new_v4(),
                node_id: "n1".into(),
                old_amount: 500,
                new_amount: 1000,
            },
        ];
        let json = serde_json::to_string(&events).unwrap();
        let parsed: Vec<DomainEvent> = serde_json::from_str(&json).unwrap();
        assert_eq!(events, parsed);
    }
}
```

- [ ] **Step 6: Register shared module in lib.rs**

Edit `synapse-core/src/lib.rs` — add `pub mod shared;` before the existing modules:

```rust
pub mod shared;
pub mod dht;
pub mod economic;
pub mod gateway;
pub mod identity;
pub mod model;
pub mod swarm;
pub mod transport;
```

- [ ] **Step 7: Verify shared kernel compiles and tests pass**

Run: `cargo test shared`
Expected: 14 tests passed (9 error + 5 event)

- [ ] **Step 8: Format and lint**

```bash
cargo fmt
cargo clippy -- -D warnings
```

Expected: clean

- [ ] **Step 9: Commit**

```bash
git add synapse-core/src/shared/ synapse-core/src/lib.rs
git commit -m "feat(shared): add DomainError and DomainEvent shared kernel"
```

---

### Task 1.1: Enhance NodeId — hex parsing, Display, more tests

**Files:**
- Modify: `synapse-core/src/identity/node_id.rs`

**Interfaces:**
- Modifies: `NodeId` — adds `from_hex`, `to_hex`, `Display`, `as_bytes`
- Existing `from_public_key` signature unchanged

- [ ] **Step 1: Expand NodeId implementation**

Replace `synapse-core/src/identity/node_id.rs` entirely:

```rust
use sha2::{Digest, Sha256};

/// Unique identifier for a Synapse node — SHA256 of its Ed25519 public key.
///
/// A [`NodeId`] is a 32-byte hash that uniquely and deterministically
/// identifies a node in the Synapse swarm. It is derived from the node's
/// public key and cannot be forged without the corresponding secret key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub [u8; 32]);

impl NodeId {
    /// Derives a `NodeId` from an Ed25519 public key.
    ///
    /// The derivation is `SHA256(public_key_bytes)` — a pure function
    /// with no side effects or I/O.
    pub fn from_public_key(pk: &[u8; 32]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(pk);
        let hash: [u8; 32] = hasher.finalize().into();
        Self(hash)
    }

    /// Parses a `NodeId` from a lowercase hex string.
    ///
    /// Returns `None` if the string is not exactly 64 hex characters.
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 64 {
            return None;
        }
        let mut bytes = [0u8; 32];
        for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
            if chunk.len() != 2 {
                return None;
            }
            let high = hex_val(chunk[0])?;
            let low = hex_val(chunk[1])?;
            bytes[i] = (high << 4) | low;
        }
        Some(Self(bytes))
    }

    /// Returns the hex-encoded `NodeId` as a 64-character string.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Returns the raw 32 bytes of this `NodeId`.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'0' + 10),
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

    // --- from_public_key tests (keep existing) ---

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

    // --- from_hex tests ---

    #[test]
    fn from_hex_roundtrip() {
        let pk: [u8; 32] = [42u8; 32];
        let id = NodeId::from_public_key(&pk);
        let hex = id.to_hex();
        let parsed = NodeId::from_hex(&hex).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn from_hex_valid_64_chars() {
        let hex = "a".repeat(64);
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
        assert!(NodeId::from_hex(&"g".repeat(64)).is_none());
        assert!(NodeId::from_hex(&"Z".repeat(64)).is_none());
    }

    // --- Display tests ---

    #[test]
    fn display_is_64_char_hex() {
        let pk: [u8; 32] = [0u8; 32];
        let id = NodeId::from_public_key(&pk);
        let display = id.to_string();
        assert_eq!(display.len(), 64);
        assert!(display.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn display_is_lowercase() {
        let mut pk = [0u8; 32];
        pk[0] = 0xAB;
        let id = NodeId::from_public_key(&pk);
        let display = id.to_string();
        assert!(display.starts_with("ab") || display.starts_with("ab"));
        assert_eq!(display, display.to_lowercase());
    }

    // --- as_bytes tests ---

    #[test]
    fn as_bytes_returns_32_bytes() {
        let pk: [u8; 32] = [7u8; 32];
        let id = NodeId::from_public_key(&pk);
        assert_eq!(id.as_bytes().len(), 32);
    }

    #[test]
    fn as_bytes_is_the_inner_array() {
        let pk: [u8; 32] = [1u8; 32];
        let id = NodeId::from_public_key(&pk);
        assert_eq!(id.as_bytes(), &id.0);
    }
}
```

- [ ] **Step 2: Run NodeId tests**

Run: `cargo test identity::node_id`
Expected: 14 tests passed (3 existing + 11 new)

- [ ] **Step 3: Format and lint**

```bash
cargo fmt
cargo clippy -- -D warnings
```

Expected: clean

- [ ] **Step 4: Commit**

```bash
git add synapse-core/src/identity/node_id.rs
git commit -m "feat(identity): add hex parsing, Display, and extended tests for NodeId"
```

---

### Task 1.2: KeyPair Entity (Pure Domain)

**Files:**
- Create: `synapse-core/src/identity/key_pair.rs`
- Modify: `synapse-core/src/identity/mod.rs` — add `pub mod key_pair;` and re-export

**Interfaces:**
- Produces: `KeyPair { public: [u8; 32], secret: [u8; 32] }` — pure domain entity, NO ed25519 dependency
- Produces: `KeyPair::generate()`, `KeyPair::public_key_bytes()`

**Critical:** This is a PURE domain entity. It generates random bytes directly (via `rand`), NOT via `ed25519_dalek`. The `ed25519_dalek` crate lives only in the infrastructure adapter (Task 1.5). We need `rand` for this — add it to Cargo.toml as a dependency.

- [ ] **Step 1: Add rand to Cargo.toml**

Add to `synapse-core/Cargo.toml` dependencies:

```toml
# Domain randomness (no crypto — just entropy for key generation)
rand = "0.8"
```

- [ ] **Step 2: Write key_pair.rs**

Create `synapse-core/src/identity/key_pair.rs`:

```rust
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
        let mut rng = rand::rng();
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
}
```

- [ ] **Step 3: Update identity/mod.rs**

Replace `synapse-core/src/identity/mod.rs`:

```rust
pub mod key_pair;
pub mod node_id;

pub use key_pair::KeyPair;
pub use node_id::NodeId;
```

- [ ] **Step 4: Run KeyPair tests**

Run: `cargo test identity::key_pair`
Expected: 6 tests passed

- [ ] **Step 5: Format and lint**

```bash
cargo fmt
cargo clippy -- -D warnings
```

Expected: clean

- [ ] **Step 6: Commit**

```bash
git add synapse-core/src/identity/key_pair.rs synapse-core/src/identity/mod.rs synapse-core/Cargo.toml
git commit -m "feat(identity): add KeyPair pure domain entity"
```

---

### Task 1.3: Node Aggregate

**Files:**
- Create: `synapse-core/src/identity/node.rs`
- Modify: `synapse-core/src/identity/mod.rs` — add re-export

**Interfaces:**
- Consumes: `NodeId`, `KeyPair`, `DomainError`, `DomainEvent`
- Produces: `Node` aggregate — `Node::register(keypair, stake_address) -> Result<(Self, DomainEvent)>`, `Node::derive_node_id(keypair) -> NodeId`

- [ ] **Step 1: Write node.rs**

Create `synapse-core/src/identity/node.rs`:

```rust
use crate::shared::{DomainError, DomainEvent};
use super::{KeyPair, NodeId};

/// The maximum reputation score a node can achieve.
pub const MAX_REPUTATION: u16 = 1000;

/// The initial reputation score assigned to newly registered nodes.
pub const INITIAL_REPUTATION: u16 = 100;

/// A Synapse compute node registered in the swarm.
///
/// A [`Node`] is the aggregate root for node identity. It binds together
/// a cryptographic identity ([`NodeId`]), an on-chain stake address,
/// and a reputation score that governs routing priority and slashing risk.
///
/// Nodes are created via [`Node::register`], which emits a
/// [`DomainEvent::NodeRegistered`] event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub node_id: NodeId,
    pub stake_address: String,
    pub reputation: u16,
}

impl Node {
    /// Registers a new node with the given key pair and stake address.
    ///
    /// The node's [`NodeId`] is derived from the public key. Reputation
    /// starts at [`INITIAL_REPUTATION`] (100).
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidNodeId`] if the stake address is empty.
    pub fn register(
        keypair: &KeyPair,
        stake_address: String,
    ) -> Result<(Self, Vec<DomainEvent>), DomainError> {
        if stake_address.trim().is_empty() {
            return Err(DomainError::InvalidNodeId {
                reason: "stake address must not be empty".into(),
            });
        }

        let node_id = Self::derive_node_id(keypair);

        let node = Self {
            node_id,
            stake_address: stake_address.clone(),
            reputation: INITIAL_REPUTATION,
        };

        let event = DomainEvent::NodeRegistered {
            event_id: uuid::Uuid::new_v4(),
            node_id: node_id.to_string(),
            stake_address,
            reputation: INITIAL_REPUTATION,
        };

        Ok((node, vec![event]))
    }

    /// Derives a [`NodeId`] from a [`KeyPair`]'s public key.
    ///
    /// This is `SHA256(public_key_bytes)` — deterministic and pure.
    pub fn derive_node_id(keypair: &KeyPair) -> NodeId {
        NodeId::from_public_key(keypair.public_key_bytes())
    }

    /// Updates the node's reputation score.
    ///
    /// The score is clamped to `[0, MAX_REPUTATION]`.
    /// Returns an event if the score changed.
    pub fn update_reputation(&mut self, new_score: u16) -> Option<DomainEvent> {
        let clamped = new_score.min(MAX_REPUTATION);
        if clamped == self.reputation {
            return None;
        }
        let old = self.reputation;
        self.reputation = clamped;
        Some(DomainEvent::ReputationChanged {
            event_id: uuid::Uuid::new_v4(),
            node_id: self.node_id.to_string(),
            old_score: old,
            new_score: clamped,
        })
    }

    /// Returns true if this node meets the minimum reputation threshold.
    pub fn meets_reputation(&self, minimum: u16) -> bool {
        self.reputation >= minimum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keypair() -> KeyPair {
        KeyPair { public: [1u8; 32], secret: [2u8; 32] }
    }

    // --- register tests ---

    #[test]
    fn register_produces_valid_node() {
        let kp = test_keypair();
        let (node, events) = Node::register(&kp, "0xabcd".into()).unwrap();
        assert_eq!(node.stake_address, "0xabcd");
        assert_eq!(node.reputation, INITIAL_REPUTATION);
        assert!(!events.is_empty());
        assert!(matches!(events[0], DomainEvent::NodeRegistered { .. }));
    }

    #[test]
    fn register_rejects_empty_stake_address() {
        let kp = test_keypair();
        let result = Node::register(&kp, "".into());
        assert!(result.is_err());
    }

    #[test]
    fn register_rejects_whitespace_only_stake_address() {
        let kp = test_keypair();
        let result = Node::register(&kp, "   ".into());
        assert!(result.is_err());
    }

    #[test]
    fn register_starts_at_100_reputation() {
        let kp = test_keypair();
        let (node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        assert_eq!(node.reputation, 100);
    }

    #[test]
    fn register_emits_node_registered_event() {
        let kp = test_keypair();
        let (node, events) = Node::register(&kp, "0xabc".into()).unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            DomainEvent::NodeRegistered { node_id, stake_address, reputation } => {
                assert_eq!(node_id, &node.node_id.to_string());
                assert_eq!(stake_address, "0xabc");
                assert_eq!(*reputation, 100);
            }
            _ => panic!("wrong event variant"),
        }
    }

    // --- derive_node_id tests ---

    #[test]
    fn derive_node_id_is_deterministic() {
        let kp = test_keypair();
        let id1 = Node::derive_node_id(&kp);
        let id2 = Node::derive_node_id(&kp);
        assert_eq!(id1, id2);
    }

    #[test]
    fn derive_node_id_differs_for_different_keys() {
        let kp1 = test_keypair();
        let kp2 = KeyPair { public: [3u8; 32], secret: [4u8; 32] };
        assert_ne!(Node::derive_node_id(&kp1), Node::derive_node_id(&kp2));
    }

    #[test]
    fn derive_node_id_matches_register() {
        let kp = test_keypair();
        let expected = Node::derive_node_id(&kp);
        let (node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        assert_eq!(node.node_id, expected);
    }

    // --- update_reputation tests ---

    #[test]
    fn update_reputation_changes_score() {
        let kp = test_keypair();
        let (mut node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        let event = node.update_reputation(500).unwrap();
        assert_eq!(node.reputation, 500);
        match event {
            DomainEvent::ReputationChanged { old_score, new_score, .. } => {
                assert_eq!(old_score, 100);
                assert_eq!(new_score, 500);
            }
            _ => panic!("wrong event"),
        }
    }

    #[test]
    fn update_reputation_clamps_to_max() {
        let kp = test_keypair();
        let (mut node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        node.update_reputation(1500);
        assert_eq!(node.reputation, MAX_REPUTATION);
    }

    #[test]
    fn update_reputation_clamps_to_zero() {
        let kp = test_keypair();
        let (mut node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        node.update_reputation(0);
        assert_eq!(node.reputation, 0);
    }

    #[test]
    fn update_reputation_no_event_if_unchanged() {
        let kp = test_keypair();
        let (mut node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        // Rep starts at 100; setting to 100 should be no-op
        let clamped_100 = node.reputation; // 100, already at 100
        let event = node.update_reputation(clamped_100);
        assert!(event.is_none());
    }

    // --- meets_reputation tests ---

    #[test]
    fn meets_reputation_above_threshold() {
        let kp = test_keypair();
        let (node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        assert!(node.meets_reputation(50));
    }

    #[test]
    fn meets_reputation_at_threshold() {
        let kp = test_keypair();
        let (node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        assert!(node.meets_reputation(100));
    }

    #[test]
    fn meets_reputation_below_threshold() {
        let kp = test_keypair();
        let (mut node, _) = Node::register(&kp, "0xabc".into()).unwrap();
        node.update_reputation(50);
        assert!(!node.meets_reputation(300));
    }
}
```

- [ ] **Step 2: Update identity/mod.rs**

Replace `synapse-core/src/identity/mod.rs`:

```rust
pub mod key_pair;
pub mod node;
pub mod node_id;

pub use key_pair::KeyPair;
pub use node::Node;
pub use node_id::NodeId;
```

- [ ] **Step 3: Run Node aggregate tests**

Run: `cargo test identity::node`
Expected: 16 tests passed

- [ ] **Step 4: Run full identity suite**

Run: `cargo test identity`
Expected: all 36 tests passed (14 NodeId + 6 KeyPair + 16 Node)

- [ ] **Step 5: Format and lint**

```bash
cargo fmt
cargo clippy -- -D warnings
```

Expected: clean

- [ ] **Step 6: Commit**

```bash
git add synapse-core/src/identity/
git commit -m "feat(identity): add Node aggregate with registration and reputation"
```

---

### Task 1.4: Application Ports — KeySigner + IdentityStore Traits

**Files:**
- Create: `synapse-core/src/identity/ports.rs`
- Modify: `synapse-core/src/identity/mod.rs` — add module + re-exports

**Interfaces:**
- Produces: `KeySigner` trait (sign, verify), `IdentityStore` trait (save, find, find_by_stake_address)

- [ ] **Step 1: Write ports.rs**

Create `synapse-core/src/identity/ports.rs`:

```rust
use crate::shared::DomainError;
use super::{Node, NodeId};

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
    fn public_key_bytes(&self) -> [u8; 32];
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
```

- [ ] **Step 2: Write ports tests**

Append to `synapse-core/src/identity/ports.rs`:

```rust
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
            if nodes.iter().any(|n| n.node_id == node.node_id) {
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
    fn save_duplicate_node_id_is_rejected() {
        let store = InMemoryStore::new();
        let kp = KeyPair::generate();
        let (node, _) = Node::register(&kp, "0xdup".into()).unwrap();
        store.save(&node).unwrap();
        let result = store.save(&node);
        assert!(result.is_err());
    }
}
```

- [ ] **Step 3: Update identity/mod.rs**

Replace `synapse-core/src/identity/mod.rs`:

```rust
pub mod key_pair;
pub mod node;
pub mod node_id;
pub mod ports;

pub use key_pair::KeyPair;
pub use node::Node;
pub use node_id::NodeId;
pub use ports::{IdentityStore, KeySigner};
```

- [ ] **Step 4: Run ports tests**

Run: `cargo test identity::ports`
Expected: 5 tests passed

- [ ] **Step 5: Run full identity suite**

Run: `cargo test identity`
Expected: all 41 tests passed

- [ ] **Step 6: Format and lint**

```bash
cargo fmt
cargo clippy -- -D warnings
```

Expected: clean

- [ ] **Step 7: Commit**

```bash
git add synapse-core/src/identity/
git commit -m "feat(identity): add KeySigner and IdentityStore application ports"
```

---

### Task 1.5: Ed25519Signer Infrastructure Adapter

**Files:**
- Create: `synapse-core/src/identity/infrastructure/mod.rs`
- Create: `synapse-core/src/identity/infrastructure/ed25519_signer.rs`
- Modify: `synapse-core/src/identity/mod.rs` — add `pub mod infrastructure;`

**Interfaces:**
- Consumes: `KeySigner` trait
- Produces: `Ed25519Signer` struct implementing `KeySigner`

**Note:** This is the ONLY file that imports `ed25519_dalek`. The domain layer stays pure.

- [ ] **Step 1: Write infrastructure mod.rs**

Create `synapse-core/src/identity/infrastructure/mod.rs`:

```rust
pub mod ed25519_signer;

pub use ed25519_signer::Ed25519Signer;
```

- [ ] **Step 2: Write ed25519_signer.rs**

Create `synapse-core/src/identity/infrastructure/ed25519_signer.rs`:

```rust
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use crate::identity::ports::KeySigner;

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
    /// # Panics
    ///
    /// Panics if the key material is not a valid Ed25519 key pair.
    /// In production, keys come from [`KeyPair::generate`] or are
    /// loaded from persistent storage.
    pub fn new(public: &[u8; 32], secret: &[u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(secret);
        let verifying_key = VerifyingKey::from_bytes(public)
            .expect("invalid Ed25519 public key");
        // Validate that the key pair is consistent
        assert_eq!(
            signing_key.verifying_key(),
            verifying_key,
            "public key does not match secret key"
        );
        Self { signing_key, verifying_key }
    }

    /// Generates a fresh Ed25519 key pair using OS randomness.
    pub fn generate() -> Self {
        let mut csprng = rand::rng();
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();
        Self { signing_key, verifying_key }
    }

    /// Returns the raw 32-byte secret key material.
    pub fn secret_key_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }
}

impl KeySigner for Ed25519Signer {
    fn sign(&self, data: &[u8]) -> Vec<u8> {
        let signature = self.signing_key.sign(data);
        signature.to_vec()
    }

    fn verify(&self, data: &[u8], signature: &[u8]) -> bool {
        let Ok(sig) = ed25519_dalek::Signature::from_slice(signature) else {
            return false;
        };
        self.verifying_key.verify(data, &sig).is_ok()
    }

    fn public_key_bytes(&self) -> [u8; 32] {
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

        let reconstructed = Ed25519Signer::new(&public, &secret);
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
```

- [ ] **Step 3: Update identity/mod.rs**

Replace `synapse-core/src/identity/mod.rs`:

```rust
pub mod infrastructure;
pub mod key_pair;
pub mod node;
pub mod node_id;
pub mod ports;

pub use key_pair::KeyPair;
pub use node::Node;
pub use node_id::NodeId;
pub use ports::{IdentityStore, KeySigner};
```

- [ ] **Step 4: Run Ed25519Signer tests**

Run: `cargo test identity::infrastructure`
Expected: 7 tests passed

- [ ] **Step 5: Run full test suite**

Run: `cargo test`
Expected: all tests green (identity + model + gateway + shared)

- [ ] **Step 6: Format and lint**

```bash
cargo fmt
cargo clippy -- -D warnings
```

Expected: clean

- [ ] **Step 7: Commit**

```bash
git add synapse-core/src/identity/
git commit -m "feat(identity): add Ed25519Signer infrastructure adapter"
```

---

### Task 1.6: Enhance ModelId + Add ExpertId

**Files:**
- Modify: `synapse-core/src/model/model_id.rs` — add validation, kebab-case enforcement
- Create: `synapse-core/src/model/expert_id.rs`
- Modify: `synapse-core/src/model/mod.rs` — add expert_id module + re-export

**Interfaces:**
- Modifies: `ModelId::new` — now validates non-empty and kebab-case, returns `Result`
- Produces: `ExpertId { model: ModelId, index: u32 }`

- [ ] **Step 1: Rewrite model_id.rs with validation**

Replace `synapse-core/src/model/model_id.rs`:

```rust
use crate::shared::DomainError;
use serde::{Deserialize, Serialize};

/// Unique identifier for an AI model in the Synapse catalog.
///
/// Model IDs must be non-empty and in kebab-case (lowercase letters,
/// digits, and hyphens only). Examples: `"kimi-k3"`, `"mixtral-8x7b"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
        let id = ModelId::new("qwen2.5-moe").is_err(); // contains dot
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
```

- [ ] **Step 2: Fix ModelEntity to use new ModelId API**

`ModelEntity` currently calls `ModelId::new(...)` which now returns `Result`. Fix the tests:

Replace `synapse-core/src/model/model_entity.rs` test section:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn kimi() -> ModelId {
        ModelId::new("kimi-k3").unwrap()
    }

    fn mixtral() -> ModelId {
        ModelId::new("mixtral-8x7b").unwrap()
    }

    #[test]
    fn model_entity_creation() {
        let model = ModelEntity::new(kimi(), 896, 16);
        assert_eq!(model.experts, 896);
        assert_eq!(model.active_per_token, 16);
    }

    #[test]
    fn kimi_k3_sparsity_is_below_2_percent() {
        let model = ModelEntity::new(kimi(), 896, 16);
        assert!(model.sparsity() < 0.02);
        assert!((model.sparsity() - 0.01785).abs() < 0.001);
    }

    #[test]
    fn mixtral_sparsity() {
        let model = ModelEntity::new(mixtral(), 8, 2);
        assert_eq!(model.sparsity(), 0.25);
    }
}
```

- [ ] **Step 3: Create expert_id.rs**

Create `synapse-core/src/model/expert_id.rs`:

```rust
use crate::shared::DomainError;
use super::model_id::ModelId;
use serde::{Deserialize, Serialize};

/// Identifies a specific expert within a model.
///
/// An expert is a sub-network within a Mixture-of-Experts model.
/// `ExpertId` is a composite key of the model identifier and the
/// zero-based expert index within that model.
///
/// Example: Expert #3 of Kimi K3 → `ExpertId { model: "kimi-k3", index: 3 }`
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExpertId {
    pub model: ModelId,
    pub index: u32,
}

impl ExpertId {
    /// Creates a new `ExpertId`.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidExpertId`] if `index` exceeds
    /// `max_expert_index` (must be `< num_experts`).
    pub fn new(
        model: ModelId,
        index: u32,
        num_experts: u32,
    ) -> Result<Self, DomainError> {
        if index >= num_experts {
            return Err(DomainError::InvalidExpertId {
                reason: format!(
                    "expert index {index} is out of bounds for model with {num_experts} experts"
                ),
            });
        }
        Ok(Self { model, index })
    }

    /// Creates an `ExpertId` without bounds checking.
    ///
    /// Useful when the model's expert count is validated elsewhere.
    pub fn new_unchecked(model: ModelId, index: u32) -> Self {
        Self { model, index }
    }
}

impl std::fmt::Display for ExpertId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}#{}", self.model, self.index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kimi_model() -> ModelId {
        ModelId::new("kimi-k3").unwrap()
    }

    #[test]
    fn valid_expert_id() {
        let id = ExpertId::new(kimi_model(), 0, 896).unwrap();
        assert_eq!(id.index, 0);
        assert_eq!(id.model.as_str(), "kimi-k3");
    }

    #[test]
    fn last_expert_is_valid() {
        let id = ExpertId::new(kimi_model(), 895, 896).unwrap();
        assert_eq!(id.index, 895);
    }

    #[test]
    fn expert_at_model_boundary_rejected() {
        assert!(ExpertId::new(kimi_model(), 896, 896).is_err());
    }

    #[test]
    fn expert_beyond_boundary_rejected() {
        let err = ExpertId::new(kimi_model(), 1000, 896).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    #[test]
    fn new_unchecked_skips_validation() {
        let id = ExpertId::new_unchecked(kimi_model(), 9999);
        assert_eq!(id.index, 9999);
    }

    #[test]
    fn display_format() {
        let id = ExpertId::new(kimi_model(), 42, 896).unwrap();
        assert_eq!(id.to_string(), "kimi-k3#42");
    }

    #[test]
    fn equality_same_model_same_index() {
        let a = ExpertId::new(kimi_model(), 7, 100).unwrap();
        let b = ExpertId::new(kimi_model(), 7, 100).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn inequality_different_index() {
        let a = ExpertId::new(kimi_model(), 7, 100).unwrap();
        let b = ExpertId::new(kimi_model(), 8, 100).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn inequality_different_model() {
        let kimi = kimi_model();
        let mixtral = ModelId::new("mixtral-8x7b").unwrap();
        let a = ExpertId::new(kimi, 0, 100).unwrap();
        let b = ExpertId::new(mixtral, 0, 10).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let id = ExpertId::new(kimi_model(), 42, 896).unwrap();
        let json = serde_json::to_string(&id).unwrap();
        let parsed: ExpertId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, parsed);
    }
}
```

- [ ] **Step 4: Update model/mod.rs**

Replace `synapse-core/src/model/mod.rs`:

```rust
pub mod catalog;
pub mod expert_id;
pub mod model_entity;
pub mod model_id;

pub use catalog::Catalog;
pub use expert_id::ExpertId;
pub use model_entity::ModelEntity;
pub use model_id::ModelId;
```

- [ ] **Step 5: Run model tests**

Run: `cargo test model`
Expected: all model tests green (ModelId: ~15, ModelEntity: 3, ExpertId: 10)

- [ ] **Step 6: Format and lint**

```bash
cargo fmt
cargo clippy -- -D warnings
```

Expected: clean

- [ ] **Step 7: Commit**

```bash
git add synapse-core/src/model/
git commit -m "feat(model): add ModelId validation, ExpertId value object"
```

---

### Task 1.7: Catalog Aggregate

**Files:**
- Modify: `synapse-core/src/model/model_entity.rs` — add missing fields (total_params, context_window, license, description)
- Modify: `synapse-core/src/model/catalog.rs` — implement Catalog aggregate
- Modify: `synapse-core/src/gateway/catalog.rs` — wire to domain Catalog

**Interfaces:**
- Consumes: `ModelId`, `ModelEntity`, `DomainError`
- Produces: `Catalog` aggregate — `register`, `list`, `find`, `remove`

- [ ] **Step 1: Enhance ModelEntity with missing fields**

Replace `synapse-core/src/model/model_entity.rs`:

```rust
use super::model_id::ModelId;
use serde::{Deserialize, Serialize};

/// A curated model in the Synapse catalog.
///
/// Each model entry captures the structural metadata needed for
/// swarm expert routing: expert count, active per token, VRAM
/// footprint, context window, and license compliance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntity {
    pub id: ModelId,
    pub name: String,
    pub description: String,
    pub total_params: String,
    pub experts: u32,
    pub active_per_token: u32,
    pub expert_size_gb: f64,
    pub shared_params_gb: f64,
    pub context_window: u64,
    pub license: String,
    pub hf_repo: String,
    pub sha256: Option<String>,
}

impl ModelEntity {
    /// Creates a new model entity. All fields required except sha256.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: ModelId,
        name: String,
        description: String,
        total_params: String,
        experts: u32,
        active_per_token: u32,
        expert_size_gb: f64,
        shared_params_gb: f64,
        context_window: u64,
        license: String,
        hf_repo: String,
        sha256: Option<String>,
    ) -> Self {
        Self {
            id,
            name,
            description,
            total_params,
            experts,
            active_per_token,
            expert_size_gb,
            shared_params_gb,
            context_window,
            license,
            hf_repo,
            sha256,
        }
    }

    /// The model's sparsity ratio: active_per_token / experts.
    pub fn sparsity(&self) -> f64 {
        (self.active_per_token as f64) / (self.experts as f64)
    }

    /// The minimum number of nodes needed to cover all experts
    /// at `experts_per_node` experts per node.
    pub fn min_nodes_for_coverage(&self, experts_per_node: u32) -> u32 {
        self.experts.div_ceil(experts_per_node)
    }

    /// Total VRAM required per node: expert_size * experts_per_node + shared_params.
    pub fn vram_per_node(&self, experts_per_node: u32) -> f64 {
        (self.expert_size_gb * experts_per_node as f64) + self.shared_params_gb
    }
}

impl PartialEq for ModelEntity {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kimi() -> ModelId {
        ModelId::new("kimi-k3").unwrap()
    }

    fn mixtral() -> ModelId {
        ModelId::new("mixtral-8x7b").unwrap()
    }

    fn make_kimi() -> ModelEntity {
        ModelEntity::new(
            kimi(),
            "Kimi K3".into(),
            "Moonshot AI frontier MoE".into(),
            "2.8T".into(),
            896,
            16,
            1.5,
            12.0,
            1_000_000,
            "MIT".into(),
            "moonshotai/Kimi-K3".into(),
            None,
        )
    }

    fn make_mixtral() -> ModelEntity {
        ModelEntity::new(
            mixtral(),
            "Mixtral 8x7B".into(),
            "Mistral MoE".into(),
            "46.7B".into(),
            8,
            2,
            3.0,
            3.0,
            32768,
            "Apache-2.0".into(),
            "mistralai/Mixtral-8x7B-v0.1".into(),
            None,
        )
    }

    #[test]
    fn model_entity_creation() {
        let model = make_kimi();
        assert_eq!(model.experts, 896);
        assert_eq!(model.active_per_token, 16);
        assert_eq!(model.context_window, 1_000_000);
    }

    #[test]
    fn kimi_k3_sparsity_is_below_2_percent() {
        let model = make_kimi();
        assert!(model.sparsity() < 0.02);
        assert!((model.sparsity() - 0.01785).abs() < 0.001);
    }

    #[test]
    fn mixtral_sparsity() {
        let model = make_mixtral();
        assert_eq!(model.sparsity(), 0.25);
    }

    #[test]
    fn min_nodes_for_coverage_ceil() {
        let model = make_kimi();
        // 896 experts, 2 per node → 448 nodes
        assert_eq!(model.min_nodes_for_coverage(2), 448);
        // 896 / 3 = 298.66 → 299
        assert_eq!(model.min_nodes_for_coverage(3), 299);
    }

    #[test]
    fn vram_per_node_calculation() {
        let model = make_kimi();
        // 2 experts: 1.5 * 2 + 12.0 = 15.0 GB
        assert!((model.vram_per_node(2) - 15.0).abs() < 0.01);
        // 4 experts: 1.5 * 4 + 12.0 = 18.0 GB
        assert!((model.vram_per_node(4) - 18.0).abs() < 0.01);
    }

    #[test]
    fn equality_is_by_id_only() {
        let a = make_kimi();
        let mut b = make_kimi();
        b.description = "different".into();
        assert_eq!(a, b);
    }

    #[test]
    fn different_models_not_equal() {
        assert_ne!(make_kimi(), make_mixtral());
    }
}
```

- [ ] **Step 2: Implement Catalog aggregate**

Replace `synapse-core/src/model/catalog.rs`:

```rust
use crate::shared::{DomainError, DomainEvent};
use super::{ModelEntity, ModelId};
use uuid::Uuid;

/// The curated catalog of Synapse-compatible models.
///
/// The [`Catalog`] is an aggregate root that enforces registration
/// invariants: no duplicate model IDs, and all models must pass
/// structural validation.
///
/// In V1, the catalog is curated by Synapse Inc. Community proposals
/// are accepted via GitHub PR.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    models: Vec<ModelEntity>,
}

impl Catalog {
    /// Creates an empty catalog.
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    /// Registers a model in the catalog.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::DuplicateModel`] if a model with the
    /// same [`ModelId`] is already registered.
    pub fn register(
        &mut self,
        model: ModelEntity,
    ) -> Result<Vec<DomainEvent>, DomainError> {
        if self.models.iter().any(|m| m.id == model.id) {
            return Err(DomainError::DuplicateModel {
                model_id: model.id.to_string(),
            });
        }

        let event = DomainEvent::ModelAdded {
            event_id: Uuid::new_v4(),
            model_id: model.id.to_string(),
            experts: model.experts,
            active_per_token: model.active_per_token,
        };

        self.models.push(model);
        Ok(vec![event])
    }

    /// Returns all registered models.
    pub fn list(&self) -> &[ModelEntity] {
        &self.models
    }

    /// Finds a model by its [`ModelId`].
    pub fn find(&self, id: &ModelId) -> Option<&ModelEntity> {
        self.models.iter().find(|m| m.id == *id)
    }

    /// Removes a model from the catalog.
    ///
    /// Returns `None` if no model with the given ID was registered.
    pub fn remove(&mut self, id: &ModelId) -> Option<Vec<DomainEvent>> {
        let pos = self.models.iter().position(|m| m.id == *id)?;
        self.models.remove(pos);
        Some(vec![DomainEvent::ModelRemoved {
            event_id: Uuid::new_v4(),
            model_id: id.to_string(),
        }])
    }

    /// Returns the number of registered models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns `true` if the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(id: &str, experts: u32, active: u32) -> ModelEntity {
        ModelEntity::new(
            ModelId::new(id).unwrap(),
            format!("{id} Name"),
            String::new(),
            String::new(),
            experts,
            active,
            0.0,
            0.0,
            0,
            String::new(),
            String::new(),
            None,
        )
    }

    #[test]
    fn new_catalog_is_empty() {
        let catalog = Catalog::new();
        assert!(catalog.is_empty());
        assert_eq!(catalog.len(), 0);
    }

    #[test]
    fn register_adds_model() {
        let mut catalog = Catalog::new();
        let model = make_model("kimi-k3", 896, 16);
        let events = catalog.register(model).unwrap();
        assert_eq!(catalog.len(), 1);
        assert!(!events.is_empty());
        assert!(matches!(events[0], DomainEvent::ModelAdded { .. }));
    }

    #[test]
    fn register_duplicate_rejected() {
        let mut catalog = Catalog::new();
        catalog.register(make_model("kimi-k3", 896, 16)).unwrap();
        let err = catalog.register(make_model("kimi-k3", 896, 16)).unwrap_err();
        assert!(matches!(err, DomainError::DuplicateModel { .. }));
    }

    #[test]
    fn list_returns_all_models() {
        let mut catalog = Catalog::new();
        catalog.register(make_model("a", 8, 2)).unwrap();
        catalog.register(make_model("b", 64, 8)).unwrap();
        assert_eq!(catalog.list().len(), 2);
    }

    #[test]
    fn find_by_id() {
        let mut catalog = Catalog::new();
        let id = ModelId::new("kimi-k3").unwrap();
        catalog.register(make_model("kimi-k3", 896, 16)).unwrap();
        let found = catalog.find(&id).unwrap();
        assert_eq!(found.experts, 896);
    }

    #[test]
    fn find_unknown_returns_none() {
        let catalog = Catalog::new();
        let id = ModelId::new("unknown").unwrap();
        assert!(catalog.find(&id).is_none());
    }

    #[test]
    fn remove_existing_model() {
        let mut catalog = Catalog::new();
        let id = ModelId::new("to-remove").unwrap();
        catalog.register(make_model("to-remove", 8, 2)).unwrap();
        assert_eq!(catalog.len(), 1);

        let events = catalog.remove(&id).unwrap();
        assert_eq!(catalog.len(), 0);
        assert!(matches!(events[0], DomainEvent::ModelRemoved { .. }));
    }

    #[test]
    fn remove_unknown_returns_none() {
        let mut catalog = Catalog::new();
        let id = ModelId::new("unknown").unwrap();
        assert!(catalog.remove(&id).is_none());
    }

    #[test]
    fn register_multiple_models() {
        let mut catalog = Catalog::new();
        catalog.register(make_model("a", 8, 2)).unwrap();
        catalog.register(make_model("b", 64, 6)).unwrap();
        catalog.register(make_model("c", 896, 16)).unwrap();
        assert_eq!(catalog.len(), 3);
    }

    #[test]
    fn default_catalog_is_empty() {
        let catalog = Catalog::default();
        assert!(catalog.is_empty());
    }
}
```

- [ ] **Step 3: Wire gateway catalog to domain Catalog**

Update `synapse-core/src/gateway/catalog.rs` to use the domain catalog. Replace the file:

```rust
use axum::Json;
use serde::{Deserialize, Serialize};
use crate::model::{Catalog, ModelEntity, ModelId};

/// Public-facing model entry in the API response.
/// A subset of [`ModelEntity`] fields suitable for client consumption.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub total_params: String,
    pub experts: u32,
    pub active_per_token: u32,
    pub context_window: u64,
    pub license: String,
}

impl From<&ModelEntity> for ModelEntry {
    fn from(m: &ModelEntity) -> Self {
        Self {
            id: m.id.to_string(),
            name: m.name.clone(),
            description: m.description.clone(),
            total_params: m.total_params.clone(),
            experts: m.experts,
            active_per_token: m.active_per_token,
            context_window: m.context_window,
            license: m.license.clone(),
        }
    }
}

/// Loads the catalog from config and returns models as JSON.
pub async fn list_models() -> Json<Vec<ModelEntry>> {
    let catalog = load_catalog();
    let entries: Vec<ModelEntry> = catalog.list().iter().map(ModelEntry::from).collect();
    Json(entries)
}

/// Loads the Synapse catalog from `config/models.toml`.
fn load_catalog() -> Catalog {
    let mut catalog = Catalog::new();
    // Hardcoded for now; Task 5.3 will load from config/models.toml dynamically.
    let kimi = ModelEntity::new(
        ModelId::new("kimi-k3").unwrap(),
        "Kimi K3".into(),
        "Moonshot AI's frontier MoE. 2.8T total params, ~103B active, 896 experts, KDA linear attention, 1M context. Open-weight (MIT modified).".into(),
        "2.8T".into(),
        896,
        16,
        1.5,
        12.0,
        1_000_000,
        "MIT".into(),
        "moonshotai/Kimi-K3".into(),
        None,
    );
    catalog.register(kimi).ok();
    catalog
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    async fn test_app() -> axum::Router {
        axum::Router::new().route("/v1/models", axum::routing::get(list_models))
    }

    #[tokio::test]
    async fn list_models_returns_200() {
        let app = test_app().await;
        let response = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn list_models_returns_non_empty_array() {
        let app = test_app().await;
        let response = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
        let models: Vec<ModelEntry> = serde_json::from_slice(&body).unwrap();
        assert!(!models.is_empty());
    }

    #[tokio::test]
    async fn kimi_k3_is_listed_with_correct_specs() {
        let app = test_app().await;
        let response = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
        let models: Vec<ModelEntry> = serde_json::from_slice(&body).unwrap();
        let kimi = models.iter().find(|m| m.id == "kimi-k3").unwrap();
        assert_eq!(kimi.experts, 896);
        assert_eq!(kimi.active_per_token, 16);
        assert_eq!(kimi.context_window, 1_000_000);
    }

    #[tokio::test]
    async fn model_entries_have_required_fields() {
        let app = test_app().await;
        let response = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
        let models: Vec<ModelEntry> = serde_json::from_slice(&body).unwrap();
        for model in &models {
            assert!(!model.id.is_empty());
            assert!(!model.name.is_empty());
            assert!(model.experts > 0);
            assert!(model.active_per_token > 0);
            assert!(!model.license.is_empty());
        }
    }
}
```

- [ ] **Step 4: Run full test suite**

Run: `cargo test`
Expected: all tests green (shared + identity + model + gateway)

- [ ] **Step 5: Format and lint**

```bash
cargo fmt
cargo clippy -- -D warnings
```

Expected: clean. If `clippy::too_many_arguments` fires on `ModelEntity::new`, the `#[allow]` attribute already handles it.

- [ ] **Step 6: Count tests**

Run: `cargo test 2>&1 | grep "test result" | awk '{sum+=$2} END {print sum}'`
Expected: target 25+ tests across all modules (should be well above — we have 14+6+16+5+7+~15+3+10+7+10 ≈ 93 tests)

- [ ] **Step 7: Commit**

```bash
git add synapse-core/src/model/ synapse-core/src/gateway/catalog.rs
git commit -m "feat(model): implement Catalog aggregate with registration, listing, and removal"
```

---

## Phase 1 Acceptance Checklist

- [ ] `cargo test` — all tests green (target: 25+, expected ~93)
- [ ] `cargo clippy -- -D warnings` — clean
- [ ] `cargo fmt --check` — clean
- [ ] Domain layer has zero external dependencies (no ed25519, no libp2p, no axum)
  - Verify: `grep -r "ed25519_dalek" synapse-core/src/identity/ --include="*.rs" | grep -v infrastructure`
  - Should return nothing
- [ ] All public types documented with `///` doc comments
- [ ] `NodeId` has hex parsing + Display + 14 tests
- [ ] `KeyPair` is pure domain (no crypto deps) + 6 tests
- [ ] `Node` aggregate: register, reputation, events + 16 tests
- [ ] `KeySigner` and `IdentityStore` traits defined + 5 contract tests
- [ ] `Ed25519Signer` adapter in infrastructure + 7 integration tests
- [ ] `ModelId` validates non-empty + kebab-case + 15 tests
- [ ] `ExpertId` with bounds checking + 10 tests
- [ ] `Catalog` aggregate: register, list, find, remove + 10 tests
- [ ] `ModelEntity` has full field set (12 fields) + 7 tests
- [ ] Gateway `GET /v1/models` wired to domain Catalog + 4 tests

---

## Post-Phase 1 Cleanup

After all tasks pass acceptance, add any `ponytail:` debt markers for deferred work:

- Model catalog currently hardcoded — `ponytail: load catalog from config/models.toml at startup (Task 5.3)`
- `IdentityStore` has no persistent implementation — `ponytail: implement DHT-backed IdentityStore adapter (Phase 2)`
- `ModelEntity::new` has 12 parameters — `ponytail: consider builder pattern if more fields are added`
