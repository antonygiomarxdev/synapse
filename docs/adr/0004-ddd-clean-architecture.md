# Domain-Driven Design with Clean Architecture

The codebase follows a strict layered architecture to manage complexity and enable reliable AI agent edits. Every module follows this pattern:

```
Domain (pure Rust, zero I/O) → Ports (traits) → Infrastructure (adapters) → Presentation (axum)
```

**Domain layer** has zero external dependencies. No `ed25519-dalek`, no `libp2p`, no `axum`, no I/O. Domain types are plain Rust structs/enums. Exception: `sha2` is allowed because it's pure computation with no I/O side effects.

**Ports** are traits defined in domain modules: `KeySigner`, `IdentityStore`, `InferenceEngine`, `StakeContract`. These are the I/O boundaries — infrastructure implements them, domain depends only on the trait.

**Infrastructure adapters** live in `infrastructure/` subdirectories and implement exactly one port. `Ed25519Signer` implements `KeySigner`. Future: `DhtIdentityStore` implements `IdentityStore`.

**Presentation (axum handlers)** never imports infrastructure directly — everything goes through trait objects.

**Why not a simpler architecture?** The protocol spans P2P networking, crypto, economic incentives, ML runtime, and smart contracts. Without strict boundaries, the codebase becomes tightly coupled and impossible to test without spinning up a full swarm. With Clean Architecture, domain logic can be tested in isolation — pure functions, no mocks, no I/O.

AI agents benefit enormously: they can edit domain files without understanding infrastructure or presentation, and vice versa. Adding a new ML runtime requires only a new `infrastructure/` adapter — zero domain changes.
