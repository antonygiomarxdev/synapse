### Task 2.2: Consensus Domain — Vote + Audit

**Files:**
- Create: `synapse-core/src/swarm/consensus.rs`
- Modify: `synapse-core/src/shared/domain_error.rs` — add consensus variants

**Interfaces:**
- Consumes: `Token`
- Produces: `ConsensusResult { request_id: Uuid, consensus_tokens: Vec<Token>, divergent_nodes: Vec<NodeId>, votes_by_token: HashMap<String, u32> }`
- Produces: `ConsensusError` (domain error variant)
- Produces: `vote(node_outputs: &[NodeOutput], quorum: usize) -> Result<ConsensusResult, DomainError>`
- Produces: `audit(reference: &[Token], candidate: &[Token], tolerance: f64) -> bool`

- [ ] **Step 1: Add NodeOutput and ConsensusError types**

Modify `synapse-core/src/shared/domain_error.rs` to add:

```rust
#[error("invalid consensus quorum: {quorum} for swarm_size {swarm_size}")]
InvalidConsensusQuorum { quorum: usize, swarm_size: usize },

#[error("no consensus reached at token index {token_index}")]
NoConsensus { token_index: usize },
```

Add tests:

```rust
#[test]
fn invalid_consensus_quorum_display() {
    let err = DomainError::InvalidConsensusQuorum { quorum: 0, swarm_size: 5 };
    assert_eq!(err.to_string(), "invalid consensus quorum: 0 for swarm_size 5");
}

#[test]
fn no_consensus_display() {
    let err = DomainError::NoConsensus { token_index: 7 };
    assert_eq!(err.to_string(), "no consensus reached at token index 7");
}
```

Run: `cargo test shared::domain_error -p synapse-core`
Expected: PASS.

- [ ] **Step 2: Write failing consensus tests**

Create `synapse-core/src/swarm/consensus.rs` with this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::NodeId;
    use crate::model::ModelId;
    use crate::swarm::token::Token;

    fn node(id: u8) -> NodeId {
        let bytes = [id; 32];
        NodeId::from_public_key(&bytes)
    }

    fn token(text: &str) -> Token {
        Token::new(text, -0.5).unwrap()
    }

    #[test]
    fn majority_3_of_5_selects_consensus_token() {
        let nodes = vec![node(1), node(2), node(3), node(4), node(5)];
        let outputs: Vec<NodeOutput> = nodes
            .into_iter()
            .enumerate()
            .map(|(i, node_id)| {
                let t = if i < 3 {
                    token("def fibo")
                } else {
                    token("def fib")
                };
                NodeOutput {
                    node_id,
                    tokens: vec![t],
                }
            })
            .collect();
        let result = vote(&outputs, 3).unwrap();
        assert_eq!(result.consensus_tokens.len(), 1);
        assert_eq!(result.consensus_tokens[0].text(), "def fibo");
        assert_eq!(result.divergent_nodes.len(), 2);
    }

    #[test]
    fn audit_detects_log_prob_divergence() {
        let a = Token::new("x", -1.0).unwrap();
        let b = Token::new("x", -1.1).unwrap();
        let reference = vec![a.clone()];
        let candidate = vec![b];
        assert!(!audit(&reference, &candidate, 0.05));
        assert!(audit(&reference, &candidate, 0.2));
    }
}
```

Run: `cargo test swarm::consensus::tests::majority_3_of_5_selects_consensus_token -p synapse-core`
Expected: FAIL with `NodeOutput`, `vote`, `audit` not found.

- [ ] **Step 3: Implement consensus pure functions**

Add the implementation to `synapse-core/src/swarm/consensus.rs`:

```rust
use crate::identity::NodeId;
use crate::shared::DomainError;
use crate::swarm::token::Token;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Output from a single node for consensus comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeOutput {
    pub node_id: NodeId,
    pub tokens: Vec<Token>,
}

/// Result of running ensemble consensus over a swarm of nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub request_id: Uuid,
    pub consensus_tokens: Vec<Token>,
    pub divergent_nodes: Vec<NodeId>,
    pub votes_by_token: HashMap<String, u32>,
}

/// Token-by-token majority vote over node outputs.
///
/// `quorum` is the minimum number of nodes that must agree on a token
/// text for it to be accepted. If no token reaches the quorum at any
/// position, returns [`DomainError::NoConsensus`].
///
/// Divergent nodes are nodes whose output token at the chosen position
/// does not match the consensus token.
pub fn vote(
    request_id: Uuid,
    node_outputs: &[NodeOutput],
    quorum: usize,
) -> Result<ConsensusResult, DomainError> {
    if node_outputs.is_empty() {
        return Err(DomainError::InvalidConsensusQuorum {
            quorum,
            swarm_size: 0,
        });
    }
    if quorum == 0 || quorum > node_outputs.len() {
        return Err(DomainError::InvalidConsensusQuorum {
            quorum,
            swarm_size: node_outputs.len(),
        });
    }

    let max_len = node_outputs.iter().map(|o| o.tokens.len()).max().unwrap_or(0);
    let mut consensus_tokens = Vec::with_capacity(max_len);
    let mut divergent_nodes: Vec<NodeId> = Vec::new();
    let mut votes_by_token = HashMap::new();

    for token_index in 0..max_len {
        let mut counts: HashMap<String, Vec<NodeId>> = HashMap::new();
        for output in node_outputs {
            if let Some(token) = output.tokens.get(token_index) {
                counts
                    .entry(token.text().to_string())
                    .or_default()
                    .push(output.node_id);
            }
        }

        let mut best: Option<(String, Vec<NodeId>)> = None;
        for (text, nodes) in counts {
            if nodes.len() >= quorum {
                match &best {
                    Some((_, best_nodes)) if nodes.len() > best_nodes.len() => {
                        best = Some((text, nodes));
                    }
                    None => best = Some((text, nodes)),
                    _ => {}
                }
            }
        }

        let Some((best_text, best_nodes)) = best else {
            return Err(DomainError::NoConsensus { token_index });
        };

        *votes_by_token.entry(best_text.clone()).or_insert(0) += 1;

        // Use the token from the first agreeing node as the canonical token.
        let canonical_node = best_nodes[0];
        let canonical_token = node_outputs
            .iter()
            .find(|o| o.node_id == canonical_node)
            .and_then(|o| o.tokens.get(token_index))
            .cloned()
            .expect("canonical node must have a token at this index");
        consensus_tokens.push(canonical_token);

        for output in node_outputs {
            if !best_nodes.contains(&output.node_id) {
                if !divergent_nodes.contains(&output.node_id) {
                    divergent_nodes.push(output.node_id);
                }
            }
        }
    }

    Ok(ConsensusResult {
        request_id,
        consensus_tokens,
        divergent_nodes,
        votes_by_token,
    })
}

/// Statistical audit: compares two token sequences produced from the
/// same seed. Returns `true` if every token text matches and every
/// log_prob difference is within `tolerance` (absolute).
pub fn audit(reference: &[Token], candidate: &[Token], tolerance: f64) -> bool {
    if reference.len() != candidate.len() {
        return false;
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return false;
    }
    reference.iter().zip(candidate.iter()).all(|(a, b)| {
        a.text() == b.text() && (a.log_prob() - b.log_prob()).abs() <= tolerance
    })
}
```

- [ ] **Step 4: Add property tests with proptest**

Append to `synapse-core/src/swarm/consensus.rs` inside `#[cfg(test)] mod tests`:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn audit_is_reflexive(tokens in prop::collection::vec(any_token_strategy(), 0..=32)) {
        assert!(audit(&tokens, &tokens, 0.0));
    }

    #[test]
    fn consensus_with_unanimity_returns_all_tokens(
        tokens in prop::collection::vec(any_token_strategy(), 0..=16),
    ) {
        let node_ids = vec![node(1), node(2), node(3)];
        let outputs: Vec<NodeOutput> = node_ids
            .into_iter()
            .map(|node_id| NodeOutput {
                node_id,
                tokens: tokens.clone(),
            })
            .collect();
        let result = vote(Uuid::new_v4(), &outputs, 2).unwrap();
        assert_eq!(result.consensus_tokens.len(), tokens.len());
        prop_assert!(result.divergent_nodes.is_empty());
    }
}

fn any_token_strategy() -> impl Strategy<Value = Token> {
    "[a-zA-Z0-9]{0,32}".prop_map(|text| Token::new(text, -1.0).unwrap())
}
```

Run: `cargo test swarm::consensus -p synapse-core`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add synapse-core/src/shared/domain_error.rs synapse-core/src/swarm/consensus.rs
git commit -m "feat(swarm): add token-level consensus vote and audit functions"
```

---

