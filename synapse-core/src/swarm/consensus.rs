use crate::identity::NodeId;
use crate::shared::DomainError;
use crate::swarm::Token;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use uuid::Uuid;

/// True if `(text, nodes)` should replace `(best_text, best_nodes)` as the
/// consensus winner. Uses a larger vote count first, then lexicographic tie-break.
#[cfg_attr(test, mutants::skip)]
fn is_better_candidate(
    nodes: &[NodeId],
    best_nodes: &[NodeId],
    text: &str,
    best_text: &str,
) -> bool {
    nodes.len() > best_nodes.len() || (nodes.len() == best_nodes.len() && text < best_text)
}

/// True if the tolerance value is invalid for audit comparison.
#[cfg_attr(test, mutants::skip)]
fn is_invalid_tolerance(tolerance: f64) -> bool {
    !tolerance.is_finite() || tolerance < 0.0
}

/// Output from a single node for consensus comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeOutput {
    pub node_id: NodeId,
    pub tokens: Vec<Token>,
}

/// Result of running ensemble consensus over a swarm of nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub request_id: Uuid,
    pub consensus_tokens: Vec<Token>,
    pub divergent_nodes: Vec<NodeId>,
    pub votes_by_token: HashMap<String, u32>,
}

/// Validates vote preconditions: non-empty outputs and valid quorum.
fn validate_vote_inputs(node_outputs: &[NodeOutput], quorum: usize) -> Result<(), DomainError> {
    if node_outputs.is_empty() {
        return Err(DomainError::InvalidConsensusQuorum { quorum, swarm_size: 0 });
    }
    if quorum == 0 || quorum > node_outputs.len() {
        return Err(DomainError::InvalidConsensusQuorum { quorum, swarm_size: node_outputs.len() });
    }
    Ok(())
}

/// Groups node outputs by token text for a single position.
fn tally_token_position(
    node_outputs: &[NodeOutput],
    token_index: usize,
) -> BTreeMap<String, Vec<NodeId>> {
    let mut counts: BTreeMap<String, Vec<NodeId>> = BTreeMap::new();
    for output in node_outputs {
        if let Some(token) = output.tokens.get(token_index) {
            counts.entry(token.text().to_string()).or_default().push(output.node_id);
        }
    }
    counts
}

/// Returns the maximum number of tokens across all node outputs.
fn max_token_length(node_outputs: &[NodeOutput]) -> usize {
    node_outputs.iter().map(|o| o.tokens.len()).max().unwrap_or(0)
}

/// Selects the winning token text and its agreeing nodes.
/// Returns `None` if no text reaches quorum.
fn select_winner(
    counts: BTreeMap<String, Vec<NodeId>>,
    quorum: usize,
) -> Option<(String, Vec<NodeId>)> {
    let mut best: Option<(String, Vec<NodeId>)> = None;
    for (text, nodes) in counts {
        if nodes.len() >= quorum {
            match &best {
                Some((best_text, best_nodes))
                    if is_better_candidate(&nodes, best_nodes, &text, best_text) =>
                {
                    best = Some((text, nodes));
                }
                None => best = Some((text, nodes)),
                _ => {}
            }
        }
    }
    best
}

/// Retrieves the canonical token from the first agreeing node.
fn extract_canonical_token(
    node_outputs: &[NodeOutput],
    canonical_node: NodeId,
    token_index: usize,
) -> Token {
    node_outputs
        .iter()
        .find(|o| o.node_id == canonical_node)
        .and_then(|o| o.tokens.get(token_index))
        .cloned()
        .expect("canonical node must have a token at this index")
}

/// Collects nodes NOT in `best_nodes` that haven't already been flagged.
fn collect_divergent_nodes(
    node_outputs: &[NodeOutput],
    best_nodes: &[NodeId],
    already_divergent: &[NodeId],
) -> Vec<NodeId> {
    node_outputs
        .iter()
        .map(|o| o.node_id)
        .filter(|id| !best_nodes.contains(id) && !already_divergent.contains(id))
        .collect()
}

/// Token-by-token majority vote over node outputs.
///
/// `quorum` is the minimum number of nodes that must agree on a token
/// text for it to be accepted. If no token reaches the quorum at any
/// position, returns [`DomainError::NoConsensus`].
pub fn vote(
    request_id: Uuid,
    node_outputs: &[NodeOutput],
    quorum: usize,
) -> Result<ConsensusResult, DomainError> {
    validate_vote_inputs(node_outputs, quorum)?;

    let max_len = max_token_length(node_outputs);
    let mut consensus_tokens = Vec::with_capacity(max_len);
    let mut divergent_nodes: Vec<NodeId> = Vec::new();
    let mut votes_by_token = HashMap::new();

    for token_index in 0..max_len {
        let counts = tally_token_position(node_outputs, token_index);
        let Some((best_text, best_nodes)) = select_winner(counts, quorum) else {
            return Err(DomainError::NoConsensus { token_index });
        };

        *votes_by_token.entry(best_text.clone()).or_insert(0) += 1;
        let canonical = extract_canonical_token(node_outputs, best_nodes[0], token_index);
        consensus_tokens.push(canonical);

        let new_divergent = collect_divergent_nodes(node_outputs, &best_nodes, &divergent_nodes);
        divergent_nodes.extend(new_divergent);
    }

    Ok(ConsensusResult { request_id, consensus_tokens, divergent_nodes, votes_by_token })
}

/// Statistical audit: compares two token sequences produced from the
/// same seed. Returns `true` if every token text matches and every
/// log_prob difference is within `tolerance` (absolute).
pub fn audit(reference: &[Token], candidate: &[Token], tolerance: f64) -> bool {
    if reference.len() != candidate.len() {
        return false;
    }
    if is_invalid_tolerance(tolerance) {
        return false;
    }
    reference
        .iter()
        .zip(candidate.iter())
        .all(|(a, b)| a.text() == b.text() && (a.log_prob() - b.log_prob()).abs() <= tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::NodeId;
    use uuid::Uuid;

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
                let t = if i < 3 { token("def fibo") } else { token("def fib") };
                NodeOutput { node_id, tokens: vec![t] }
            })
            .collect();
        let result = vote(Uuid::new_v4(), &outputs, 3).unwrap();
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

    #[test]
    fn audit_rejects_different_lengths() {
        let a = Token::new("x", -1.0).unwrap();
        assert!(!audit(&[a.clone()], &[], 0.1));
        assert!(!audit(&[], &[a], 0.1));
    }

    #[test]
    fn audit_rejects_invalid_tolerance() {
        let a = Token::new("x", -1.0).unwrap();
        assert!(!audit(&[a.clone()], &[a.clone()], f64::NAN));
        assert!(!audit(&[a.clone()], &[a], f64::NEG_INFINITY));
    }

    #[test]
    fn vote_rejects_empty_outputs() {
        let err = vote(Uuid::new_v4(), &[], 3).unwrap_err();
        assert!(matches!(err, DomainError::InvalidConsensusQuorum { .. }));
    }

    #[test]
    fn vote_rejects_zero_quorum() {
        let outputs = vec![NodeOutput { node_id: node(1), tokens: vec![token("a")] }];
        let err = vote(Uuid::new_v4(), &outputs, 0).unwrap_err();
        assert!(matches!(err, DomainError::InvalidConsensusQuorum { .. }));
    }

    #[test]
    fn vote_rejects_quorum_exceeding_swarm() {
        let outputs = vec![NodeOutput { node_id: node(1), tokens: vec![token("a")] }];
        let err = vote(Uuid::new_v4(), &outputs, 2).unwrap_err();
        assert!(matches!(err, DomainError::InvalidConsensusQuorum { .. }));
    }

    #[test]
    fn vote_returns_no_consensus_when_no_quorum() {
        let outputs: Vec<NodeOutput> = (1..=3)
            .map(|i| NodeOutput { node_id: node(i), tokens: vec![token(&format!("token{i}"))] })
            .collect();
        let err = vote(Uuid::new_v4(), &outputs, 2).unwrap_err();
        assert!(matches!(err, DomainError::NoConsensus { token_index: 0 }));
    }

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

    #[test]
    fn vote_with_exact_quorum_succeeds() {
        let outputs: Vec<NodeOutput> =
            (1..=3).map(|i| NodeOutput { node_id: node(i), tokens: vec![token("same")] }).collect();
        let result = vote(Uuid::new_v4(), &outputs, 3).unwrap();
        assert_eq!(result.consensus_tokens[0].text(), "same");
    }

    #[test]
    fn vote_counts_votes_by_token_text() {
        // 3 nodes: two say "yes", one says "no"
        let outputs = vec![
            NodeOutput { node_id: node(1), tokens: vec![token("yes"), token("yes")] },
            NodeOutput { node_id: node(2), tokens: vec![token("yes"), token("no")] },
            NodeOutput { node_id: node(3), tokens: vec![token("yes"), token("yes")] },
        ];
        let result = vote(Uuid::new_v4(), &outputs, 2).unwrap();
        assert_eq!(result.consensus_tokens.len(), 2);
        assert_eq!(result.consensus_tokens[0].text(), "yes");
        assert_eq!(result.consensus_tokens[1].text(), "yes");
        // "yes" wins both positions
        assert!(result.votes_by_token.get("no").copied().unwrap_or(0) < 2);
        assert!(
            result.votes_by_token.get("yes").copied().unwrap_or(0) > 0,
            "votes_by_token should track counts"
        );
    }

    #[test]
    fn consensus_uses_correct_canonical_node() {
        let outputs = vec![
            NodeOutput { node_id: node(1), tokens: vec![token("alpha")] },
            NodeOutput { node_id: node(2), tokens: vec![token("alpha")] },
            NodeOutput { node_id: node(3), tokens: vec![token("beta")] },
        ];
        let result = vote(Uuid::new_v4(), &outputs, 2).unwrap();
        assert_eq!(result.consensus_tokens[0].text(), "alpha");
        assert!(result.divergent_nodes.contains(&node(3)));
        assert!(!result.divergent_nodes.contains(&node(1)));
    }

    #[test]
    fn audit_rejects_negative_tolerance() {
        let a = Token::new("x", -1.0).unwrap();
        assert!(!audit(&[a.clone()], &[a], -0.1));
    }

    #[test]
    fn tie_uses_lexicographic_ordering() {
        // 4 nodes: 2 say "beta", 2 say "alpha" -> tie
        let outputs = vec![
            NodeOutput { node_id: node(1), tokens: vec![token("beta")] },
            NodeOutput { node_id: node(2), tokens: vec![token("beta")] },
            NodeOutput { node_id: node(3), tokens: vec![token("alpha")] },
            NodeOutput { node_id: node(4), tokens: vec![token("alpha")] },
        ];
        let result = vote(Uuid::new_v4(), &outputs, 2).unwrap();
        // Deterministic: "alpha" < "beta" -> alpha wins
        assert_eq!(result.consensus_tokens[0].text(), "alpha");
    }

    #[test]
    fn select_winner_picks_highest_votes_not_alpha_first() {
        // "alpha" has 2 votes (reaches quorum=2) but "beta" has 3 votes.
        // BTreeMap iterates "alpha" first. The guard must replace it.
        let outputs = vec![
            NodeOutput { node_id: node(1), tokens: vec![token("alpha")] },
            NodeOutput { node_id: node(2), tokens: vec![token("alpha")] },
            NodeOutput { node_id: node(3), tokens: vec![token("beta")] },
            NodeOutput { node_id: node(4), tokens: vec![token("beta")] },
            NodeOutput { node_id: node(5), tokens: vec![token("beta")] },
        ];
        let result = vote(Uuid::new_v4(), &outputs, 2).unwrap();
        assert_eq!(
            result.consensus_tokens[0].text(),
            "beta",
            "must pick highest votes, not alphabetical first"
        );
    }
}
