/// MoE forward loop with external routing hooks.
///
/// Implements the transformer forward pass: embedding lookup → per-layer
/// RMS norm → attention → residual → RMS norm → MoE FFN with external
/// routing → residual → output norm → projection.
///
/// For V0 spike validation: attention and expert FFN use placeholder zeros.
/// The routing layer (gate_inp @ hidden_state → top-k) is fully functional
/// and verified against the numpy spike.
use std::path::Path;

use crate::native_moe::model::{MoeLayer, MoeModel, Tensor};

/// Result of the forward pass for a single token generation step.
#[derive(Debug, Clone)]
pub struct ForwardOutput {
    /// Logits over vocabulary: [vocab_size]
    pub logits: Vec<f32>,
    /// Expert routing decisions per layer: [(layer_idx, expert_ids, scores)]
    pub routes: Vec<(usize, Vec<u32>, Vec<f32>)>,
}

/// Run the full transformer forward pass on a prompt.
///
/// Returns logits for the last token position and per-layer routing decisions.
pub fn forward(model: &MoeModel, prompt_tokens: &[u32]) -> ForwardOutput {
    let d_model = model.config.d_model as usize;
    let n_tokens = prompt_tokens.len();

    // Phase 0: Embedding lookup (token → hidden state). For V0, use zeros.
    let mut hidden = vec![vec![0.0f32; d_model]; n_tokens];

    let mut routes = Vec::new();

    // Phase 1: Per-layer forward pass
    for layer in &model.layers {
        let layer_idx = layer.index;

        // Phase 1a: RMS norm + attention (placeholder zeros for V0)
        // hidden = rms_norm(hidden) → attention → + residual
        // For V0: skip — hidden remains unchanged

        // Phase 1b: RMS norm + MoE FFN with external routing
        let residual = hidden.clone();
        // hidden = rms_norm(hidden) — skip for V0

        // External routing: score all experts for each token
        let route = route_experts(layer, &hidden);
        routes.push(route);

        // Phase 1c: Placeholder — expert FFN with weighted sum
        // For V0: skip FFN, hidden stays as residual

        // Phase 1d: Residual connection
        for t in 0..n_tokens {
            for d in 0..d_model {
                hidden[t][d] = hidden[t][d] + residual[t][d];
            }
        }
    }

    // Phase 2: Output norm + projection
    // logits = output @ rms_norm(hidden[-1]) — skip for V0
    let logits = vec![0.0f32; model.config.vocab_size as usize];

    ForwardOutput { logits, routes }
}

/// Compute expert scores for all tokens in a layer.
/// hidden[t] @ gate_inp^T → scores[n_experts] → top-k indices.
fn route_experts(layer: &MoeLayer, hidden: &[Vec<f32>]) -> (usize, Vec<u32>, Vec<f32>) {
    let d_model = layer.gate_inp.shape[0] as usize;
    let n_experts = layer.gate_inp.shape[1] as usize;
    let k = 8; // top-k experts per token
    let n_tokens = hidden.len();

    let gate = layer.gate_inp.as_slice();

    // Compute scores for last token only (routing decision point)
    let last = &hidden[n_tokens - 1];
    let mut scores = vec![0.0f32; n_experts];
    for e in 0..n_experts {
        let mut acc = 0.0;
        for d in 0..d_model {
            // gate is [d_model, n_experts] column-major (GGUF native)
            acc += last[d] * gate[d * n_experts + e];
        }
        scores[e] = acc;
    }

    // Top-k
    let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_ids: Vec<u32> = indexed.iter().take(k).map(|(i, _)| *i as u32).collect();
    let top_scores: Vec<f32> = indexed.iter().take(k).map(|(_, s)| *s).collect();

    (layer.index, top_ids, top_scores)
}

/// RMS normalization: y = x / sqrt(mean(x^2) + eps) * weight
/// Placeholder for V0 — returns input unchanged.
#[allow(dead_code)]
fn rms_norm(_x: &[Vec<f32>], _weight: &[f32], _eps: f32) -> Vec<Vec<f32>> {
    // V0 placeholder
    _x.to_vec()
}

/// Softmax over the last dimension.
#[allow(dead_code)]
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_moe::model::MoeModel;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(
            "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b"
        )
    }

    #[test]
    fn forward_pass_on_real_model_routes_experts() {
        let model = MoeModel::load(&model_path()).expect("load failed");
        let tokens = vec![0u32, 1, 2, 3]; // dummy tokens (embedding zeros for V0)
        let output = forward(&model, &tokens);

        // Should have one routing entry per layer
        assert_eq!(output.routes.len(), model.layers.len());

        // Each route should have top-8 experts
        for (layer_idx, expert_ids, scores) in &output.routes {
            assert_eq!(expert_ids.len(), 8, "layer {layer_idx}: expected 8 experts");
            assert_eq!(scores.len(), 8);
            // All expert IDs should be in [0, n_experts)
            assert!(expert_ids.iter().all(|&id| (id as u32) < model.config.n_experts));
            // All scores should be finite
            assert!(scores.iter().all(|&s| s.is_finite()));
        }
    }

    #[test]
    fn routing_with_zeros_input_still_produces_valid_scores() {
        let model = MoeModel::load(&model_path()).expect("load failed");
        let tokens = vec![0u32]; // single token, zero embedding
        let output = forward(&model, &tokens);

        let (_, expert_ids, _) = &output.routes[0];
        // With zero hidden state, all scores are zero. Top-k selects first 8.
        // Verify we get valid indices (not NaN or negative)
        assert_eq!(expert_ids.len(), 8);
        assert!(expert_ids.iter().all(|&id| id < 40));
    }
}
