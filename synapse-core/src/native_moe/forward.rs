/// MoE forward loop with external routing hooks.
///
/// Implements the transformer forward pass: embedding lookup → per-layer
/// RMS norm → attention → residual → RMS norm → MoE FFN with external
/// routing → residual → output norm → projection.
///
/// All heavy weights (attention, expert FFN) are loaded lazily via the
/// WeightProvider trait. The routing layer (gate_inp @ hidden_state → top-k)
/// is always computed externally from pre-loaded gate_inp weights.
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
    let n_tokens = prompt_tokens.len();
    let d_model = model.config.d_model as usize;

    // Phase 0: Embedding lookup. token_embd: [vocab, d_model]
    let mut hidden = if let Some(ref embd) = model.token_embd {
        let vocab = embd.shape[0] as usize;
        let d = embd.shape[1] as usize;
        prompt_tokens.iter().map(|&tid| {
            let idx = tid as usize % vocab;
            let offset = idx * d;
            embd.data[offset..offset + d].to_vec()
        }).collect()
    } else {
        vec![vec![0.0f32; d_model]; n_tokens]
    };

    let mut routes = Vec::new();

    // Phase 1: Per-layer forward pass
    for layer in &model.layers {
        // Phase 1a: RMS norm + attention (placeholder zeros for V0)
        // attention_weights loaded later — skip for V0

        // Phase 1b: RMS norm → gate_inp routing → store route
        let residual = hidden.clone();

        // Apply FFN norm if loaded
        if let Some(ref norm) = layer.ffn_norm {
            for t in 0..n_tokens {
                hidden[t] = rms_norm(&hidden[t], &norm.data);
            }
        }

        // External routing: score all experts for each token
        let route = route_experts(layer, &hidden);
        routes.push(route);

        // Phase 1c: Expert FFN placeholder (loaded later)
        // hidden stays as residual

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

    // Compute scores for last token
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
fn rms_norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let d = x.len();
    let eps = 1e-6_f32;

    // mean of squares
    let mut ss = 0.0_f32;
    for &v in x {
        ss += v * v;
    }
    let rms = (ss / d as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    x.iter().zip(weight.iter()).map(|(&v, &w)| v * inv_rms * w).collect()
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
        let model = MoeModel::load_routing(&model_path()).expect("load failed");
        let tokens = vec![0u32, 1, 2, 3];
        let output = forward(&model, &tokens);

        assert_eq!(output.routes.len(), model.layers.len());

        for (layer_idx, expert_ids, scores) in &output.routes {
            assert_eq!(expert_ids.len(), 8, "layer {layer_idx}: expected 8 experts");
            assert_eq!(scores.len(), 8);
            assert!(expert_ids.iter().all(|&id| (id as u32) < model.config.n_experts));
            assert!(scores.iter().all(|&s| s.is_finite()));
        }
    }

    #[test]
    fn routing_with_zeros_input_still_produces_valid_scores() {
        let model = MoeModel::load_routing(&model_path()).expect("load failed");
        let tokens = vec![0u32];
        let output = forward(&model, &tokens);

        let (_, expert_ids, _) = &output.routes[0];
        assert_eq!(expert_ids.len(), 8);
        assert!(expert_ids.iter().all(|&id| id < 40));
    }
}
