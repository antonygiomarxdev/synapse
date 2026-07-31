/// Multi-token generation with KV cache.
///
/// Implements autoregressive token generation: given a prompt, generates
/// tokens one at a time using a KV cache to avoid recomputing attention
/// for previous tokens.
use crate::native_moe::forward::{ForwardOutput, forward, softmax};
use crate::native_moe::model::MoeModel;
use crate::native_moe::ops::top_k;

/// KV cache for efficient autoregressive generation.
///
/// Stores key-value pairs for each layer and attention head.
/// On each new token, only the new K and V are computed and appended.
#[derive(Clone)]
pub struct KvCache {
    /// Key cache: [n_layers][n_kv_heads][seq_len][head_dim]
    pub keys: Vec<Vec<Vec<Vec<f32>>>>,
    /// Value cache: [n_layers][n_kv_heads][seq_len][head_dim]
    pub values: Vec<Vec<Vec<Vec<f32>>>>,
    /// Current sequence length in the cache
    pub seq_len: usize,
}

impl KvCache {
    /// Create a new empty KV cache for the given model configuration.
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            keys: vec![vec![vec![vec![0.0; head_dim]; 0]; n_kv_heads]; n_layers],
            values: vec![vec![vec![vec![0.0; head_dim]; 0]; n_kv_heads]; n_layers],
            seq_len: 0,
        }
    }

    /// Append new key-value pairs for a single token.
    pub fn append(&mut self, layer_idx: usize, kv_heads: usize, head_dim: usize, k: &[f32], v: &[f32]) {
        for h in 0..kv_heads {
            let k_start = h * head_dim;
            let v_start = h * head_dim;
            self.keys[layer_idx][h].push(k[k_start..k_start + head_dim].to_vec());
            self.values[layer_idx][h].push(v[v_start..v_start + head_dim].to_vec());
        }
        if layer_idx == 0 {
            self.seq_len += 1;
        }
    }
}

/// Sampling configuration.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for sampling (0.0 = greedy, 1.0 = default).
    pub temperature: f32,
    /// Top-k sampling (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) sampling (0.0 = disabled).
    pub top_p: f32,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// EOS token ID (stop generation when sampled).
    pub eos_token_id: Option<u32>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
            max_tokens: 100,
            eos_token_id: None,
        }
    }
}

/// Result of token generation.
#[derive(Debug, Clone)]
pub struct GenerateOutput {
    /// Generated tokens (excluding prompt).
    pub tokens: Vec<u32>,
    /// Logits for each generated token.
    pub logits: Vec<Vec<f32>>,
    /// Whether generation stopped due to EOS.
    pub stopped_by_eos: bool,
}

/// Generate tokens autoregressively using the model.
///
/// Runs the prompt through the model once, then generates tokens one at a time
/// using a KV cache for efficiency.
pub fn generate(
    model: &MoeModel,
    prompt_tokens: &[u32],
    config: &SamplingConfig,
) -> GenerateOutput {
    let mut all_tokens = prompt_tokens.to_vec();
    let mut generated_tokens = Vec::new();
    let mut generated_logits = Vec::new();
    let mut stopped_by_eos = false;

    // Run prompt through model (no KV cache optimization yet — full forward pass)
    let prompt_output = forward(model, &all_tokens);
    let mut next_token = sample_token(&prompt_output.logits, config);
    generated_tokens.push(next_token);
    generated_logits.push(prompt_output.logits.clone());
    all_tokens.push(next_token);

    // Check EOS
    if let Some(eos_id) = config.eos_token_id {
        if next_token == eos_id {
            stopped_by_eos = true;
        }
    }

    // Generate remaining tokens
    for _ in 1..config.max_tokens {
        if stopped_by_eos {
            break;
        }

        // Run forward pass on full sequence (no KV cache optimization)
        let output = forward(model, &all_tokens);
        next_token = sample_token(&output.logits, config);
        generated_tokens.push(next_token);
        generated_logits.push(output.logits.clone());
        all_tokens.push(next_token);

        // Check EOS
        if let Some(eos_id) = config.eos_token_id {
            if next_token == eos_id {
                stopped_by_eos = true;
            }
        }
    }

    GenerateOutput {
        tokens: generated_tokens,
        logits: generated_logits,
        stopped_by_eos,
    }
}

/// Sample a token from logits using the sampling configuration.
pub fn sample_token(logits: &[f32], config: &SamplingConfig) -> u32 {
    // Apply temperature
    let scaled: Vec<f32> = if config.temperature > 0.0 {
        logits.iter().map(|&l| l / config.temperature).collect()
    } else {
        logits.to_vec()
    };

    // Apply top-k
    let candidates = if config.top_k > 0 {
        top_k(&scaled, config.top_k)
    } else {
        scaled.iter().enumerate().map(|(i, &s)| (i, s)).collect()
    };

    // Apply top-p (nucleus sampling)
    let candidates = if config.top_p > 0.0 && config.top_p < 1.0 {
        let probs = softmax(&candidates.iter().map(|(_, s)| *s).collect::<Vec<_>>());
        let mut cumsum = 0.0;
        let mut nucleus = Vec::new();
        for (i, (idx, _)) in candidates.iter().enumerate() {
            cumsum += probs[i];
            nucleus.push((*idx, candidates[i].1));
            if cumsum >= config.top_p {
                break;
            }
        }
        nucleus
    } else {
        candidates
    };

    // Greedy or sample
    if config.temperature == 0.0 {
        // Greedy: pick highest logit from candidates
        candidates
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| *idx as u32)
            .unwrap_or(0)
    } else {
        // Sample from distribution
        let probs = softmax(&candidates.iter().map(|(_, s)| *s).collect::<Vec<_>>());
        let r: f32 = rand::random();
        let mut cumsum = 0.0;
        for (i, (idx, _)) in candidates.iter().enumerate() {
            cumsum += probs[i];
            if r <= cumsum {
                return *idx as u32;
            }
        }
        candidates.last().unwrap().0 as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_token_greedy_picks_highest() {
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let config = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let token = sample_token(&logits, &config);
        assert_eq!(token, 3); // index 3 has highest logit
    }

    #[test]
    fn sample_token_temperature_1_samples() {
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let config = SamplingConfig {
            temperature: 1.0,
            ..Default::default()
        };
        // With temperature 1.0, should sample (not deterministic)
        let token = sample_token(&logits, &config);
        assert!(token < 5);
    }

    #[test]
    fn sample_token_top_k_limits_candidates() {
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let config = SamplingConfig {
            temperature: 0.0,
            top_k: 2,
            ..Default::default()
        };
        // Top-2: indices 3 (0.8) and 1 (0.5)
        let token = sample_token(&logits, &config);
        assert_eq!(token, 3); // greedy picks highest
    }

    #[test]
    fn kv_cache_new_creates_empty() {
        let cache = KvCache::new(32, 8, 64);
        assert_eq!(cache.seq_len, 0);
        assert_eq!(cache.keys.len(), 32);
        assert_eq!(cache.keys[0].len(), 8);
    }

    #[test]
    fn kv_cache_append_increases_length() {
        let mut cache = KvCache::new(1, 2, 4);
        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 heads * 4 dims
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        cache.append(0, 2, 4, &k, &v);
        assert_eq!(cache.seq_len, 1);
        assert_eq!(cache.keys[0][0].len(), 1);
    }
}
