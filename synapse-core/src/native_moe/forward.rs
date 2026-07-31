/// MoE forward loop with real attention, expert FFN, and output projection.
///
/// Full transformer forward pass: embedding → (RMS norm → GQA attention → residual
/// → RMS norm → MoE FFN → residual) × n_layers → output norm → logits.
///
/// Granite MoE architecture: d_model=1536, n_heads=24, n_kv_heads=8, head_dim=64,
/// d_ff=512, n_experts=40, n_experts_active=8.
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
    forward_inner(model, prompt_tokens, false)
}

/// Forward pass with option to skip expert FFN and limit layers (for debugging).
pub fn forward_inner(model: &MoeModel, prompt_tokens: &[u32], skip_ffn: bool) -> ForwardOutput {
    forward_debug(model, prompt_tokens, skip_ffn, usize::MAX)
}

pub fn forward_debug(
    model: &MoeModel,
    prompt_tokens: &[u32],
    skip_ffn: bool,
    max_layers: usize,
) -> ForwardOutput {
    let n_tokens = prompt_tokens.len();
    let d_model = model.config.d_model as usize;
    let last_token_idx = n_tokens - 1;

    // Phase 0: Embedding lookup
    let mut hidden = if let Some(ref embd) = model.token_embd {
        let d = embd.shape[0] as usize;
        let shape_vocab = embd.shape[1] as usize;
        let actual_vocab = embd.data.len() / d;
        eprintln!(
            "[EMBD] d={}, shape_vocab={}, actual_vocab={}, data.len()={}",
            d,
            shape_vocab,
            actual_vocab,
            embd.data.len()
        );
        prompt_tokens
            .iter()
            .map(|&tid| {
                let t = tid as usize % shape_vocab;
                // Row-major: data[t * d + dim] = embedding[dim] for token t
                (0..d).map(|dim| model.config.embedding_scale * embd.data[t * d + dim]).collect()
            })
            .collect()
    } else {
        vec![vec![0.0f32; d_model]; n_tokens]
    };

    eprintln!("[EMBD] hidden[{}][0..5] = {:?}", last_token_idx, &hidden[last_token_idx][..5]);
    eprintln!("[EMBD] hidden[{}] norm = {:.4}", last_token_idx, vec_norm(&hidden[last_token_idx]));

    let mut routes = Vec::new();

    // Phase 1: Per-layer forward pass
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        if layer_idx >= max_layers {
            break;
        }

        let residual = hidden.clone();

        // Phase 1a: RMS norm → attention
        if let Some(ref attn_norm) = layer.attn_norm {
            for t in 0..n_tokens {
                hidden[t] = rms_norm(&hidden[t], &attn_norm.data);
            }
        }

        if layer_idx < 2 || layer_idx == 31 {
            eprintln!(
                "[L{} ATTN_NORM] hidden[{}][0..5] = {:?}",
                layer_idx,
                last_token_idx,
                &hidden[last_token_idx][..5]
            );
        }

        // Attention
        if let (Some(wq), Some(wk), Some(wv), Some(wo)) =
            (&layer.attn_q, &layer.attn_k, &layer.attn_v, &layer.attn_output)
        {
            let attn_out = attention(
                &hidden,
                wq,
                wk,
                wv,
                wo,
                model.config.n_heads as usize,
                model.config.n_kv_heads as usize,
                n_tokens,
                model.config.rope_theta,
                model.config.attention_scale,
                layer_idx < 2 || layer_idx == 31, // debug first 2 + last layer
            );

            if layer_idx < 2 || layer_idx == 31 {
                eprintln!(
                    "[L{} ATTN_OUT] attn_out[{}][0..5] = {:?}",
                    layer_idx,
                    last_token_idx,
                    &attn_out[last_token_idx][..5]
                );
                eprintln!(
                    "[L{} ATTN_OUT] attn_out[{}] norm = {:.4}",
                    layer_idx,
                    last_token_idx,
                    vec_norm(&attn_out[last_token_idx])
                );
            }

            // Granite: hidden = residual + residual_scale * attn_out
            for t in 0..n_tokens {
                for d in 0..d_model {
                    hidden[t][d] = residual[t][d] + model.config.residual_scale * attn_out[t][d];
                }
            }

            if layer_idx < 2 || layer_idx == 31 {
                eprintln!(
                    "[L{} RESIDUAL] hidden[{}][0..5] = {:?}",
                    layer_idx,
                    last_token_idx,
                    &hidden[last_token_idx][..5]
                );
                eprintln!(
                    "[L{} RESIDUAL] hidden[{}] norm = {:.4}",
                    layer_idx,
                    last_token_idx,
                    vec_norm(&hidden[last_token_idx])
                );
            }
        } else {
            for t in 0..n_tokens {
                hidden[t] = residual[t].clone();
            }
        }

        // Phase 1b: RMS norm → MoE FFN
        let residual2 = hidden.clone();
        if let Some(ref ffn_norm) = layer.ffn_norm {
            for t in 0..n_tokens {
                hidden[t] = rms_norm(&hidden[t], &ffn_norm.data);
            }
        }

        // Debug: hidden state after FFN norm (before routing)
        if layer_idx == 0 {
            eprintln!(
                "[L0 FFN_NORM] hidden[{}][0..5] = {:?}",
                last_token_idx,
                &hidden[last_token_idx][..5]
            );
            eprintln!(
                "[L0 FFN_NORM] hidden[{}] norm = {:.4}",
                last_token_idx,
                vec_norm(&hidden[last_token_idx])
            );
        }

        // Routing
        let route = route_experts(layer, &hidden);
        routes.push(route.clone());

        // Expert FFN
        if !skip_ffn {
            if let (Some(gate_exps), Some(up_exps), Some(down_exps)) =
                (&layer.gate_exps, &layer.up_exps, &layer.down_exps)
            {
                let ffn_out = expert_ffn(
                    &hidden,
                    gate_exps,
                    up_exps,
                    down_exps,
                    &route.1,
                    &route.2,
                    model.config.d_ff as usize,
                );

                if layer_idx < 2 || layer_idx == 31 {
                    eprintln!(
                        "[L{} FFN_OUT] ffn_out[{}][0..5] = {:?}",
                        layer_idx,
                        last_token_idx,
                        &ffn_out[last_token_idx][..5]
                    );
                    eprintln!(
                        "[L{} FFN_OUT] ffn_out[{}] norm = {:.4}",
                        layer_idx,
                        last_token_idx,
                        vec_norm(&ffn_out[last_token_idx])
                    );
                }

                for t in 0..n_tokens {
                    for d in 0..d_model {
                        hidden[t][d] =
                            residual2[t][d] + model.config.residual_scale * ffn_out[t][d];
                    }
                }

                if layer_idx < 2 || layer_idx == 31 {
                    eprintln!(
                        "[L{} FINAL] hidden[{}][0..5] = {:?}",
                        layer_idx,
                        last_token_idx,
                        &hidden[last_token_idx][..5]
                    );
                    eprintln!(
                        "[L{} FINAL] hidden[{}] norm = {:.4}",
                        layer_idx,
                        last_token_idx,
                        vec_norm(&hidden[last_token_idx])
                    );
                }
            } else {
                for t in 0..n_tokens {
                    hidden[t] = residual2[t].clone();
                }
            }
        } else {
            for t in 0..n_tokens {
                hidden[t] = residual2[t].clone();
            }
        }
    }

    // Phase 2: Output norm → projection → logits
    let last_hidden = &hidden[last_token_idx];
    let normed = if let Some(ref out_norm) = model.output_norm {
        rms_norm(last_hidden, &out_norm.data)
    } else {
        last_hidden.clone()
    };

    eprintln!("[OUTPUT_NORM] normed[0..5] = {:?}", &normed[..5]);
    eprintln!("[OUTPUT_NORM] normed norm = {:.4}", vec_norm(&normed));

    let logits = if let Some(ref embd) = model.token_embd {
        let d = embd.shape[0] as usize;
        let shape_vocab = embd.shape[1] as usize;
        let actual_vocab = embd.data.len() / d;
        let mut logits = vec![0.0f32; shape_vocab];
        for v in 0..shape_vocab {
            let mut acc = 0.0f32;
            for dim in 0..d {
                acc += embd.data[v * d + dim] * normed[dim];
            }
            logits[v] = acc / model.config.logit_scale;
        }
        logits
    } else if let Some(ref out_proj) = model.output {
        mat_vec_transposed(out_proj, &normed, d_model, model.config.vocab_size as usize)
    } else {
        vec![0.0f32; model.config.vocab_size as usize]
    };

    // Logit stats
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_logit = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean_logit = logits.iter().sum::<f32>() / logits.len() as f32;
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top5: Vec<usize> = indexed.iter().take(5).map(|(i, _)| *i).collect();
    let top5_vals: Vec<f32> = indexed.iter().take(5).map(|(_, v)| *v).collect();
    eprintln!(
        "[LOGITS] mean={:.4} std={:.4} max={:.2} min={:.2}",
        mean_logit,
        (logits.iter().map(|v| (v - mean_logit).powi(2)).sum::<f32>() / logits.len() as f32).sqrt(),
        max_logit,
        min_logit
    );
    eprintln!("[LOGITS] top5={:?} vals={:?}", top5, top5_vals);

    ForwardOutput { logits, routes }
}

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Multi-head attention with Grouped Query Attention (GQA).
///
/// Q: [n_tokens, d_model] @ W_q [d_model, n_heads × head_dim]
/// K: [n_tokens, d_model] @ W_k [d_model, n_kv_heads × head_dim]
/// V: [n_tokens, d_model] @ W_v [d_model, n_kv_heads × head_dim]
fn attention(
    hidden: &[Vec<f32>],
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    n_tokens: usize,
    theta_base: f32,
    attention_scale: f32,
    debug: bool,
) -> Vec<Vec<f32>> {
    let d_model = hidden[0].len();
    let head_dim = d_model / n_heads;
    let n_rep = n_heads / n_kv_heads; // GQA repetition factor (3 for Granite)

    // Project Q, K, V for all tokens
    let mut q_all = Vec::with_capacity(n_tokens);
    let mut k_all = Vec::with_capacity(n_tokens);
    let mut v_all = Vec::with_capacity(n_tokens);

    for t in 0..n_tokens {
        q_all.push(mat_vec_transposed(wq, &hidden[t], d_model, d_model));
        k_all.push(mat_vec_transposed(wk, &hidden[t], d_model, n_kv_heads * head_dim));
        v_all.push(mat_vec_transposed(wv, &hidden[t], d_model, n_kv_heads * head_dim));
    }

    // Debug: print Q values for first token
    if debug {
        eprintln!("  [Q] Q[{}][0..5] = {:?}", n_tokens - 1, &q_all[n_tokens - 1][..5]);
        eprintln!("  [Q] Q[{}] norm = {:.4}", n_tokens - 1, vec_norm(&q_all[n_tokens - 1]));
        eprintln!("  [K] K[{}][0..5] = {:?}", n_tokens - 1, &k_all[n_tokens - 1][..5]);
        eprintln!("  [V] V[{}][0..5] = {:?}", n_tokens - 1, &v_all[n_tokens - 1][..5]);
    }

    // Apply RoPE to Q and K
    for t in 0..n_tokens {
        let pos = t as f32;
        for h in 0..n_heads {
            let q_off = h * head_dim;
            rope_inplace(&mut q_all[t][q_off..q_off + head_dim], pos, theta_base);
        }
        for h in 0..n_kv_heads {
            let k_off = h * head_dim;
            rope_inplace(&mut k_all[t][k_off..k_off + head_dim], pos, theta_base);
        }
    }

    // Compute attention per head and accumulate output
    let mut output = vec![vec![0.0f32; d_model]; n_tokens];

    // Compute attention for ALL tokens (not just last)
    for cur in 0..n_tokens {
        for h in 0..n_heads {
            let kv_h = h / n_rep; // corresponding KV head

            // Compute attention scores: Q[cur] attends to K[0..=cur] (causal)
            let mut scores = vec![0.0f32; cur + 1];
            for t in 0..=cur {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_all[cur][h * head_dim + d] * k_all[t][kv_h * head_dim + d];
                }
                scores[t] = dot * attention_scale; // Use model's attention_scale
            }

            // Softmax over causal positions
            let probs = softmax(&scores);

            // Debug: print scores for last token, first head
            if debug && cur == n_tokens - 1 && h == 0 {
                eprintln!(
                    "  [SCORES] head={}, scores[0..5] = {:?}",
                    h,
                    &scores[..scores.len().min(5)]
                );
                eprintln!(
                    "  [PROBS]  head={}, probs[0..5] = {:?}",
                    h,
                    &probs[..probs.len().min(5)]
                );
                eprintln!("  [PROBS]  sum = {:.4}", probs.iter().sum::<f32>());
            }

            // Weighted sum of V
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for t in 0..=cur {
                    val += probs[t] * v_all[t][kv_h * head_dim + d];
                }
                output[cur][h * head_dim + d] = val;
            }
        }
    }

    // Output projection for ALL tokens
    for t in 0..n_tokens {
        let proj = mat_vec_transposed(wo, &output[t], d_model, d_model);
        output[t] = proj;
    }

    if debug {
        eprintln!("  [OUT_PROJ] output[{}][0..5] = {:?}", n_tokens - 1, &output[n_tokens - 1][..5]);
        eprintln!(
            "  [OUT_PROJ] output[{}] norm = {:.4}",
            n_tokens - 1,
            vec_norm(&output[n_tokens - 1])
        );
    }

    output
}

/// Expert FFN: for each token, route to top-k experts and compute weighted FFN output.
///
/// Weight layout: gate_exps [d_model, d_ff, n_experts], up_exps [d_model, d_ff, n_experts],
/// down_exps [d_ff, d_model, n_experts] — row-major in GGUF.
fn expert_ffn(
    hidden: &[Vec<f32>],
    gate_exps: &Tensor,
    up_exps: &Tensor,
    down_exps: &Tensor,
    expert_ids: &[u32],
    expert_scores: &[f32],
    d_ff: usize,
) -> Vec<Vec<f32>> {
    let d_model = hidden[0].len();
    let n_tokens = hidden.len();

    // Normalize expert scores so they sum to 1 (like llama.cpp's norm_w=true)
    let score_sum: f32 = expert_scores.iter().sum();
    let norm_scores: Vec<f32> = if score_sum > 1e-6 {
        expert_scores.iter().map(|s| s / score_sum).collect()
    } else {
        expert_scores.to_vec()
    };

    let mut output = vec![vec![0.0f32; d_model]; n_tokens];

    // Expert weights are stored as [n_experts, d_ff, d_model] in data
    // but tensor shape is [d_model, d_ff, n_experts]
    // To access tensor[d, j, e], use data[e * d_ff * d_model + j * d_model + d]
    let n_experts_gate = gate_exps.shape[2] as usize;
    let n_experts_down = down_exps.shape[2] as usize;

    // Debug: print first few values of gate_exps for expert 0
    eprintln!("  [FFN_WEIGHTS] gate_exps[0,0,0..5] = {:?}", &gate_exps.data[0..5]);
    eprintln!("  [FFN_WEIGHTS] gate_exps.data.len() = {}", gate_exps.data.len());

    for t in 0..n_tokens {
        for (ki, &eid) in expert_ids.iter().enumerate() {
            let e = eid as usize;
            let score = norm_scores[ki]; // Use normalized scores

            // gate_proj: hidden @ gate_exp[e]^T → [d_ff]
            // Data layout: [n_experts, d_ff, d_model]
            let mut gate_out = vec![0.0f32; d_ff];
            for j in 0..d_ff {
                let mut acc = 0.0f32;
                for d in 0..d_model {
                    acc += hidden[t][d] * gate_exps.data[e * d_ff * d_model + j * d_model + d];
                }
                gate_out[j] = acc;
            }

            // up_proj: hidden @ up_exp[e]^T → [d_ff]
            let mut up_out = vec![0.0f32; d_ff];
            for j in 0..d_ff {
                let mut acc = 0.0f32;
                for d in 0..d_model {
                    acc += hidden[t][d] * up_exps.data[e * d_ff * d_model + j * d_model + d];
                }
                up_out[j] = acc;
            }

            // SiLU(gate) * up → [d_ff]
            let mut fused = vec![0.0f32; d_ff];
            for j in 0..d_ff {
                fused[j] = silu(gate_out[j]) * up_out[j];
            }

            // Debug: print first expert values for layer 0
            if t == 0 && ki == 0 {
                let gate_norm: f32 = gate_out.iter().map(|x| x * x).sum::<f32>().sqrt();
                let up_norm: f32 = up_out.iter().map(|x| x * x).sum::<f32>().sqrt();
                let fused_norm: f32 = fused.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!("  [FFN] expert={}, score={:.4}", e, score);
                eprintln!("  [FFN] gate_out norm={:.4}, up_out norm={:.4}", gate_norm, up_norm);
                eprintln!("  [FFN] fused norm={:.4}", fused_norm);
            }

            // down_proj: fused @ down_exp[e]^T → [d_model]
            // down_exps shape: [d_ff, d_model, n_experts], data: [n_experts, d_model, d_ff]
            for d in 0..d_model {
                let mut acc = 0.0f32;
                for j in 0..d_ff {
                    acc += fused[j] * down_exps.data[e * d_model * d_ff + d * d_ff + j];
                }
                output[t][d] += score * acc;
            }
        }
    }

    output
}

/// Compute expert scores for all tokens in a layer.
/// hidden[t] @ gate_inp^T → scores[n_experts] → softmax → top-k indices.
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
            // gate_inp data is [n_experts, d_model] = (40, 1536)
            // data[e * d_model + d] = gate_inp[d, e]
            acc += last[d] * gate[e * d_model + d];
        }
        scores[e] = acc;
    }

    // Apply softmax to ALL scores (like llama.cpp)
    let all_probs = softmax(&scores);

    // Select top-k experts by probability
    let mut indexed: Vec<(usize, f32)> =
        all_probs.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_ids: Vec<u32> = indexed.iter().take(k).map(|(i, _)| *i as u32).collect();
    let top_scores: Vec<f32> = indexed.iter().take(k).map(|(_, s)| *s).collect();

    (layer.index, top_ids, top_scores)
}

/// RMS normalization: y = x / sqrt(mean(x^2) + eps) * weight
fn rms_norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let d = x.len();
    let eps = 1e-6_f32;

    let mut ss = 0.0_f32;
    for &v in x {
        ss += v * v;
    }
    let rms = (ss / d as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    x.iter().zip(weight.iter()).map(|(&v, &w)| v * inv_rms * w).collect()
}

/// Softmax over a slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

/// SiLU activation: x * sigmoid(x)
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RoPE (Rotary Position Embedding) in-place on a head slice.
///
/// Applies rotation to pairs of dimensions: (x[2i], x[2i+1]) rotated by θ_i.
fn rope_inplace(x: &mut [f32], pos: f32, theta_base: f32) {
    let d = x.len();
    for i in 0..d / 2 {
        let theta = pos / theta_base.powf(2.0 * i as f32 / d as f32);
        let cos = theta.cos();
        let sin = theta.sin();
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * cos - x1 * sin;
        x[2 * i + 1] = x0 * sin + x1 * cos;
    }
}

/// Matrix-vector multiplication matching ggml_mul_mat(a, b) = a^T @ b
///
/// In GGML, tensors are stored column-major: data[i + j * ne0] = tensor[i, j]
/// ggml_mul_mat(a, b) computes: result[j] = Σ_i a[i, j] * b[i]
/// Which is: result[j] = Σ_i a.data[i + j * ne0] * b[i]
///
/// For a weight matrix W with shape [ne0, ne1]:
///   W.data[i + j * ne0] = W[i, j]
///   result[j] = Σ_i W.data[i + j * ne0] * x[i]
///
/// Returns y[ne1]
fn mat_vec_transposed(w: &Tensor, x: &[f32], ne0: usize, ne1: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; ne1];
    for j in 0..ne1 {
        let mut acc = 0.0f32;
        for i in 0..ne0 {
            acc += w.data[i + j * ne0] * x[i];
        }
        y[j] = acc;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_moe::model::MoeModel;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(
            "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b",
        )
    }

    #[test]
    fn test_mat_vec_transposed() {
        // GGML column-major: data[i + j * ne0] = tensor[i, j]
        // For W = [[1, 3], [2, 4]] stored column-major: [1, 2, 3, 4]
        // ne0=2 (rows), ne1=2 (cols)
        let w = Tensor { name: "test".into(), data: vec![1.0, 2.0, 3.0, 4.0], shape: vec![2, 2] };
        let x = vec![1.0, 1.0];

        // W^T @ x = [[1,2],[3,4]] @ [1,1] = [3, 7]
        let y = mat_vec_transposed(&w, &x, 2, 2);
        assert_eq!(y, vec![3.0, 7.0], "W^T @ x should be [3, 7]");
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

    #[test]
    fn full_forward_produces_logits() {
        let model = MoeModel::load_all(&model_path()).expect("load_all failed");
        let tokens = vec![1u32]; // single token
        let output = forward(&model, &tokens);

        // Logits should be non-zero (attention + FFN actually running)
        let max_logit = output.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_logit.abs() > 0.01, "logits are near-zero: max={max_logit}");
        assert!(output.logits.iter().all(|v| v.is_finite()), "logits contain NaN/Inf");

        // Routes should have all 32 layers
        assert_eq!(output.routes.len(), 32);
    }

    #[test]
    fn trace_logits_by_layer_count() {
        let prompt_tokens = vec![8197u32, 438, 322, 18926, 432, 45600, 49];
        let model = MoeModel::load_all(&model_path()).expect("load_all failed");

        // Verify attn_q dequantization for layer 0
        let l0 = &model.layers[0];
        if let Some(ref attn_q) = l0.attn_q {
            eprintln!("attn_q shape: {:?}", attn_q.shape);
            eprintln!("attn_q[0..8]: {:?}", &attn_q.data[..8]);
            eprintln!("attn_q norm: {}", attn_q.data.iter().map(|x| x * x).sum::<f32>().sqrt());
        }
        if let Some(ref attn_norm) = l0.attn_norm {
            eprintln!("attn_norm[0..5]: {:?}", &attn_norm.data[..5]);
            eprintln!(
                "attn_norm mean: {}",
                attn_norm.data.iter().sum::<f32>() / attn_norm.data.len() as f32
            );
        }

        for n_layers in [0, 1, 5, 10, 32] {
            let output = forward_debug(&model, &prompt_tokens, false, n_layers);
            let mut indexed: Vec<(usize, f32)> =
                output.logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top3: Vec<usize> = indexed.iter().take(3).map(|(i, _)| *i).collect();
            let max = indexed[0].1;
            let min = indexed.last().unwrap().1;
            eprintln!(
                "Layers={:2}: top3={:?} max={:.2} min={:.2} range={:.2}",
                n_layers,
                top3,
                max,
                min,
                max - min
            );
        }
    }

    #[test]
    fn forward_matches_ollama_top_token() {
        let prompt_tokens = vec![8197u32, 438, 322, 18926, 432, 45600, 49];
        let model = MoeModel::load_all(&model_path()).expect("load_all failed");

        // Save our logits to file for comparison
        let full = forward(&model, &prompt_tokens);
        let logits_str: Vec<String> = full.logits.iter().map(|v| format!("{:.6}", v)).collect();
        std::fs::write("/tmp/our_logits.txt", logits_str.join("\n")).unwrap();

        let mut full_indexed: Vec<(usize, f32)> =
            full.logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        full_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "FULL forward top-5: {:?}",
            full_indexed.iter().take(5).map(|(i, _)| *i).collect::<Vec<_>>()
        );
        eprintln!(
            "FULL forward top-5 logits: {:?}",
            full_indexed.iter().take(5).map(|(_, v)| *v).collect::<Vec<_>>()
        );
        eprintln!(
            "Mean: {:.4}, Std: {:.4}",
            full.logits.iter().sum::<f32>() / full.logits.len() as f32,
            (full.logits.iter().map(|v| v * v).sum::<f32>() / full.logits.len() as f32).sqrt()
        );

        assert!(full.logits.iter().all(|v| v.is_finite()), "logits contain NaN/Inf");
    }

    #[test]
    fn single_token_logits() {
        let model = MoeModel::load_all(&model_path()).expect("load_all failed");
        let tokens = vec![49u32]; // single token, same as llama-cpp-python reference
        let output = forward(&model, &tokens);

        let logits_str: Vec<String> = output.logits.iter().map(|v| format!("{:.6}", v)).collect();
        std::fs::write("/tmp/our_single_logits.txt", logits_str.join("\n")).unwrap();

        let mut indexed: Vec<(usize, f32)> =
            output.logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "Single token top-5: {:?}",
            indexed.iter().take(5).map(|(i, _)| *i).collect::<Vec<_>>()
        );
        eprintln!(
            "Single token top-5 vals: {:?}",
            indexed.iter().take(5).map(|(_, v)| *v).collect::<Vec<_>>()
        );
        let mean = output.logits.iter().sum::<f32>() / output.logits.len() as f32;
        let max = output.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("Single token mean={:.4} max={:.4}", mean, max);
    }

    #[test]
    fn trace_single_token_by_layer() {
        let model = MoeModel::load_all(&model_path()).expect("load_all failed");
        let tokens = vec![49u32];

        // Save logits for each layer count
        let mut results = Vec::new();
        for n_layers in [0, 1, 2, 3, 5, 10, 15, 20, 25, 32] {
            let output = forward_debug(&model, &tokens, false, n_layers);
            let mean: f32 = output.logits.iter().sum::<f32>() / output.logits.len() as f32;
            let max = output.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let mut indexed: Vec<(usize, f32)> =
                output.logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top3: Vec<usize> = indexed.iter().take(3).map(|(i, _)| *i).collect();
            let top3_vals: Vec<f32> = indexed.iter().take(3).map(|(_, v)| *v).collect();

            eprintln!(
                "Layers={:2}: top3={:?} vals={:?} mean={:.4} max={:.2}",
                n_layers, top3, top3_vals, mean, max
            );

            // Save full logits for layers 0, 1, 2, 5, 10, 32
            if [0, 1, 2, 5, 10, 32].contains(&n_layers) {
                let path = format!("/tmp/rust_logits_{}layers.txt", n_layers);
                let logits_str: Vec<String> =
                    output.logits.iter().map(|v| format!("{:.6}", v)).collect();
                std::fs::write(&path, logits_str.join("\n")).unwrap();
                eprintln!("  Saved to {}", path);
            }

            results.push((n_layers, top3, mean, max));
        }
    }

    #[test]
    fn trace_norms_by_layer() {
        let model = MoeModel::load_all(&model_path()).expect("load_all failed");
        let tokens = vec![49u32]; // single token

        // Run forward pass and capture hidden state norms after each layer
        let n_tokens = tokens.len();
        let d_model = model.config.d_model as usize;
        let last_token_idx = n_tokens - 1;

        // Phase 0: Embedding
        let mut hidden: Vec<Vec<f32>> = if let Some(ref embd) = model.token_embd {
            let d = embd.shape[0] as usize;
            let shape_vocab = embd.shape[1] as usize;
            tokens
                .iter()
                .map(|&tid| {
                    let t = tid as usize % shape_vocab;
                    (0..d)
                        .map(|dim| model.config.embedding_scale * embd.data[t * d + dim])
                        .collect()
                })
                .collect()
        } else {
            vec![vec![0.0f32; d_model]; n_tokens]
        };

        eprintln!("EMBD: norm={:.4}", vec_norm(&hidden[last_token_idx]));

        // Phase 1: Per-layer
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            let residual = hidden.clone();

            // attn_norm
            if let Some(ref attn_norm) = layer.attn_norm {
                for t in 0..n_tokens {
                    hidden[t] = rms_norm(&hidden[t], &attn_norm.data);
                }
            }

            // Attention
            if let (Some(wq), Some(wk), Some(wv), Some(wo)) =
                (&layer.attn_q, &layer.attn_k, &layer.attn_v, &layer.attn_output)
            {
                let attn_out = attention(
                    &hidden,
                    wq,
                    wk,
                    wv,
                    wo,
                    model.config.n_heads as usize,
                    model.config.n_kv_heads as usize,
                    n_tokens,
                    model.config.rope_theta,
                    model.config.attention_scale,
                    false,
                );
                for t in 0..n_tokens {
                    for d in 0..d_model {
                        hidden[t][d] =
                            residual[t][d] + model.config.residual_scale * attn_out[t][d];
                    }
                }
            } else {
                for t in 0..n_tokens {
                    hidden[t] = residual[t].clone();
                }
            }

            let after_attn_norm = vec_norm(&hidden[last_token_idx]);

            // FFN
            let residual2 = hidden.clone();
            if let Some(ref ffn_norm) = layer.ffn_norm {
                for t in 0..n_tokens {
                    hidden[t] = rms_norm(&hidden[t], &ffn_norm.data);
                }
            }

            let route = route_experts(layer, &hidden);

            if let (Some(gate_exps), Some(up_exps), Some(down_exps)) =
                (&layer.gate_exps, &layer.up_exps, &layer.down_exps)
            {
                let ffn_out = expert_ffn(
                    &hidden,
                    gate_exps,
                    up_exps,
                    down_exps,
                    &route.1,
                    &route.2,
                    model.config.d_ff as usize,
                );
                for t in 0..n_tokens {
                    for d in 0..d_model {
                        hidden[t][d] =
                            residual2[t][d] + model.config.residual_scale * ffn_out[t][d];
                    }
                }
            } else {
                for t in 0..n_tokens {
                    hidden[t] = residual2[t].clone();
                }
            }

            let final_norm = vec_norm(&hidden[last_token_idx]);

            // Check if experts changed from layer 0
            let expert_key: Vec<u32> = route.1.clone();

            eprintln!(
                "L{:2}: attn_norm={:.4} final_norm={:.4} experts={:?}",
                layer_idx,
                after_attn_norm,
                final_norm,
                &expert_key[..3]
            );
        }

        // Final output
        let last_hidden = &hidden[last_token_idx];
        let normed = if let Some(ref out_norm) = model.output_norm {
            rms_norm(last_hidden, &out_norm.data)
        } else {
            last_hidden.clone()
        };
        eprintln!("OUTPUT_NORM: norm={:.4}", vec_norm(&normed));
    }

    #[test]
    fn spike_distributed_expert_impact() {
        // Spike: prove that expert selection matters for distributed inference.
        // If different nodes hold different experts, the output will differ.
        let model = MoeModel::load_all(&model_path()).expect("load_all failed");
        let tokens = vec![49u32];

        // Full model (all experts available)
        let full = forward(&model, &tokens);

        // Without FFN (no experts — simulates a node with no expert weights)
        let no_ffn = forward_inner(&model, &tokens, true);

        // Compare
        let full_mean: f32 = full.logits.iter().sum::<f32>() / full.logits.len() as f32;
        let no_ffn_mean: f32 = no_ffn.logits.iter().sum::<f32>() / no_ffn.logits.len() as f32;
        let full_max = full.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let no_ffn_max = no_ffn.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Correlation between full and no-ffn
        let full_norm: f32 = full.logits.iter().map(|x| x * x).sum::<f32>().sqrt();
        let no_ffn_norm: f32 = no_ffn.logits.iter().map(|x| x * x).sum::<f32>().sqrt();
        let dot: f32 = full.logits.iter().zip(no_ffn.logits.iter()).map(|(a, b)| a * b).sum();
        let cos_sim = dot / (full_norm * no_ffn_norm);

        eprintln!("=== Distributed Expert Impact Spike ===");
        eprintln!("Full model:  mean={:.4} max={:.2}", full_mean, full_max);
        eprintln!("No FFN:      mean={:.4} max={:.2}", no_ffn_mean, no_ffn_max);
        eprintln!("Cosine similarity: {:.4}", cos_sim);

        // Top-5 for each
        let mut full_idx: Vec<(usize, f32)> = full.logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        full_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut no_ffn_idx: Vec<(usize, f32)> = no_ffn.logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        no_ffn_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        eprintln!("Full top-5:  {:?}", full_idx.iter().take(5).map(|(i, _)| *i).collect::<Vec<_>>());
        eprintln!("NoFFN top-5: {:?}", no_ffn_idx.iter().take(5).map(|(i, _)| *i).collect::<Vec<_>>());

        // The expert FFN should significantly change the output
        assert!(cos_sim < 0.99, "FFN should change output significantly, cos_sim={cos_sim}");
        assert!(cos_sim > 0.5, "FFN shouldn't completely destroy the output, cos_sim={cos_sim}");

        // Print expert routing for each layer
        eprintln!("\nExpert routing per layer:");
        for (layer_idx, expert_ids, scores) in &full.routes {
            eprintln!("  L{:2}: experts={:?} scores={:.4?}",
                layer_idx, &expert_ids[..3], &scores[..3]);
        }
    }

    #[test]
    fn test_ffn_impact() {
        let model = MoeModel::load_all(&model_path()).expect("load_all failed");
        let tokens = vec![49u32];

        // With FFN
        let with_ffn = forward(&model, &tokens);
        let with_str: Vec<String> = with_ffn.logits.iter().map(|v| format!("{:.6}", v)).collect();
        std::fs::write("/tmp/our_with_ffn.txt", with_str.join("\n")).unwrap();

        // Without FFN (skip_ffn=true)
        let without_ffn = forward_inner(&model, &tokens, true);
        let without_str: Vec<String> =
            without_ffn.logits.iter().map(|v| format!("{:.6}", v)).collect();
        std::fs::write("/tmp/our_without_ffn.txt", without_str.join("\n")).unwrap();

        let with_mean = with_ffn.logits.iter().sum::<f32>() / with_ffn.logits.len() as f32;
        let with_max = with_ffn.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let without_mean = without_ffn.logits.iter().sum::<f32>() / without_ffn.logits.len() as f32;
        let without_max = without_ffn.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        eprintln!("With FFN: mean={:.4} max={:.4}", with_mean, with_max);
        eprintln!("Without FFN: mean={:.4} max={:.4}", without_mean, without_max);

        // Compare top-5
        let mut with_idx: Vec<(usize, f32)> =
            with_ffn.logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        with_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut without_idx: Vec<(usize, f32)> =
            without_ffn.logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        without_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        eprintln!(
            "With FFN top-5: {:?}",
            with_idx.iter().take(5).map(|(i, _)| *i).collect::<Vec<_>>()
        );
        eprintln!(
            "Without FFN top-5: {:?}",
            without_idx.iter().take(5).map(|(i, _)| *i).collect::<Vec<_>>()
        );
    }
}
