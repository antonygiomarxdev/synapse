/// Distributed forward pass: coordinator runs attention locally,
/// dispatches expert FFN to remote workers.
///
/// The coordinator loads only embedding, attention weights, norms,
/// and gate_inp (routing). Heavy expert FFN weights live on worker
/// nodes that serve FFN requests over HTTP.
use std::collections::HashMap;

use super::expert_worker_client::ExpertWorkerClient;
use super::forward::{
    combine_ffn_residual, compute_logits, forward_layer_attention,
    ForwardOutput,
};
use super::model::{MoeConfig, MoeModel};

/// Configuration for a remote expert worker.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub url: String,
    pub expert_indices: Vec<usize>,
}

/// Distributed inference model.
///
/// Coordinator holds attention + routing weights.
/// Expert FFN is dispatched to remote workers.
pub struct DistributedModel {
    pub model: MoeModel,
    pub workers: Vec<ExpertWorkerClient>,
    /// Global expert ID → worker index
    pub expert_map: HashMap<usize, usize>,
}

impl DistributedModel {
    /// Create a distributed model from a coordinator model (with routing
    /// weights but no expert FFN weights) and a list of worker configs.
    pub fn new(model: MoeModel, worker_configs: &[WorkerConfig]) -> Self {
        let workers: Vec<ExpertWorkerClient> = worker_configs
            .iter()
            .map(|c| ExpertWorkerClient::new(c.url.clone()))
            .collect();

        let mut expert_map = HashMap::new();
        for (wid, config) in worker_configs.iter().enumerate() {
            for &eid in &config.expert_indices {
                expert_map.insert(eid, wid);
            }
        }

        DistributedModel {
            model,
            workers,
            expert_map,
        }
    }

    /// Run distributed forward pass on prompt tokens.
    ///
    /// For each layer:
    /// 1. Run attention locally (coordinator)
    /// 2. Route experts via gate_inp
    /// 3. Dispatch expert FFN to remote workers (concurrent)
    /// 4. Combine results and continue
    pub async fn forward(
        &self,
        prompt_tokens: &[u32],
    ) -> ForwardOutput {
        let d_model = self.model.config.d_model as usize;
        let d_ff = self.model.config.d_ff as usize;
        let residual_scale = self.model.config.residual_scale;

        // Phase 0: Embedding lookup (local)
        let mut hidden: Vec<Vec<f32>> =
            if let Some(ref embd) = self.model.token_embd {
                let d = embd.shape[0] as usize;
                let shape_vocab = embd.shape[1] as usize;
                prompt_tokens
                    .iter()
                    .map(|&tid| {
                        let t = tid as usize % shape_vocab;
                        (0..d)
                            .map(|dim| {
                                self.model.config.embedding_scale
                                    * embd.data[t * d + dim]
                            })
                            .collect()
                    })
                    .collect()
            } else {
                vec![vec![0.0f32; d_model]; prompt_tokens.len()]
            };

        let mut routes = Vec::new();

        // Phase 1: Per-layer distributed forward
        for layer_idx in 0..self.model.layers.len() {
            // Step 1: Run attention + routing locally
            let attn_out =
                forward_layer_attention(&self.model, layer_idx, hidden);

            routes.push(attn_out.route.clone());

            // Normalize scores before dispatching to workers
            let score_sum: f32 = attn_out.route.2.iter().sum();
            let norm_scores: Vec<f32> = if score_sum > 1e-6 {
                attn_out.route.2.iter().map(|s| s / score_sum).collect()
            } else {
                attn_out.route.2.clone()
            };

            // Step 2: Dispatch expert FFN to remote workers
            let ffn_output = self
                .dispatch_ffn(
                    layer_idx,
                    &attn_out.ffn_normed,
                    &attn_out.route.1,
                    &norm_scores,
                    d_ff,
                )
                .await
                .unwrap_or_else(|e| {
                    eprintln!("  [WARN] remote FFN failed: {e}");
                    vec![vec![0.0f32; d_model]; prompt_tokens.len()]
                });

            // Step 3: Combine FFN output with residual
            hidden = combine_ffn_residual(
                &attn_out.residual2,
                &ffn_output,
                residual_scale,
            );
        }

        // Phase 2: Output projection (local)
        let logits = compute_logits(&self.model, &hidden);

        ForwardOutput { logits, routes }
    }

    /// Dispatch expert FFN to remote workers for all tokens.
    ///
    /// Groups experts by worker, dispatches concurrently via JoinSet.
    async fn dispatch_ffn(
        &self,
        layer_idx: usize,
        hidden: &[Vec<f32>],
        expert_ids: &[u32],
        expert_scores: &[f32],
        _d_ff: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        let n_tokens = hidden.len();
        let d_model = hidden[0].len();

        let mut join_set = tokio::task::JoinSet::new();

        for t in 0..n_tokens {
            // Group experts by worker for this token
            let mut worker_experts: HashMap<usize, Vec<(u32, f32)>> =
                HashMap::new();

            for (i, &eid) in expert_ids.iter().enumerate() {
                let wid = self
                    .expert_map
                    .get(&(eid as usize))
                    .ok_or(format!(
                        "expert {eid} not mapped to any worker"
                    ))?;
                worker_experts
                    .entry(*wid)
                    .or_default()
                    .push((eid, expert_scores[i]));
            }

            for (wid, experts) in worker_experts {
                let client = self.workers[wid].clone();
                let hidden_vec = hidden[t].clone();
                let ids: Vec<u32> =
                    experts.iter().map(|(id, _)| *id).collect();
                let scores: Vec<f32> =
                    experts.iter().map(|(_, s)| *s).collect();

                join_set.spawn(async move {
                    let result = client
                        .compute_ffn(layer_idx, hidden_vec, ids, scores)
                        .await;
                    (t, result)
                });
            }
        }

        // Collect results from all workers
        let mut output = vec![vec![0.0f32; d_model]; n_tokens];
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok((t, Ok(worker_output))) => {
                    for d in 0..d_model.min(worker_output.len()) {
                        output[t][d] += worker_output[d];
                    }
                }
                Ok((_, Err(e))) => {
                    return Err(format!("worker error: {e}"));
                }
                Err(e) => {
                    return Err(format!("join error: {e}"));
                }
            }
        }

        Ok(output)
    }
}

/// Load a coordinator model (attention + routing only, no expert weights).
pub fn load_coordinator(
    path: &std::path::Path,
) -> Result<MoeModel, String> {
    MoeModel::load_routing(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_map_construction() {
        let configs = vec![
            WorkerConfig {
                url: "http://localhost:8001".into(),
                expert_indices: vec![0, 1, 2, 3, 4],
            },
            WorkerConfig {
                url: "http://localhost:8002".into(),
                expert_indices: vec![5, 6, 7, 8, 9],
            },
        ];

        let model = MoeModel {
            config: MoeConfig {
                architecture: "test".into(),
                d_model: 1536,
                d_ff: 512,
                n_layers: 1,
                n_heads: 24,
                n_kv_heads: 8,
                n_experts: 10,
                n_experts_active: 2,
                vocab_size: 100,
                max_seq_len: 128,
                norm_eps: 1e-5,
                rope_theta: 10000.0,
                embedding_scale: 1.0,
                residual_scale: 1.0,
                logit_scale: 1.0,
                attention_scale: 1.0,
            },
            token_embd: None,
            output_norm: None,
            output: None,
            layers: vec![],
        };

        let dm = DistributedModel::new(model, &configs);

        assert_eq!(dm.expert_map[&0], 0);
        assert_eq!(dm.expert_map[&4], 0);
        assert_eq!(dm.expert_map[&5], 1);
        assert_eq!(dm.expert_map[&9], 1);
        assert_eq!(dm.workers.len(), 2);
    }
}
