#![allow(clippy::needless_range_loop)]

/// Distributed forward pass: coordinator runs attention locally,
/// dispatches expert FFN to remote workers.
///
/// The coordinator loads only embedding, attention weights, norms,
/// and gate_inp (routing). Heavy expert FFN weights live on worker
/// nodes that serve FFN requests over HTTP.
use std::collections::HashMap;

use super::expert_worker_client::{ExpertWorkerClient, FfnRow};
use super::forward::{
    ForwardOutput, LayerAttentionOutput, combine_ffn_residual, compute_logits,
    forward_layer_attention,
};
use super::model::MoeModel;

/// Rows assigned to one worker, plus each row's index in the original batch.
#[derive(Debug)]
pub struct WorkerBatch {
    /// Rows to compute FFN for.
    pub rows: Vec<FfnRow>,
    /// Index of each row in the original batch.
    pub row_index: Vec<usize>,
}

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
        let workers: Vec<ExpertWorkerClient> =
            worker_configs.iter().map(|c| ExpertWorkerClient::new(c.url.clone())).collect();

        let mut expert_map = HashMap::new();
        for (wid, config) in worker_configs.iter().enumerate() {
            for &eid in &config.expert_indices {
                expert_map.insert(eid, wid);
            }
        }

        DistributedModel { model, workers, expert_map }
    }

    /// Run a forward pass for many independent sequences.
    ///
    /// Attention runs per sequence locally; expert FFN for all sequences
    /// is batched into one request per worker per layer.
    pub async fn forward_batch(&self, seqs: &[Vec<u32>]) -> Vec<ForwardOutput> {
        let d_model = self.model.config.d_model as usize;
        let residual_scale = self.model.config.residual_scale;

        // Phase 0: Embedding lookup per sequence (local)
        let mut seq_hiddens: Vec<Vec<Vec<f32>>> = seqs
            .iter()
            .map(|prompt_tokens| {
                if let Some(ref embd) = self.model.token_embd {
                    let d = embd.shape[0] as usize;
                    let shape_vocab = embd.shape[1] as usize;
                    prompt_tokens
                        .iter()
                        .map(|&tid| {
                            let t = tid as usize % shape_vocab;
                            (0..d)
                                .map(|dim| {
                                    self.model.config.embedding_scale * embd.data[t * d + dim]
                                })
                                .collect()
                        })
                        .collect()
                } else {
                    vec![vec![0.0f32; d_model]; prompt_tokens.len()]
                }
            })
            .collect();

        let mut seq_routes: Vec<Vec<_>> = vec![Vec::new(); seqs.len()];

        // Phase 1: Per-layer distributed forward
        for layer_idx in 0..self.model.layers.len() {
            // Step 1: Run attention + routing locally per sequence
            let mut attn_outs: Vec<LayerAttentionOutput> = Vec::new();
            for seq_idx in 0..seqs.len() {
                let hidden = std::mem::take(&mut seq_hiddens[seq_idx]);
                let attn_out = forward_layer_attention(&self.model, layer_idx, hidden);
                seq_routes[seq_idx].push(attn_out.route.clone());
                attn_outs.push(attn_out);
            }

            // Step 2: Collect all rows from all sequences with their routes
            let mut all_rows: Vec<(Vec<f32>, Vec<u32>, Vec<f32>)> = Vec::new();
            let mut seq_offsets: Vec<usize> = vec![0];

            for attn_out in &attn_outs {
                // Normalize scores for this sequence
                let score_sum: f32 = attn_out.route.2.iter().sum();
                let norm_scores: Vec<f32> = if score_sum > 1e-6 {
                    attn_out.route.2.iter().map(|s| s / score_sum).collect()
                } else {
                    attn_out.route.2.clone()
                };

                // Create rows for each token in this sequence
                for hidden_state in &attn_out.ffn_normed {
                    all_rows.push((
                        hidden_state.clone(),
                        attn_out.route.1.clone(),
                        norm_scores.clone(),
                    ));
                }

                seq_offsets.push(all_rows.len());
            }

            // Step 3: Dispatch all rows in one batch per worker
            let ffn_outputs = self.dispatch_ffn(layer_idx, &all_rows).await.unwrap_or_else(|e| {
                eprintln!("  [WARN] remote FFN failed: {e}");
                vec![vec![0.0f32; d_model]; all_rows.len()]
            });

            // Step 4: Split outputs back per sequence and combine with residual
            for seq_idx in 0..seqs.len() {
                let start = seq_offsets[seq_idx];
                let end = seq_offsets[seq_idx + 1];
                let seq_ffn_outputs = ffn_outputs[start..end].to_vec();
                seq_hiddens[seq_idx] = combine_ffn_residual(
                    &attn_outs[seq_idx].residual2,
                    &seq_ffn_outputs,
                    residual_scale,
                );
            }
        }

        // Phase 2: Output projection (local) per sequence
        let mut outputs = Vec::new();
        for (seq_idx, hidden) in seq_hiddens.into_iter().enumerate() {
            let logits = compute_logits(&self.model, &hidden);
            outputs.push(ForwardOutput { logits, routes: seq_routes[seq_idx].clone() });
        }

        outputs
    }

    /// Run distributed forward pass on prompt tokens.
    ///
    /// For each layer:
    /// 1. Run attention locally (coordinator)
    /// 2. Route experts via gate_inp
    /// 3. Dispatch expert FFN to remote workers (concurrent)
    /// 4. Combine results and continue
    pub async fn forward(&self, prompt_tokens: &[u32]) -> ForwardOutput {
        self.forward_batch(&[prompt_tokens.to_vec()]).await.remove(0)
    }

    /// Dispatch expert FFN to remote workers for all rows.
    ///
    /// Groups rows by worker (each row has its own route), dispatches
    /// one batched request per worker concurrently via JoinSet.
    async fn dispatch_ffn(
        &self,
        layer_idx: usize,
        rows: &[(Vec<f32>, Vec<u32>, Vec<f32>)],
    ) -> Result<Vec<Vec<f32>>, String> {
        let d_model = self.model.config.d_model as usize;
        let n_rows = rows.len();

        // Group rows by worker
        let batches = group_rows_by_worker(&self.expert_map, rows)?;

        let mut join_set = tokio::task::JoinSet::new();

        // Spawn one task per worker
        for (wid, batch) in batches.into_iter() {
            let client = self.workers[wid].clone();
            let row_indices = batch.row_index.clone();

            join_set.spawn(async move {
                let result = client.compute_ffn_batch(layer_idx, batch.rows).await;
                (row_indices, result)
            });
        }

        // Collect results from all workers
        let mut output = vec![vec![0.0f32; d_model]; n_rows];
        while let Some(task_result) = join_set.join_next().await {
            match task_result {
                Ok((row_indices, Ok(worker_outputs))) => {
                    for (j, out) in worker_outputs.iter().enumerate() {
                        for d in 0..d_model.min(out.len()) {
                            output[row_indices[j]][d] += out[d];
                        }
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

/// Group rows by worker: each row's experts are split by `expert_map`
/// (expert id -> worker index). A row appears in a worker's batch only with
/// that worker's experts/scores. Errors if an expert id is unmapped.
pub fn group_rows_by_worker(
    expert_map: &HashMap<usize, usize>,
    rows: &[(Vec<f32>, Vec<u32>, Vec<f32>)],
) -> Result<HashMap<usize, WorkerBatch>, String> {
    let mut batches: HashMap<usize, WorkerBatch> = HashMap::new();

    for (row_idx, (hidden, expert_ids, expert_scores)) in rows.iter().enumerate() {
        // Group this row's experts by worker
        let mut row_experts: HashMap<usize, Vec<(u32, f32)>> = HashMap::new();

        for (exp_i, &eid) in expert_ids.iter().enumerate() {
            let wid = expert_map
                .get(&(eid as usize))
                .ok_or(format!("expert {eid} not mapped to any worker"))?;
            row_experts.entry(*wid).or_default().push((eid, expert_scores[exp_i]));
        }

        // Add this row to each worker's batch (only for workers with experts from this row)
        for (wid, experts) in row_experts {
            let batch = batches
                .entry(wid)
                .or_insert_with(|| WorkerBatch { rows: Vec::new(), row_index: Vec::new() });
            let expert_ids_for_worker: Vec<u32> = experts.iter().map(|(id, _)| *id).collect();
            let expert_scores_for_worker: Vec<f32> = experts.iter().map(|(_, s)| *s).collect();
            batch.rows.push(FfnRow {
                hidden: hidden.clone(),
                expert_ids: expert_ids_for_worker,
                expert_scores: expert_scores_for_worker,
            });
            batch.row_index.push(row_idx);
        }
    }

    Ok(batches)
}

/// Load a coordinator model (attention + routing only, no expert weights).
pub fn load_coordinator(path: &std::path::Path) -> Result<MoeModel, String> {
    MoeModel::load_routing(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_moe::model::MoeConfig;

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

    #[test]
    fn group_rows_two_workers_different_experts() {
        let mut expert_map = HashMap::new();
        expert_map.insert(0, 0);
        expert_map.insert(1, 0);
        expert_map.insert(2, 1);
        expert_map.insert(3, 1);

        let hidden0 = vec![1.0, 2.0, 3.0];
        let hidden1 = vec![4.0, 5.0, 6.0];
        let rows = vec![
            (hidden0.clone(), vec![0, 2], vec![0.3, 0.7]),
            (hidden1.clone(), vec![1, 3], vec![0.4, 0.6]),
        ];

        let result = group_rows_by_worker(&expert_map, &rows).unwrap();

        // Worker 0 should have both rows, one with expert 0, one with expert 1
        assert!(result.contains_key(&0));
        let batch0 = &result[&0];
        assert_eq!(batch0.rows.len(), 2);
        assert_eq!(batch0.row_index.len(), 2);
        assert_eq!(batch0.row_index, vec![0, 1]);
        assert_eq!(batch0.rows[0].expert_ids, vec![0]);
        assert_eq!(batch0.rows[0].expert_scores, vec![0.3]);
        assert_eq!(batch0.rows[1].expert_ids, vec![1]);
        assert_eq!(batch0.rows[1].expert_scores, vec![0.4]);

        // Worker 1 should have both rows, one with expert 2, one with expert 3
        assert!(result.contains_key(&1));
        let batch1 = &result[&1];
        assert_eq!(batch1.rows.len(), 2);
        assert_eq!(batch1.row_index.len(), 2);
        assert_eq!(batch1.row_index, vec![0, 1]);
        assert_eq!(batch1.rows[0].expert_ids, vec![2]);
        assert_eq!(batch1.rows[0].expert_scores, vec![0.7]);
        assert_eq!(batch1.rows[1].expert_ids, vec![3]);
        assert_eq!(batch1.rows[1].expert_scores, vec![0.6]);
    }

    #[test]
    fn group_rows_unmapped_expert() {
        let expert_map = HashMap::new(); // Empty: no experts mapped

        let hidden = vec![1.0, 2.0];
        let rows = vec![(hidden, vec![5], vec![1.0])];

        let result = group_rows_by_worker(&expert_map, &rows);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("expert 5 not mapped"));
    }
}
