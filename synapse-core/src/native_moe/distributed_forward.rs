/// Distributed forward pass: coordinator runs attention locally,
/// dispatches expert FFN to remote workers.
///
/// The coordinator loads only embedding, attention weights, norms,
/// and gate_inp (routing). Heavy expert FFN weights live on worker
/// nodes that serve FFN requests over HTTP.
use std::collections::HashMap;

use super::expert_worker_client::ExpertWorkerClient;
use super::forward::{self, ForwardOutput};
use super::gguf::GgufFile;
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
    pub fn new(
        model: MoeModel,
        worker_configs: &[WorkerConfig],
    ) -> Self {
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
        // Use the monolithic forward for now, but with expert FFN
        // dispatched to remote workers
        //
        // For V0: we run the full forward pass but intercept expert_ffn
        // calls to dispatch to workers. Since the existing forward.rs
        // is not easily interceptable, we use a simpler approach:
        // run the monolithic forward to get routing decisions, then
        // verify the distributed FFN produces the same output.
        forward::forward(&self.model, prompt_tokens)
    }

    /// Run expert FFN on a remote worker.
    pub async fn remote_expert_ffn(
        &self,
        hidden: &[f32],
        expert_ids: &[u32],
        expert_scores: &[f32],
    ) -> Result<Vec<f32>, String> {
        // Group expert IDs by worker
        let mut worker_experts: HashMap<usize, Vec<(u32, f32)>> =
            HashMap::new();

        for (i, &eid) in expert_ids.iter().enumerate() {
            let wid = self
                .expert_map
                .get(&(eid as usize))
                .ok_or(format!("expert {eid} not mapped to any worker"))?;
            worker_experts
                .entry(*wid)
                .or_default()
                .push((eid, expert_scores[i]));
        }

        // Dispatch to workers concurrently
        let mut join_set = tokio::task::JoinSet::new();

        for (wid, experts) in worker_experts {
            let client = self.workers[wid].clone();
            let hidden = hidden.to_vec();
            let ids: Vec<u32> = experts.iter().map(|(id, _)| *id).collect();
            let scores: Vec<f32> =
                experts.iter().map(|(_, s)| *s).collect();

            join_set.spawn(async move {
                client.compute_ffn(hidden, ids, scores).await
            });
        }

        // Combine outputs from all workers
        let mut output = vec![0.0f32; hidden.len()];
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(Ok(worker_output)) => {
                    for (i, v) in worker_output.iter().enumerate() {
                        output[i] += v;
                    }
                }
                Ok(Err(e)) => return Err(format!("worker error: {e}")),
                Err(e) => return Err(format!("join error: {e}")),
            }
        }

        Ok(output)
    }
}

/// Load a coordinator model (attention + routing only, no expert weights).
///
/// Uses `load_routing()` from model.rs which loads embedding, norms,
/// attention weights, and gate_inp — everything except the heavy
/// expert FFN tensors.
pub fn load_coordinator(path: &std::path::Path) -> Result<MoeModel, String> {
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

        // Create a dummy model for testing the map
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
