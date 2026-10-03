/// HTTP client for communicating with expert worker nodes.
///
/// Sends hidden state + routing info to a worker, receives FFN output.
use serde::{Deserialize, Serialize};

use crate::shared::DomainError;

/// A single token's hidden state and expert routing info for FFN computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfnRow {
    /// Hidden state vector (length d_model).
    pub hidden: Vec<f32>,
    /// Expert IDs to route to.
    pub expert_ids: Vec<u32>,
    /// Normalized expert scores (should sum ≈ 1.0).
    pub expert_scores: Vec<f32>,
}

/// Request to compute FFN for multiple rows in one batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfnRequest {
    /// Layer index.
    pub layer: usize,
    /// Multiple rows to compute FFN for.
    pub rows: Vec<FfnRow>,
}

/// Response from FFN computation for a batch of rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfnResponse {
    /// One output vector per row (length d_model each), same order as request.
    pub outputs: Vec<Vec<f32>>,
}

/// Client for a remote expert worker.
#[derive(Clone)]
pub struct ExpertWorkerClient {
    base_url: String,
    client: reqwest::Client,
}

impl ExpertWorkerClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    /// Check if the worker is healthy.
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        self.client.get(&url).send().await.map(|r| r.status().is_success()).unwrap_or(false)
    }

    /// Send a batch of rows for one layer; returns one output per row, same order.
    pub async fn compute_ffn_batch(
        &self,
        layer: usize,
        rows: Vec<FfnRow>,
    ) -> Result<Vec<Vec<f32>>, DomainError> {
        let url = format!("{}/ffn", self.base_url);
        let n_rows = rows.len();
        let req = FfnRequest { layer, rows };

        let resp = self
            .client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| DomainError::WorkerDispatchFailed {
                reason: format!("HTTP request failed: {e}"),
            })?
            .json::<FfnResponse>()
            .await
            .map_err(|e| DomainError::WorkerDispatchFailed {
                reason: format!("failed to parse response: {e}"),
            })?;

        if resp.outputs.len() != n_rows {
            return Err(DomainError::WorkerDispatchFailed {
                reason: format!("expected {n_rows} outputs, got {}", resp.outputs.len()),
            });
        }

        Ok(resp.outputs)
    }
}
