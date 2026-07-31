/// HTTP client for communicating with expert worker nodes.
///
/// Sends hidden state + routing info to a worker, receives FFN output.
use serde::{Deserialize, Serialize};

use crate::shared::DomainError;

#[derive(Debug, Deserialize)]
struct FfnResponse {
    output: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct FfnRequest {
    layer: usize,
    hidden: Vec<f32>,
    expert_ids: Vec<u32>,
    expert_scores: Vec<f32>,
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
        self.client
            .get(&url)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Send hidden state + routing to worker, get FFN output back.
    pub async fn compute_ffn(
        &self,
        layer: usize,
        hidden: Vec<f32>,
        expert_ids: Vec<u32>,
        expert_scores: Vec<f32>,
    ) -> Result<Vec<f32>, DomainError> {
        let url = format!("{}/ffn", self.base_url);
        let req = FfnRequest {
            layer,
            hidden,
            expert_ids,
            expert_scores,
        };

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

        Ok(resp.output)
    }
}
