use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::scheduler::ports::WorkerPort;
use crate::scheduler::task::Task;
use crate::scheduler::worker_id::WorkerId;
use crate::shared::DomainError;

/// Configuration for a single Ollama worker.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub id: WorkerId,
    pub model: String,
    pub base_url: String,
}

/// Real worker port that dispatches tasks to Ollama via HTTP.
///
/// Maps `WorkerId` to an Ollama model and sends inference requests
/// to `POST /api/generate`.
pub struct OllamaWorkerPort {
    client: reqwest::blocking::Client,
    workers: HashMap<WorkerId, WorkerConfig>,
}

/// Request body for Ollama's `/api/generate`.
#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

/// Response from Ollama's `/api/generate` (non-streaming).
#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

impl OllamaWorkerPort {
    /// Creates a new OllamaWorkerPort with the given worker configurations.
    pub fn new(configs: Vec<WorkerConfig>) -> Self {
        let workers: HashMap<WorkerId, WorkerConfig> =
            configs.into_iter().map(|c| (c.id.clone(), c)).collect();
        Self {
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("failed to build HTTP client"),
            workers,
        }
    }
}

impl WorkerPort for OllamaWorkerPort {
    fn dispatch(&self, worker_id: &WorkerId, task: &Task) -> Result<String, DomainError> {
        let config = self.workers.get(worker_id).ok_or_else(|| DomainError::WorkerDispatchFailed {
            reason: format!("unknown worker: {worker_id}"),
        })?;

        let url = format!("{}/api/generate", config.base_url);
        let body = OllamaRequest {
            model: config.model.clone(),
            prompt: task.message.content.clone(),
            stream: false,
        };

        let response = self.client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| DomainError::WorkerDispatchFailed {
                reason: format!("HTTP request failed: {e}"),
            })?
            .json::<OllamaResponse>()
            .map_err(|e| DomainError::WorkerDispatchFailed {
                reason: format!("failed to parse response: {e}"),
            })?;

        Ok(response.response)
    }

    fn health_check(&self, worker_id: &WorkerId) -> Result<bool, DomainError> {
        let config = self.workers.get(worker_id).ok_or_else(|| DomainError::WorkerDispatchFailed {
            reason: format!("unknown worker: {worker_id}"),
        })?;

        let url = format!("{}/api/tags", config.base_url);
        match self.client.get(&url).send() {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::job::Message;
    use crate::job::job_id::JobId;
    use chrono::Utc;

    fn test_task() -> Task {
        Task::new(
            JobId::new(),
            "granite3.1-moe:3b".into(),
            Message { role: "user".into(), content: "Say hello in one word.".into() },
            Utc::now(),
        )
    }

    #[test]
    fn health_check_localhost() {
        let port = OllamaWorkerPort::new(vec![WorkerConfig {
            id: WorkerId::new("w-0"),
            model: "granite3.1-moe:3b".into(),
            base_url: "http://localhost:11434".into(),
        }]);

        let healthy = port.health_check(&WorkerId::new("w-0")).unwrap();
        assert!(healthy);
    }

    #[test]
    fn health_check_unknown_worker() {
        let port = OllamaWorkerPort::new(vec![]);
        let result = port.health_check(&WorkerId::new("unknown"));
        assert!(matches!(result, Err(DomainError::WorkerDispatchFailed { .. })));
    }
}
