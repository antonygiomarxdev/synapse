use std::sync::Arc;

use axum::{Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};

use crate::job::job::{Job, Message as JobMessage, Priority};
use crate::job::ports::JobStore;
use crate::scheduler::scheduler::Scheduler;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default = "default_priority")]
    pub priority: String,
    #[serde(default = "default_swarm_size")]
    pub swarm_size: u32,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
}

/// Error response body.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

fn default_priority() -> String {
    "normal".into()
}
fn default_swarm_size() -> u32 {
    5
}

/// Handles OpenAI-compatible chat completion requests.
///
/// Submits the request as an async job to the scheduler, waits for completion,
/// and returns the result in OpenAI format.
pub async fn chat_completions(
    State(state): State<super::jobs::AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, Json<ErrorResponse>)> {
    let priority = match req.priority.parse::<Priority>() {
        Ok(p) => p,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse { error: format!("invalid priority: {e}") }),
            ));
        }
    };

    // Convert messages to job format
    let messages: Vec<JobMessage> = req
        .messages
        .into_iter()
        .map(|m| JobMessage { role: m.role, content: m.content })
        .collect();

    // Create and save job
    let job = Job::submit(req.model.clone(), messages, priority).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    let job_id = job.id;
    state.job_store.save(&job).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;

    // Process job if scheduler is available
    if let Some(scheduler) = &state.scheduler {
        let scheduler = scheduler.clone();
        let messages = job.messages.clone();
        let model = job.model.clone();

        // Decompose and process
        let now = chrono::Utc::now();
        scheduler.decompose(&job_id, &messages, &model, now).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: format!("failed to decompose job: {e}") }),
            )
        })?;

        // Process tasks with retry loop
        let max_ticks = 10;
        for _ in 0..max_ticks {
            match scheduler.tick(now).await {
                Ok(0) => break,
                Ok(_) => {}
                Err(e) => {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse { error: format!("scheduler error: {e}") }),
                    ));
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    // Get completed job
    let job = state
        .job_store
        .find_by_id(&job_id)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e.to_string() }),
            )
        })?
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse { error: "job not found".into() }),
            )
        })?;

    // Build response
    let content = job
        .result
        .as_ref()
        .map(|r| r.text.clone())
        .unwrap_or_else(|| "No result".into());

    Ok(Json(ChatResponse {
        id: format!("chatcmpl-{job_id}"),
        object: "chat.completion".into(),
        created: chrono::Utc::now().timestamp() as u64,
        model: job.model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".into(),
                content,
            },
            finish_reason: "stop".into(),
        }],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::infrastructure::InMemoryJobStore;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_state() -> super::super::jobs::AppState {
        super::super::jobs::AppState {
            job_store: Arc::new(InMemoryJobStore::new()),
            scheduler: None,
        }
    }

    async fn test_app() -> axum::Router {
        let state = test_state();
        axum::Router::new()
            .route("/v1/chat/completions", axum::routing::post(chat_completions))
            .with_state(state)
    }

    #[tokio::test]
    async fn chat_completions_returns_200() {
        let app = test_app().await;
        let body = serde_json::json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        // Without scheduler, job is created but not processed
        // Returns 200 with "No result" content
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn chat_completions_returns_job_id_in_response() {
        let app = test_app().await;
        let body = serde_json::json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let resp_body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let chat: ChatResponse = serde_json::from_slice(&resp_body).unwrap();
        assert!(chat.id.starts_with("chatcmpl-"));
    }

    #[tokio::test]
    async fn chat_completions_returns_model() {
        let app = test_app().await;
        let body = serde_json::json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let resp_body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let chat: ChatResponse = serde_json::from_slice(&resp_body).unwrap();
        assert_eq!(chat.model, "kimi-k3");
    }

    #[tokio::test]
    async fn defaults_applied_without_explicit_fields() {
        let app = test_app().await;
        let body = serde_json::json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}

#[cfg(test)]
mod default_tests {
    use super::*;

    #[test]
    fn default_priority_is_normal() {
        assert_eq!(default_priority(), "normal");
    }

    #[test]
    fn default_swarm_size_is_5() {
        assert_eq!(default_swarm_size(), 5);
    }

    #[test]
    fn default_priority_is_not_empty() {
        assert!(!default_priority().is_empty());
    }

    #[test]
    fn default_swarm_size_is_nonzero() {
        assert!(default_swarm_size() > 0);
    }
}
