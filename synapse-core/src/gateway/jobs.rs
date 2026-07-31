use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::job::job::{Job, Message, Priority};
use crate::job::job_id::JobId;
use crate::job::ports::JobStore;
use crate::scheduler::scheduler::Scheduler;

/// Shared application state injected into handlers.
#[derive(Clone)]
pub struct AppState {
    pub job_store: Arc<dyn JobStore>,
    pub scheduler: Option<Arc<Scheduler>>,
}

// --- Request types ---

/// An OpenAI-compatible message in a chat request.
#[derive(Debug, Deserialize, ToSchema)]
pub struct MessageRequest {
    /// Role of the message author (e.g., "user", "system", "assistant").
    pub role: String,
    /// Content of the message.
    pub content: String,
}

/// Request body for `POST /v1/jobs`.
#[derive(Debug, Deserialize, ToSchema)]
pub struct CreateJobRequest {
    /// Model to use for inference (e.g., "granite-3b-moe").
    pub model: String,
    /// Messages in OpenAI format.
    pub messages: Vec<MessageRequest>,
    /// Scheduling priority. Defaults to "normal".
    #[serde(default)]
    pub priority: Option<String>,
}

// --- Response types ---

/// Response for `POST /v1/jobs`.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CreateJobResponse {
    /// Unique job identifier.
    pub job_id: String,
}

/// Response for `GET /v1/jobs/{id}`.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct JobResponse {
    /// Unique job identifier.
    pub job_id: String,
    /// Object type (always "job").
    pub object: String,
    /// Current status: pending, running, completed, or failed.
    pub status: String,
    /// Model used for inference.
    pub model: String,
    /// Result if completed.
    pub result: Option<JobResultResponse>,
    /// Error message if failed.
    pub error: Option<String>,
    /// ISO 8601 creation timestamp.
    pub created_at: String,
    /// ISO 8601 last update timestamp.
    pub updated_at: String,
}

/// Result of a completed job.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct JobResultResponse {
    /// Generated text.
    pub text: String,
    /// Number of tokens generated.
    pub tokens: u32,
}

/// Error response body.
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    /// Error message.
    pub error: String,
}

// --- Handlers ---

/// Submit a new inference job.
#[utoipa::path(
    post,
    path = "/v1/jobs",
    request_body = CreateJobRequest,
    responses(
        (status = 202, description = "Job accepted", body = CreateJobResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
    ),
    tag = "jobs"
)]
pub async fn create_job(
    State(state): State<AppState>,
    Json(req): Json<CreateJobRequest>,
) -> impl IntoResponse {
    let priority = match req.priority.as_deref() {
        Some(p) => match p.parse::<Priority>() {
            Ok(pr) => pr,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse { error: format!("invalid priority: {e}") }),
                )
                    .into_response();
            }
        },
        None => Priority::Normal,
    };

    let messages: Vec<Message> = req
        .messages
        .into_iter()
        .map(|m| Message { role: m.role, content: m.content })
        .collect();

    let job = match Job::submit(req.model, messages, priority) {
        Ok(job) => job,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse { error: e.to_string() }),
            )
                .into_response();
        }
    };

    let job_id = job.id;

    if let Err(e) = state.job_store.save(&job) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
            .into_response();
    }

    // Trigger scheduler to process the job if available.
    //
    // Spawns a background task that:
    // 1. Decomposes the job into individual tasks (one per message)
    // 2. Dispatches tasks to workers via the scheduler
    // 3. Retries failed tasks up to the scheduler's max retry limit
    // 4. Completes or fails the job based on task outcomes
    //
    // Errors are logged via `tracing::error!` and the job transitions to Failed state.
    if let Some(scheduler) = &state.scheduler {
        let scheduler = scheduler.clone();
        let job_id = job.id;
        let messages = job.messages.clone();
        let model = job.model.clone();

        tokio::spawn(async move {
            let now = chrono::Utc::now();

            // Decompose job into tasks
            if let Err(e) = scheduler.decompose(&job_id, &messages, &model, now) {
                tracing::error!(job_id = %job_id, error = %e, "Failed to decompose job");
                // Mark job as failed
                if let Err(e2) = scheduler.job_store.fail(&job_id, e.to_string()) {
                    tracing::error!(job_id = %job_id, error = %e2, "Failed to mark job as failed");
                }
                return;
            }

            // Process tasks with retry loop
            let max_ticks = 10; // Prevent infinite loops
            for tick_num in 0..max_ticks {
                match scheduler.tick(now).await {
                    Ok(0) => {
                        // No tasks processed, job might be complete
                        tracing::debug!(job_id = %job_id, tick = tick_num, "No tasks to process");
                        break;
                    }
                    Ok(processed) => {
                        tracing::debug!(job_id = %job_id, tick = tick_num, tasks = processed, "Processed tasks");
                    }
                    Err(e) => {
                        tracing::error!(job_id = %job_id, tick = tick_num, error = %e, "Scheduler tick failed");
                        // Mark job as failed on scheduler error
                        if let Err(e2) = scheduler.job_store.fail(&job_id, e.to_string()) {
                            tracing::error!(job_id = %job_id, error = %e2, "Failed to mark job as failed");
                        }
                        return;
                    }
                }

                // Small delay between ticks to prevent busy-waiting
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }

            tracing::info!(job_id = %job_id, "Job processing completed");
        });
    }

    (
        StatusCode::ACCEPTED,
        Json(CreateJobResponse { job_id: job_id.to_string() }),
    )
        .into_response()
}

/// Get job status and result.
#[utoipa::path(
    get,
    path = "/v1/jobs/{id}",
    params(
        ("id" = String, Path, description = "Job ID (UUID)")
    ),
    responses(
        (status = 200, description = "Job found", body = JobResponse),
        (status = 400, description = "Invalid job ID", body = ErrorResponse),
        (status = 404, description = "Job not found", body = ErrorResponse),
    ),
    tag = "jobs"
)]
pub async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let job_id: JobId = match id.parse() {
        Ok(id) => id,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse { error: format!("invalid job id: {e}") }),
            )
                .into_response();
        }
    };

    let job = match state.job_store.find_by_id(&job_id) {
        Ok(Some(job)) => job,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse { error: format!("job not found: {job_id}") }),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: e.to_string() }),
            )
                .into_response();
        }
    };

    (StatusCode::OK, Json(job_to_response(&job))).into_response()
}

fn job_to_response(job: &Job) -> JobResponse {
    JobResponse {
        job_id: job.id.to_string(),
        object: "job".into(),
        status: job.status.to_string(),
        model: job.model.clone(),
        result: job.result.as_ref().map(|r| JobResultResponse {
            text: r.text.clone(),
            tokens: r.tokens,
        }),
        error: job.error.clone(),
        created_at: job.created_at.to_rfc3339(),
        updated_at: job.updated_at.to_rfc3339(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::infrastructure::InMemoryJobStore;
    use axum::body::Body;
    use axum::http::{self, Request};
    use tower::ServiceExt;

    fn test_app() -> axum::Router {
        let state = AppState {
            job_store: Arc::new(InMemoryJobStore::new()),
            scheduler: None,
        };
        axum::Router::new()
            .route("/v1/jobs", axum::routing::post(create_job))
            .route("/v1/jobs/{id}", axum::routing::get(get_job))
            .with_state(state)
    }

    fn valid_request_body() -> String {
        serde_json::json!({
            "model": "granite-3b-moe",
            "messages": [{ "role": "user", "content": "Hello" }]
        })
        .to_string()
    }

    #[tokio::test]
    async fn create_job_returns_202() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method(http::Method::POST)
                    .uri("/v1/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(valid_request_body()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::ACCEPTED);
    }

    #[tokio::test]
    async fn create_job_returns_job_id() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method(http::Method::POST)
                    .uri("/v1/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(valid_request_body()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let resp: CreateJobResponse = serde_json::from_slice(&body).unwrap();
        assert!(!resp.job_id.is_empty());
        assert!(uuid::Uuid::parse_str(&resp.job_id).is_ok());
    }

    #[tokio::test]
    async fn create_job_rejects_empty_model() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method(http::Method::POST)
                    .uri("/v1/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model": "",
                            "messages": [{ "role": "user", "content": "Hello" }]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn create_job_rejects_empty_messages() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method(http::Method::POST)
                    .uri("/v1/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model": "granite-3b-moe",
                            "messages": []
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn get_job_returns_pending_status() {
        let state = AppState {
            job_store: Arc::new(InMemoryJobStore::new()),
            scheduler: None,
        };

        // Create a job first
        let job = Job::submit(
            "granite-3b-moe".into(),
            vec![Message { role: "user".into(), content: "hi".into() }],
            Priority::Normal,
        )
        .unwrap();
        let job_id = job.id.to_string();
        state.job_store.save(&job).unwrap();

        let app = axum::Router::new()
            .route("/v1/jobs/{id}", axum::routing::get(get_job))
            .with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/jobs/{job_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let resp: JobResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(resp.status, "pending");
        assert_eq!(resp.object, "job");
        assert_eq!(resp.model, "granite-3b-moe");
        assert!(resp.result.is_none());
    }

    #[tokio::test]
    async fn get_job_returns_404_for_unknown() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/jobs/{}", uuid::Uuid::new_v4()))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_job_returns_400_for_invalid_id() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/jobs/not-a-uuid")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn end_to_end_submit_and_poll() {
        let state = AppState {
            job_store: Arc::new(InMemoryJobStore::new()),
            scheduler: None,
        };

        let app = axum::Router::new()
            .route("/v1/jobs", axum::routing::post(create_job))
            .route("/v1/jobs/{id}", axum::routing::get(get_job))
            .with_state(state);

        // Submit
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(http::Method::POST)
                    .uri("/v1/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(valid_request_body()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let created: CreateJobResponse = serde_json::from_slice(&body).unwrap();

        // Poll
        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/jobs/{}", created.job_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let job: JobResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(job.status, "pending");
        assert_eq!(job.job_id, created.job_id);
    }

    #[tokio::test]
    async fn create_job_with_priority() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method(http::Method::POST)
                    .uri("/v1/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model": "granite-3b-moe",
                            "messages": [{ "role": "user", "content": "Hello" }],
                            "priority": "high"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::ACCEPTED);
    }

    #[tokio::test]
    async fn create_job_defaults_priority_to_normal() {
        let state = AppState {
            job_store: Arc::new(InMemoryJobStore::new()),
            scheduler: None,
        };

        let app = axum::Router::new()
            .route("/v1/jobs", axum::routing::post(create_job))
            .route("/v1/jobs/{id}", axum::routing::get(get_job))
            .with_state(state.clone());

        // Submit without priority
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(http::Method::POST)
                    .uri("/v1/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model": "granite-3b-moe",
                            "messages": [{ "role": "user", "content": "Hello" }]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let created: CreateJobResponse = serde_json::from_slice(&body).unwrap();

        // Verify the stored job has normal priority
        let job_id: JobId = created.job_id.parse().unwrap();
        let job = state.job_store.find_by_id(&job_id).unwrap().unwrap();
        assert_eq!(job.priority, Priority::Normal);
    }

    #[tokio::test]
    async fn create_job_rejects_invalid_priority() {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method(http::Method::POST)
                    .uri("/v1/jobs")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "model": "granite-3b-moe",
                            "messages": [{ "role": "user", "content": "Hello" }],
                            "priority": "urgent"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
