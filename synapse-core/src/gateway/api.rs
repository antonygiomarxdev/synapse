use std::sync::Arc;

use axum::{Json, Router, routing::get};
use serde::Serialize;
use utoipa::OpenApi;

use super::{catalog, jobs, router};
use crate::job::infrastructure::InMemoryJobStore;
use crate::scheduler::scheduler::Scheduler;
use crate::scheduler::infrastructure::{InMemoryTaskStore, OllamaWorkerPort, WorkerConfig};
use crate::scheduler::task::WorkerInfo;
use crate::scheduler::worker_id::WorkerId;

/// Default Ollama endpoint.
pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Default model for inference.
pub const DEFAULT_MODEL: &str = "granite3.1-moe:3b";

/// Default worker ID.
pub const DEFAULT_WORKER_ID: &str = "ollama-0";

/// Configuration for the gateway.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Ollama endpoint URL.
    pub ollama_url: String,
    /// Model to use for inference.
    pub model: String,
    /// Worker ID.
    pub worker_id: String,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            ollama_url: DEFAULT_OLLAMA_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            worker_id: DEFAULT_WORKER_ID.to_string(),
        }
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    swarm_nodes: usize,
}

/// OpenAPI documentation.
#[derive(OpenApi)]
#[openapi(
    paths(jobs::create_job, jobs::get_job),
    components(schemas(
        jobs::CreateJobRequest,
        jobs::MessageRequest,
        jobs::CreateJobResponse,
        jobs::JobResponse,
        jobs::JobResultResponse,
        jobs::ErrorResponse,
    )),
    tags((name = "jobs", description = "Async inference job management"))
)]
pub struct ApiDoc;

/// Default bind address for the gateway HTTP server.
pub const DEFAULT_BIND_ADDR: &str = "0.0.0.0:8000";

/// Binds to `addr` and starts the axum HTTP gateway.
///
/// This is a thin wrapper around [`serve_on`] that additionally binds
/// the TCP listener. The binding and logging are I/O glue that cannot
/// be tested without a real network interface.
#[cfg_attr(test, mutants::skip)]
pub async fn serve(bind_addr: &str) {
    let listener =
        tokio::net::TcpListener::bind(bind_addr).await.expect("failed to bind TCP listener");
    tracing::info!("Synapse Gateway listening on http://{bind_addr}");
    serve_on(listener).await;
}

/// Starts the gateway on an already-bound TCP listener.
///
/// Useful for integration tests that bind to port 0 first to discover
/// the assigned address before starting the server.
pub async fn serve_on(listener: tokio::net::TcpListener) {
    let app = build_router();
    axum::serve(listener, app).await.unwrap();
}

/// Builds the gateway router with default configuration.
///
/// Creates a scheduler with an OllamaWorkerPort using default settings.
/// Use [`build_router_with_config`] for custom configuration.
pub fn build_router() -> Router {
    build_router_with_config(GatewayConfig::default())
}

/// Builds the gateway router with custom configuration.
///
/// Creates a scheduler with an OllamaWorkerPort connected to the specified
/// Ollama endpoint. The scheduler processes jobs asynchronously using JoinSet
/// for concurrent task dispatch.
pub fn build_router_with_config(config: GatewayConfig) -> Router {
    let job_store = Arc::new(InMemoryJobStore::new());
    let task_store = Arc::new(InMemoryTaskStore::new());

    // Create Ollama worker
    let worker_id = WorkerId::new(&config.worker_id);
    let ollama_config = WorkerConfig {
        id: worker_id.clone(),
        model: config.model.clone(),
        base_url: config.ollama_url.clone(),
    };
    let worker_port = Arc::new(OllamaWorkerPort::new(vec![ollama_config]));
    let workers = vec![WorkerInfo {
        id: worker_id,
        model: config.model,
        healthy: true,
    }];

    let scheduler = Arc::new(Scheduler::new(
        task_store,
        job_store.clone(),
        worker_port,
        workers,
    ));

    let state = jobs::AppState {
        job_store,
        scheduler: Some(scheduler),
    };
    build_router_with_state(state)
}

/// Builds the gateway router with the given application state.
pub fn build_router_with_state(state: jobs::AppState) -> Router {
    let openapi = ApiDoc::openapi();

    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(catalog::list_models))
        .route("/v1/chat/completions", axum::routing::post(router::chat_completions))
        .route("/v1/jobs", axum::routing::post(jobs::create_job))
        .route("/v1/jobs/{id}", axum::routing::get(jobs::get_job))
        .merge(utoipa_swagger_ui::SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", openapi))
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok", version: env!("CARGO_PKG_VERSION"), swarm_nodes: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn health_endpoint_returns_ok() {
        let app = build_router();

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn openapi_json_endpoint() {
        let app = build_router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api-docs/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn swagger_ui_endpoint() {
        let app = build_router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/swagger-ui/")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
