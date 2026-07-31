use std::sync::Arc;

use axum::{Json, Router, routing::get};
use serde::Serialize;
use utoipa::OpenApi;

use super::{catalog, jobs, router};
use crate::job::infrastructure::InMemoryJobStore;

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
    println!("Synapse Gateway listening on http://{bind_addr}");
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

/// Builds the gateway router with default in-memory state.
pub fn build_router() -> Router {
    let state = jobs::AppState {
        job_store: Arc::new(InMemoryJobStore::new()),
        scheduler: None,
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
