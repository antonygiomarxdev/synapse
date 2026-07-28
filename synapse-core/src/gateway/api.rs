use axum::{Json, Router, routing::get};
use serde::Serialize;

use super::{catalog, router};

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    swarm_nodes: usize,
}

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
pub fn build_router() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(catalog::list_models))
        .route("/v1/chat/completions", axum::routing::post(router::chat_completions))
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
}
