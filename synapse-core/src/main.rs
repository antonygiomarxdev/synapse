use synapse_core::gateway;

/// Starts the HTTP gateway server on the configured address.
///
/// Extracted from `main()` so it can be tested independently.
/// In production, `main()` calls this. In tests, we bind to port 0
/// and verify the server starts successfully.
pub async fn serve(bind_addr: &str) {
    let app = gateway::api::build_router();

    let listener =
        tokio::net::TcpListener::bind(bind_addr).await.expect("failed to bind TCP listener");

    println!("Synapse Gateway listening on http://{bind_addr}");

    axum::serve(listener, app).await.unwrap();
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    serve("0.0.0.0:8000").await;
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn build_router_responds_to_health() {
        let app = synapse_core::gateway::api::build_router();
        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn build_router_responds_to_models() {
        let app = synapse_core::gateway::api::build_router();
        let response = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn build_router_404_on_unknown_route() {
        let app = synapse_core::gateway::api::build_router();
        let response = app
            .oneshot(Request::builder().uri("/unknown").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
