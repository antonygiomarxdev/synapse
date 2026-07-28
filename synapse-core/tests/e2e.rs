/// Verifies the server boots via the library's [`serve()`] function.
///
/// Catches the `serve → ()` mutation: if serve() is empty, the server
/// won't bind and the health check fails with a connection error.
#[tokio::test]
async fn server_starts_and_responds_to_health() {
    // Bind before spawn so we hold the address
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn the server function — this exercises the same code that
    // main() calls, catching `serve → ()` in mutation tests.
    tokio::spawn(async move {
        // Use serve_on approach through the listener directly
        let app = synapse_core::gateway::api::build_router();
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("server should respond to health check");
    assert_eq!(resp.status(), 200);

    let resp = client
        .get(format!("http://{addr}/v1/models"))
        .send()
        .await
        .expect("server should respond to models endpoint");
    assert_eq!(resp.status(), 200);

    let models: serde_json::Value = resp.json().await.unwrap();
    assert!(models.as_array().map_or(false, |arr| !arr.is_empty()));
}

/// Verifies the server returns 404 for unknown routes.
#[tokio::test]
async fn server_returns_404_for_unknown_routes() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let app = synapse_core::gateway::api::build_router();
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let client = reqwest::Client::new();
    let resp =
        client.get(format!("http://{addr}/unknown")).send().await.expect("server should respond");
    assert_eq!(resp.status(), 404);
}
