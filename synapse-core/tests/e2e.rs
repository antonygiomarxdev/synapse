/// Verifies the server boots and responds to health checks.
///
/// Calls the library's `serve_on()` to exercise the same code path
/// used by `serve()` and `main()`. Catches `serve_on → ()` in
/// mutation tests.
#[tokio::test]
async fn server_starts_and_responds_to_health() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        synapse_core::gateway::api::serve_on(listener).await;
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
    assert!(models.as_array().is_some_and(|arr| !arr.is_empty()));
}

/// Verifies the server returns 404 for unknown routes.
#[tokio::test]
async fn server_returns_404_for_unknown_routes() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        synapse_core::gateway::api::serve_on(listener).await;
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let client = reqwest::Client::new();
    let resp =
        client.get(format!("http://{addr}/unknown")).send().await.expect("server should respond");
    assert_eq!(resp.status(), 404);
}
