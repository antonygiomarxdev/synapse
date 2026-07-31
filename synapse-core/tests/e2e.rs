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

/// Verifies the metrics endpoint returns Prometheus format.
#[tokio::test]
async fn metrics_endpoint_returns_prometheus() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        synapse_core::gateway::api::serve_on(listener).await;
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .expect("server should respond to metrics endpoint");
    assert_eq!(resp.status(), 200);

    let body = resp.text().await.unwrap();
    assert!(body.contains("# HELP synapse_jobs_total"));
    assert!(body.contains("# TYPE synapse_jobs_total counter"));
}

/// Verifies job creation and polling lifecycle.
#[tokio::test]
async fn job_creation_and_polling() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        synapse_core::gateway::api::serve_on(listener).await;
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let client = reqwest::Client::new();

    // Create job
    let resp = client
        .post(format!("http://{addr}/v1/jobs"))
        .json(&serde_json::json!({
            "model": "granite-3b-moe",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .expect("server should create job");
    assert_eq!(resp.status(), 202);

    let create_resp: serde_json::Value = resp.json().await.unwrap();
    let job_id = create_resp["job_id"].as_str().unwrap();
    assert!(!job_id.is_empty());

    // Poll job
    let resp = client
        .get(format!("http://{addr}/v1/jobs/{job_id}"))
        .send()
        .await
        .expect("server should poll job");
    assert_eq!(resp.status(), 200);

    let poll_resp: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(poll_resp["job_id"], job_id);
    assert_eq!(poll_resp["object"], "job");
}

/// Verifies chat completions endpoint returns OpenAI-compatible format.
#[tokio::test]
async fn chat_completions_returns_openai_format() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        synapse_core::gateway::api::serve_on(listener).await;
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .expect("server should respond to chat completions");
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(body["object"], "chat.completion");
    assert!(body["choices"].is_array());
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
}

/// Verifies job creation rejects invalid requests.
#[tokio::test]
async fn job_creation_rejects_invalid_requests() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        synapse_core::gateway::api::serve_on(listener).await;
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let client = reqwest::Client::new();

    // Empty model
    let resp = client
        .post(format!("http://{addr}/v1/jobs"))
        .json(&serde_json::json!({
            "model": "",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .expect("server should respond");
    assert_eq!(resp.status(), 400);

    // Empty messages
    let resp = client
        .post(format!("http://{addr}/v1/jobs"))
        .json(&serde_json::json!({
            "model": "granite-3b-moe",
            "messages": []
        }))
        .send()
        .await
        .expect("server should respond");
    assert_eq!(resp.status(), 400);
}

/// Verifies Swagger UI is accessible.
#[tokio::test]
async fn swagger_ui_is_accessible() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        synapse_core::gateway::api::serve_on(listener).await;
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/swagger-ui/"))
        .send()
        .await
        .expect("server should respond to swagger-ui");
    assert_eq!(resp.status(), 200);
}

/// Verifies OpenAPI spec is accessible.
#[tokio::test]
async fn openapi_spec_is_accessible() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        synapse_core::gateway::api::serve_on(listener).await;
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/api-docs/openapi.json"))
        .send()
        .await
        .expect("server should respond to openapi spec");
    assert_eq!(resp.status(), 200);

    let spec: serde_json::Value = resp.json().await.unwrap();
    assert!(spec["openapi"].is_string());
}
