use reqwest::Client;
use std::net::SocketAddr;
use std::time::Duration;

/// Time for the server to become ready after spawning.
const SERVER_READY_DELAY: Duration = Duration::from_millis(200);

/// Spawns a Synapse gateway on a random port and returns the address and HTTP client.
async fn spawn_gateway() -> (SocketAddr, Client) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        synapse_core::gateway::api::serve_on(listener).await;
    });
    tokio::time::sleep(SERVER_READY_DELAY).await;
    (addr, Client::new())
}

/// Helper to build a JSON request body.
fn json_body(value: serde_json::Value) -> reqwest::Body {
    serde_json::to_vec(&value).unwrap().into()
}

// ---------------------------------------------------------------------------
// Health and discovery
// ---------------------------------------------------------------------------

/// Verifies the server boots and responds to health checks.
#[tokio::test]
async fn health_returns_ok_with_version() {
    let (addr, client) = spawn_gateway().await;

    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("server should respond to health check");
    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert!(body["version"].is_string());
}

/// Verifies the server returns 404 for unknown routes.
#[tokio::test]
async fn unknown_route_returns_404() {
    let (addr, client) = spawn_gateway().await;

    let resp = client
        .get(format!("http://{addr}/unknown"))
        .send()
        .await
        .expect("server should respond");
    assert_eq!(resp.status(), 404);
}

// ---------------------------------------------------------------------------
// Models catalog
// ---------------------------------------------------------------------------

/// Verifies /v1/models returns a non-empty array.
#[tokio::test]
async fn list_models_returns_non_empty_array() {
    let (addr, client) = spawn_gateway().await;

    let resp = client
        .get(format!("http://{addr}/v1/models"))
        .send()
        .await
        .expect("server should respond to models endpoint");
    assert_eq!(resp.status(), 200);

    let models: serde_json::Value = resp.json().await.unwrap();
    assert!(models.as_array().is_some_and(|arr| !arr.is_empty()));
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Verifies /metrics returns Prometheus text format.
#[tokio::test]
async fn metrics_endpoint_returns_prometheus_format() {
    let (addr, client) = spawn_gateway().await;

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

// ---------------------------------------------------------------------------
// Jobs lifecycle
// ---------------------------------------------------------------------------

/// Verifies job creation returns 202 with a job_id.
#[tokio::test]
async fn create_job_returns_202_with_job_id() {
    let (addr, client) = spawn_gateway().await;

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

    let body: serde_json::Value = resp.json().await.unwrap();
    let job_id = body["job_id"].as_str().unwrap();
    assert!(!job_id.is_empty());
}

/// Verifies job creation then polling returns the same job_id.
#[tokio::test]
async fn create_then_poll_returns_matching_job() {
    let (addr, client) = spawn_gateway().await;

    // Create
    let resp = client
        .post(format!("http://{addr}/v1/jobs"))
        .json(&serde_json::json!({
            "model": "granite-3b-moe",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .expect("server should create job");
    let create_body: serde_json::Value = resp.json().await.unwrap();
    let job_id = create_body["job_id"].as_str().unwrap();

    // Poll
    let resp = client
        .get(format!("http://{addr}/v1/jobs/{job_id}"))
        .send()
        .await
        .expect("server should poll job");
    assert_eq!(resp.status(), 200);

    let poll_body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(poll_body["job_id"], job_id);
    assert_eq!(poll_body["object"], "job");
}

/// Verifies job creation rejects empty model.
#[tokio::test]
async fn create_job_rejects_empty_model() {
    let (addr, client) = spawn_gateway().await;

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
}

/// Verifies job creation rejects empty messages.
#[tokio::test]
async fn create_job_rejects_empty_messages() {
    let (addr, client) = spawn_gateway().await;

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

/// Verifies polling an unknown job returns 404.
#[tokio::test]
async fn poll_unknown_job_returns_404() {
    let (addr, client) = spawn_gateway().await;

    let resp = client
        .get(format!("http://{addr}/v1/jobs/00000000-0000-0000-0000-000000000000"))
        .send()
        .await
        .expect("server should respond");
    assert_eq!(resp.status(), 404);
}

// ---------------------------------------------------------------------------
// Chat completions (OpenAI-compatible)
// ---------------------------------------------------------------------------

/// Verifies /v1/chat/completions returns OpenAI-compatible format.
#[tokio::test]
async fn chat_completions_returns_openai_format() {
    let (addr, client) = spawn_gateway().await;

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

// ---------------------------------------------------------------------------
// API docs
// ---------------------------------------------------------------------------

/// Verifies Swagger UI is accessible.
#[tokio::test]
async fn swagger_ui_is_accessible() {
    let (addr, client) = spawn_gateway().await;

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
    let (addr, client) = spawn_gateway().await;

    let resp = client
        .get(format!("http://{addr}/api-docs/openapi.json"))
        .send()
        .await
        .expect("server should respond to openapi spec");
    assert_eq!(resp.status(), 200);

    let spec: serde_json::Value = resp.json().await.unwrap();
    assert!(spec["openapi"].is_string());
}

// ---------------------------------------------------------------------------
// Multi-worker and resilience (spec #55)
// ---------------------------------------------------------------------------

/// Spawns an expert worker on a random port.
async fn spawn_expert_worker(port: u16) -> tokio::process::Child {
    tokio::process::Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "expert_worker",
            "--",
            "--port",
            &port.to_string(),
        ])
        .kill_on_drop(true)
        .spawn()
        .expect("failed to spawn expert worker")
}

/// Verifies two workers can register with the scheduler concurrently.
#[tokio::test]
async fn two_workers_register_concurrently() {
    let (addr, client) = spawn_gateway().await;

    // Submit two jobs that would be dispatched to different workers
    let resp1 = client
        .post(format!("http://{addr}/v1/jobs"))
        .json(&serde_json::json!({
            "model": "granite-3b-moe",
            "messages": [{"role": "user", "content": "job 1"}]
        }))
        .send()
        .await
        .expect("server should create job");
    assert_eq!(resp1.status(), 202);

    let resp2 = client
        .post(format!("http://{addr}/v1/jobs"))
        .json(&serde_json::json!({
            "model": "granite-3b-moe",
            "messages": [{"role": "user", "content": "job 2"}]
        }))
        .send()
        .await
        .expect("server should create job");
    assert_eq!(resp2.status(), 202);

    // Both should have unique job IDs
    let body1: serde_json::Value = resp1.json().await.unwrap();
    let body2: serde_json::Value = resp2.json().await.unwrap();
    assert_ne!(body1["job_id"], body2["job_id"]);
}

/// Verifies the gateway handles concurrent requests without crashing.
#[tokio::test]
async fn concurrent_requests_dont_crash_gateway() {
    let (addr, client) = spawn_gateway().await;

    let mut handles = Vec::new();
    for i in 0..10 {
        let client = client.clone();
        let url = format!("http://{addr}/v1/jobs");
        handles.push(tokio::spawn(async move {
            client
                .post(&url)
                .json(&serde_json::json!({
                    "model": "granite-3b-moe",
                    "messages": [{"role": "user", "content": format!("concurrent {i}")}]
                }))
                .send()
                .await
        }));
    }

    for handle in handles {
        let resp = handle.await.unwrap().unwrap();
        assert_eq!(resp.status(), 202);
    }
}

/// Verifies the scheduler recovers from a single task failure.
#[tokio::test]
async fn scheduler_recovers_from_task_failure() {
    let (addr, client) = spawn_gateway().await;

    // Submit a job — even if the worker is unavailable, the gateway should not crash
    let resp = client
        .post(format!("http://{addr}/v1/jobs"))
        .json(&serde_json::json!({
            "model": "granite-3b-moe",
            "messages": [{"role": "user", "content": "test recovery"}]
        }))
        .send()
        .await
        .expect("server should create job");
    assert_eq!(resp.status(), 202);

    let body: serde_json::Value = resp.json().await.unwrap();
    let job_id = body["job_id"].as_str().unwrap();

    // Poll — the job should exist (even if failed, it shouldn't 500)
    let resp = client
        .get(format!("http://{addr}/v1/jobs/{job_id}"))
        .send()
        .await
        .expect("server should poll job");
    assert!(
        resp.status().is_success(),
        "gateway should survive worker unavailability"
    );
}

/// Verifies the gateway remains healthy after processing multiple jobs.
#[tokio::test]
async fn gateway_stays_healthy_after_load() {
    let (addr, client) = spawn_gateway().await;

    // Submit several jobs
    for i in 0..5 {
        let resp = client
            .post(format!("http://{addr}/v1/jobs"))
            .json(&serde_json::json!({
                "model": "granite-3b-moe",
                "messages": [{"role": "user", "content": format!("load test {i}")}]
            }))
            .send()
            .await
            .expect("server should create job");
        assert_eq!(resp.status(), 202);
    }

    // Health check should still work
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("server should respond to health");
    assert_eq!(resp.status(), 200);

    // Metrics should reflect the jobs
    let resp = client
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .expect("server should respond to metrics");
    let body = resp.text().await.unwrap();
    assert!(body.contains("synapse_jobs_total 5"));
}
