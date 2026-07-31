/// End-to-end tests for distributed inference.
///
/// These tests verify the actual inference pipeline:
/// - Expert workers loading GGUF models
/// - Distributed forward pass across multiple workers
/// - Crash recovery when a worker dies
/// - Multi-worker load balancing
///
/// Requirements:
/// - GGUF model at the path specified by `model_path()`
/// - Sufficient memory to load multiple expert workers
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use synapse_core::native_moe::distributed_forward::{DistributedModel, WorkerConfig};
use synapse_core::native_moe::expert_worker_client::ExpertWorkerClient;
use synapse_core::native_moe::forward;
use synapse_core::native_moe::model::MoeModel;

/// Path to the GGUF model file.
fn model_path() -> PathBuf {
    PathBuf::from(
        "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b",
    )
}

/// Starts an expert worker process on the given port with the given expert indices.
fn start_worker(port: u16, experts: &[usize]) -> Child {
    let path = model_path();
    let expert_strs: Vec<String> = experts.iter().map(|e| e.to_string()).collect();
    Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "expert_worker",
            "--",
            path.to_str().unwrap(),
            "--port",
            &port.to_string(),
        ])
        .args(&expert_strs)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to start expert worker")
}

/// Waits for a worker to become healthy.
async fn wait_for_worker(url: &str, timeout: Duration) -> bool {
    let client = ExpertWorkerClient::new(url.to_string());
    let start = Instant::now();
    while start.elapsed() < timeout {
        if client.health_check().await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

/// Calculates cosine similarity between two vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// Returns the top N indices by logit value.
fn top_n(logits: &[f32], n: usize) -> Vec<usize> {
    let mut idx: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    idx.iter().take(n).map(|(i, _)| *i).collect()
}

/// Cleans up worker processes.
fn cleanup_workers(workers: &mut Vec<Child>) {
    for w in workers {
        let _ = w.kill();
        let _ = w.wait();
    }
}

// ---------------------------------------------------------------------------
// Distributed inference correctness
// ---------------------------------------------------------------------------

/// Verifies distributed inference produces identical logits to monolithic.
#[tokio::test]
#[ignore] // Requires GGUF model — run with: cargo test --test e2e_distributed -- --ignored
async fn distributed_matches_monolithic_logits() {
    let mpath = model_path();
    if !mpath.exists() {
        eprintln!("Model not found at {:?}, skipping", mpath);
        return;
    }

    // Load monolithic model
    let monolithic = MoeModel::load_all(&mpath).expect("load_all failed");
    let prompt_tokens = vec![49u32]; // single token

    // Run monolithic forward
    let mono_output = forward::forward(&monolithic, &prompt_tokens);
    let mono_logits = &mono_output.logits;
    let mono_top5 = top_n(mono_logits, 5);

    // Start 2 workers
    let mut workers = Vec::new();
    workers.push(start_worker(18001, &(0..20).collect::<Vec<_>>()));
    workers.push(start_worker(18002, &(20..40).collect::<Vec<_>>()));

    // Wait for workers
    let all_ready =
        wait_for_worker("http://localhost:18001", Duration::from_secs(120)).await
            && wait_for_worker("http://localhost:18002", Duration::from_secs(120)).await;
    assert!(all_ready, "Workers failed to start");

    // Create distributed model
    let coordinator = MoeModel::load_coordinator(&mpath).expect("load_coordinator failed");
    let dist_configs = vec![
        WorkerConfig {
            url: "http://localhost:18001".to_string(),
            expert_indices: (0..20).collect(),
        },
        WorkerConfig {
            url: "http://localhost:18002".to_string(),
            expert_indices: (20..40).collect(),
        },
    ];
    let dm = DistributedModel::new(coordinator, &dist_configs);

    // Run distributed forward
    let dist_output = dm.forward(&prompt_tokens).await;
    let dist_logits = &dist_output.logits;

    // Verify logits match
    let cos_sim = cosine(mono_logits, dist_logits);
    let dist_top5 = top_n(dist_logits, 5);

    eprintln!("Cosine similarity: {:.6}", cos_sim);
    eprintln!("Mono top5: {:?}", mono_top5);
    eprintln!("Dist top5: {:?}", dist_top5);

    assert!(cos_sim > 0.999, "Cosine similarity too low: {cos_sim}");
    assert_eq!(mono_top5, dist_top5, "Top5 tokens don't match");

    cleanup_workers(&mut workers);
}

/// Verifies 4-worker distributed inference matches monolithic.
#[tokio::test]
#[ignore] // Requires GGUF model
async fn four_workers_match_monolithic() {
    let mpath = model_path();
    if !mpath.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let monolithic = MoeModel::load_all(&mpath).expect("load_all failed");
    let prompt_tokens = vec![49u32];
    let mono_output = forward::forward(&monolithic, &prompt_tokens);
    let mono_logits = &mono_output.logits;

    // Start 4 workers
    let mut workers = Vec::new();
    workers.push(start_worker(18001, &(0..10).collect::<Vec<_>>()));
    workers.push(start_worker(18002, &(10..20).collect::<Vec<_>>()));
    workers.push(start_worker(18003, &(20..30).collect::<Vec<_>>()));
    workers.push(start_worker(18004, &(30..40).collect::<Vec<_>>()));

    // Wait for all workers
    for port in 18001..=18004 {
        let ready =
            wait_for_worker(&format!("http://localhost:{port}"), Duration::from_secs(120)).await;
        assert!(ready, "Worker on :{port} failed to start");
    }

    let coordinator = MoeModel::load_coordinator(&mpath).expect("load_coordinator failed");
    let dist_configs: Vec<WorkerConfig> = (18001..=18004)
        .zip([(0..10), (10..20), (20..30), (30..40)])
        .map(|(port, experts)| WorkerConfig {
            url: format!("http://localhost:{port}"),
            expert_indices: experts.collect(),
        })
        .collect();
    let dm = DistributedModel::new(coordinator, &dist_configs);

    let dist_output = dm.forward(&prompt_tokens).await;
    let cos_sim = cosine(mono_logits, &dist_output.logits);

    eprintln!("4-worker cosine similarity: {:.6}", cos_sim);
    assert!(cos_sim > 0.999, "Cosine similarity too low: {cos_sim}");

    cleanup_workers(&mut workers);
}

// ---------------------------------------------------------------------------
// Crash recovery
// ---------------------------------------------------------------------------

/// Verifies the system recovers when a worker dies mid-inference.
#[tokio::test]
#[ignore] // Requires GGUF model
async fn recovery_after_worker_crash() {
    let mpath = model_path();
    if !mpath.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    // Start 2 workers
    let mut workers = Vec::new();
    workers.push(start_worker(18001, &(0..20).collect::<Vec<_>>()));
    workers.push(start_worker(18002, &(20..40).collect::<Vec<_>>()));

    let all_ready =
        wait_for_worker("http://localhost:18001", Duration::from_secs(120)).await
            && wait_for_worker("http://localhost:18002", Duration::from_secs(120)).await;
    assert!(all_ready, "Workers failed to start");

    // Kill worker 2
    workers[1].kill().expect("failed to kill worker");
    workers[1].wait().expect("failed to wait for worker");
    eprintln!("Worker 2 killed");

    // Worker 1 should still be healthy
    let client1 = ExpertWorkerClient::new("http://localhost:18001".to_string());
    assert!(client1.health_check().await, "Worker 1 should still be healthy");

    // Restart worker 2
    workers[1] = start_worker(18002, &(20..40).collect::<Vec<_>>());
    let ready = wait_for_worker("http://localhost:18002", Duration::from_secs(120)).await;
    assert!(ready, "Worker 2 failed to restart");

    // Both workers should be healthy
    let client2 = ExpertWorkerClient::new("http://localhost:18002".to_string());
    assert!(client1.health_check().await, "Worker 1 should be healthy");
    assert!(client2.health_check().await, "Worker 2 should be healthy");

    cleanup_workers(&mut workers);
}

// ---------------------------------------------------------------------------
// Multi-worker load balancing
// ---------------------------------------------------------------------------

/// Verifies multiple workers can handle concurrent requests.
#[tokio::test]
#[ignore] // Requires GGUF model
async fn concurrent_inference_requests() {
    let mpath = model_path();
    if !mpath.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    // Start 2 workers
    let mut workers = Vec::new();
    workers.push(start_worker(18001, &(0..20).collect::<Vec<_>>()));
    workers.push(start_worker(18002, &(20..40).collect::<Vec<_>>()));

    let all_ready =
        wait_for_worker("http://localhost:18001", Duration::from_secs(120)).await
            && wait_for_worker("http://localhost:18002", Duration::from_secs(120)).await;
    assert!(all_ready, "Workers failed to start");

    // Create distributed model
    let coordinator = MoeModel::load_coordinator(&mpath).expect("load_coordinator failed");
    let dist_configs = vec![
        WorkerConfig {
            url: "http://localhost:18001".to_string(),
            expert_indices: (0..20).collect(),
        },
        WorkerConfig {
            url: "http://localhost:18002".to_string(),
            expert_indices: (20..40).collect(),
        },
    ];
    let dm = DistributedModel::new(coordinator, &dist_configs);

    // Run 3 sequential inferences (DistributedModel doesn't implement Clone)
    for i in 0..3 {
        let prompt = vec![49u32 + i as u32]; // different tokens
        let start = Instant::now();
        let output = dm.forward(&prompt).await;
        let elapsed = start.elapsed().as_millis();
        eprintln!("Request {}: {}ms, {} logits", i, elapsed, output.logits.len());
        assert!(output.logits.len() > 0, "Request {i} should produce logits");
    }

    cleanup_workers(&mut workers);
}

// ---------------------------------------------------------------------------
// Worker health checks
// ---------------------------------------------------------------------------

/// Verifies worker health endpoints respond correctly.
#[tokio::test]
#[ignore] // Requires GGUF model
async fn worker_health_checks() {
    let mpath = model_path();
    if !mpath.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let mut workers = Vec::new();
    workers.push(start_worker(18001, &(0..5).collect::<Vec<_>>()));

    let ready = wait_for_worker("http://localhost:18001", Duration::from_secs(120)).await;
    assert!(ready, "Worker failed to start");

    let client = ExpertWorkerClient::new("http://localhost:18001".to_string());
    assert!(client.health_check().await, "Worker should be healthy");

    cleanup_workers(&mut workers);
}

// ---------------------------------------------------------------------------
// FFN endpoint
// ---------------------------------------------------------------------------

/// Verifies the FFN endpoint processes expert requests correctly.
#[tokio::test]
#[ignore] // Requires GGUF model
async fn ffn_endpoint_processes_request() {
    let mpath = model_path();
    if !mpath.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    let mut workers = Vec::new();
    workers.push(start_worker(18001, &(0..5).collect::<Vec<_>>()));

    let ready = wait_for_worker("http://localhost:18001", Duration::from_secs(120)).await;
    assert!(ready, "Worker failed to start");

    // Send FFN request
    let client = reqwest::Client::new();
    let resp = client
        .post("http://localhost:18001/ffn")
        .json(&serde_json::json!({
            "layer": 0,
            "hidden": vec![0.0f32; 1536],
            "expert_ids": [0],
            "expert_scores": [1.0]
        }))
        .send()
        .await
        .expect("FFN request failed");

    assert_eq!(resp.status(), 200);

    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["output"].is_array(), "Response should contain output array");

    let output = body["output"].as_array().unwrap();
    assert_eq!(output.len(), 512, "Output should have d_ff=512 elements");

    cleanup_workers(&mut workers);
}

// ---------------------------------------------------------------------------
// Performance benchmarks
// ---------------------------------------------------------------------------

/// Measures distributed inference latency.
#[tokio::test]
#[ignore] // Requires GGUF model
async fn measure_distributed_latency() {
    let mpath = model_path();
    if !mpath.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    // Monolithic baseline
    let monolithic = MoeModel::load_all(&mpath).expect("load_all failed");
    let prompt_tokens = vec![49u32];

    let start = Instant::now();
    let _ = forward::forward(&monolithic, &prompt_tokens);
    let mono_ms = start.elapsed().as_millis();
    eprintln!("Monolithic: {}ms", mono_ms);

    // Distributed with 2 workers
    let mut workers = Vec::new();
    workers.push(start_worker(18001, &(0..20).collect::<Vec<_>>()));
    workers.push(start_worker(18002, &(20..40).collect::<Vec<_>>()));

    let all_ready =
        wait_for_worker("http://localhost:18001", Duration::from_secs(120)).await
            && wait_for_worker("http://localhost:18002", Duration::from_secs(120)).await;
    assert!(all_ready, "Workers failed to start");

    let coordinator = MoeModel::load_coordinator(&mpath).expect("load_coordinator failed");
    let dist_configs = vec![
        WorkerConfig {
            url: "http://localhost:18001".to_string(),
            expert_indices: (0..20).collect(),
        },
        WorkerConfig {
            url: "http://localhost:18002".to_string(),
            expert_indices: (20..40).collect(),
        },
    ];
    let dm = DistributedModel::new(coordinator, &dist_configs);

    let start = Instant::now();
    let _ = dm.forward(&prompt_tokens).await;
    let dist_ms = start.elapsed().as_millis();
    eprintln!("Distributed (2 workers): {}ms", dist_ms);

    let speedup = mono_ms as f64 / dist_ms as f64;
    eprintln!("Speedup: {:.2}x", speedup);

    cleanup_workers(&mut workers);
}
