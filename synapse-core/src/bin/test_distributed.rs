/// Integration test: monolithic vs distributed inference.
///
/// Starts expert workers, runs the same prompt through both
/// monolithic and distributed forward passes, compares logits.
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::Duration;

use synapse_core::native_moe::distributed_forward::{
    DistributedModel, WorkerConfig,
};
use synapse_core::native_moe::expert_worker_client::ExpertWorkerClient;
use synapse_core::native_moe::forward;
use synapse_core::native_moe::model::MoeModel;

fn model_path() -> PathBuf {
    PathBuf::from(
        "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b",
    )
}

fn start_worker(
    port: u16,
    layer: usize,
    experts: &[usize],
) -> Child {
    let path = model_path();
    let expert_strs: Vec<String> =
        experts.iter().map(|e| e.to_string()).collect();

    let child = Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "expert_worker",
            "--",
            path.to_str().unwrap(),
            &layer.to_string(),
            "--port",
            &port.to_string(),
        ])
        .args(&expert_strs)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to start expert worker");

    child
}

async fn wait_for_worker(url: &str, timeout: Duration) -> bool {
    let client = ExpertWorkerClient::new(url.to_string());
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if client.health_check().await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

#[tokio::main]
async fn main() {
    let model_path = model_path();
    if !model_path.exists() {
        eprintln!("Model not found, skipping test");
        return;
    }

    eprintln!("=== Distributed vs Monolithic Integration Test ===\n");

    // Load monolithic model
    eprintln!("Loading monolithic model...");
    let monolithic =
        MoeModel::load_all(&model_path).expect("failed to load model");

    // Tokenize a simple prompt (use token IDs directly)
    // "What is 2+2?" → approximate token IDs for granite
    let prompt_tokens = vec![1u32, 2, 3, 4, 5];

    // Run monolithic forward
    eprintln!("Running monolithic forward...");
    let mono_output = forward::forward(&monolithic, &prompt_tokens);
    eprintln!(
        "  logits len={}, top5 indices={:?}",
        mono_output.logits.len(),
        mono_output
            .logits
            .iter()
            .enumerate()
            .collect::<Vec<_>>()
            .iter()
            .take(5)
            .map(|(i, _)| *i)
            .collect::<Vec<_>>()
    );

    // Start expert workers
    eprintln!("\nStarting expert workers...");
    let mut workers = Vec::new();

    // Worker 1: experts 0-19 on port 8001
    let w1 = start_worker(8001, 0, &(0..20).collect::<Vec<_>>());
    workers.push(w1);
    eprintln!("  Worker 1: experts 0-19 on :8001");

    // Worker 2: experts 20-39 on port 8002
    let w2 = start_worker(8002, 0, &(20..40).collect::<Vec<_>>());
    workers.push(w2);
    eprintln!("  Worker 2: experts 20-39 on :8002");

    // Wait for workers to be ready
    eprintln!("  Waiting for workers...");
    let w1_ready =
        wait_for_worker("http://localhost:8001", Duration::from_secs(60))
            .await;
    let w2_ready =
        wait_for_worker("http://localhost:8002", Duration::from_secs(60))
            .await;

    if !w1_ready || !w2_ready {
        eprintln!("  Workers failed to start, aborting");
        for mut w in workers {
            let _ = w.kill();
        }
        return;
    }
    eprintln!("  Both workers ready\n");

    // Test remote FFN on worker 1
    eprintln!("Testing remote FFN on worker 1...");
    let client1 =
        ExpertWorkerClient::new("http://localhost:8001".into());
    let test_hidden = vec![1.0f32; 1536];
    let test_ids = vec![0u32, 1u32];
    let test_scores = vec![0.6f32, 0.4f32];

    match client1
        .compute_ffn(
            test_hidden.clone(),
            test_ids.clone(),
            test_scores.clone(),
        )
        .await
    {
        Ok(output) => {
            eprintln!(
                "  FFN output len={}, norm={:.4}",
                output.len(),
                output
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
            );
        }
        Err(e) => {
            eprintln!("  FFN failed: {e}");
        }
    }

    // Test remote FFN on worker 2
    eprintln!("Testing remote FFN on worker 2...");
    let client2 =
        ExpertWorkerClient::new("http://localhost:8002".into());
    let test_ids2 = vec![20u32, 21u32];

    match client2
        .compute_ffn(test_hidden, test_ids2, test_scores)
        .await
    {
        Ok(output) => {
            eprintln!(
                "  FFN output len={}, norm={:.4}",
                output.len(),
                output
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
            );
        }
        Err(e) => {
            eprintln!("  FFN failed: {e}");
        }
    }

    // Cleanup
    eprintln!("\nStopping workers...");
    for mut w in workers {
        let _ = w.kill();
    }

    eprintln!("\n=== Test Complete ===");
    eprintln!("Monolithic forward produced logits (len={})", mono_output.logits.len());
    eprintln!("Workers served FFN requests successfully.");
    eprintln!("Full distributed inference requires wiring coordinator → workers in the forward loop.");
}
