/// End-to-end test: distributed vs monolithic inference.
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

fn start_worker(port: u16, layer: usize, experts: &[usize]) -> Child {
    let path = model_path();
    let expert_strs: Vec<String> =
        experts.iter().map(|e| e.to_string()).collect();

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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn top_n(logits: &[f32], n: usize) -> Vec<usize> {
    let mut idx: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    idx.iter().take(n).map(|(i, _)| *i).collect()
}

#[tokio::main]
async fn main() {
    let mpath = model_path();
    if !mpath.exists() {
        eprintln!("Model not found, skipping");
        return;
    }

    eprintln!("=== Distributed vs Monolithic End-to-End Test ===\n");

    // Load monolithic model
    eprintln!("Loading monolithic model...");
    let monolithic =
        MoeModel::load_all(&mpath).expect("failed to load model");

    let prompt_tokens = vec![49u32]; // single token

    // Run monolithic forward
    eprintln!("Running monolithic forward...");
    let mono_output = forward::forward(&monolithic, &prompt_tokens);
    let mono_logits = &mono_output.logits;

    eprintln!(
        "  logits len={}, max={:.2}, top5={:?}",
        mono_logits.len(),
        mono_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        top_n(mono_logits, 5)
    );

    // Start expert workers
    eprintln!("\nStarting expert workers...");
    let mut workers = Vec::new();

    workers.push(start_worker(8001, 0, &(0..20).collect::<Vec<_>>()));
    eprintln!("  Worker 1: experts 0-19 on :8001");

    workers.push(start_worker(8002, 0, &(20..40).collect::<Vec<_>>()));
    eprintln!("  Worker 2: experts 20-39 on :8002");

    eprintln!("  Waiting for workers...");
    let w1_ok =
        wait_for_worker("http://localhost:8001", Duration::from_secs(120))
            .await;
    let w2_ok =
        wait_for_worker("http://localhost:8002", Duration::from_secs(120))
            .await;

    if !w1_ok || !w2_ok {
        eprintln!("  Workers failed to start");
        for mut w in workers {
            let _ = w.kill();
        }
        return;
    }
    eprintln!("  Both workers ready\n");

    // Create distributed model with coordinator (attention + routing, no expert FFN)
    let coordinator =
        MoeModel::load_coordinator(&mpath).expect("load_coordinator failed");

    let configs = vec![
        WorkerConfig {
            url: "http://localhost:8001".into(),
            expert_indices: (0..20).collect(),
        },
        WorkerConfig {
            url: "http://localhost:8002".into(),
            expert_indices: (20..40).collect(),
        },
    ];

    let dm = DistributedModel::new(coordinator, &configs);

    // Run distributed forward
    eprintln!("Running distributed forward...");
    let dist_output = dm.forward(&prompt_tokens).await;
    let dist_logits = &dist_output.logits;

    eprintln!(
        "  logits len={}, max={:.2}, top5={:?}",
        dist_logits.len(),
        dist_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        top_n(dist_logits, 5)
    );

    // Compare
    let cos_sim = cosine_similarity(mono_logits, dist_logits);
    let mono_top5 = top_n(mono_logits, 5);
    let dist_top5 = top_n(dist_logits, 5);
    let top5_match = mono_top5 == dist_top5;

    eprintln!("\n=== Results ===");
    eprintln!("  Cosine similarity: {:.6}", cos_sim);
    eprintln!("  Mono top5: {:?}", mono_top5);
    eprintln!("  Dist top5: {:?}", dist_top5);
    eprintln!("  Top5 match: {}", top5_match);

    // Cleanup
    eprintln!("\nStopping workers...");
    for mut w in workers {
        let _ = w.kill();
    }

    // Verdict
    eprintln!("\n=== Verdict ===");
    if cos_sim > 0.99 {
        eprintln!(
            "  PASS: Distributed matches monolithic (cos_sim={:.4})",
            cos_sim
        );
    } else if cos_sim > 0.95 {
        eprintln!(
            "  WARN: Close but not exact (cos_sim={:.4})",
            cos_sim
        );
    } else {
        eprintln!(
            "  FAIL: Significant divergence (cos_sim={:.4})",
            cos_sim
        );
    }
}
