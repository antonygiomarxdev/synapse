/// Benchmark: async throughput with variable network delay.
///
/// Tests how network round-trip delay (simulated via `--delay-ms`) and batch size (K)
/// affect distributed inference throughput. Runs forward_batch for K single-token sequences
/// and measures tokens/second and latency fraction (wall time / theoretical network bound).
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use synapse_core::native_moe::distributed_forward::{
    DistributedModel, WorkerConfig, load_coordinator,
};
use synapse_core::native_moe::expert_worker_client::ExpertWorkerClient;

/// Returns the path to the model GGUF file.
fn model_path() -> PathBuf {
    PathBuf::from(
        "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b",
    )
}

/// Spawns an expert worker process on a given port for specified expert indices.
/// If delay_ms is provided, appends `--delay-ms <delay_ms>` to worker args.
fn start_worker(port: u16, experts: &[usize], delay_ms: u64) -> Child {
    let path = model_path();
    let expert_strs: Vec<String> = experts.iter().map(|e| e.to_string()).collect();
    let mut args = vec![
        "run".to_string(),
        "--release".to_string(),
        "--bin".to_string(),
        "expert_worker".to_string(),
        "--".to_string(),
        path.to_str().unwrap().to_string(),
        "--port".to_string(),
        port.to_string(),
    ];
    args.extend(expert_strs);
    if delay_ms > 0 {
        args.push("--delay-ms".to_string());
        args.push(delay_ms.to_string());
    }

    Command::new("cargo")
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to start expert worker")
}

/// Waits for a worker to become healthy (responds to health check).
/// Returns true if healthy within timeout, false if timeout expires.
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

/// Parses a comma-separated list of integers from a string.
fn parse_list(s: &str) -> Vec<u64> {
    s.split(',').filter_map(|part| part.trim().parse::<u64>().ok()).collect()
}

/// Extracts CLI argument value after `--key`.
fn get_arg(args: &[String], key: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == key).map(|w| w[1].clone())
}

#[tokio::main]
async fn main() {
    let mpath = model_path();
    if !mpath.exists() {
        eprintln!("Model not found at {:?}", mpath);
        return;
    }

    let args: Vec<String> = std::env::args().collect();

    // Parse CLI arguments
    let k_str = get_arg(&args, "--k").unwrap_or_else(|| "1,8,32,128".to_string());
    let delay_str = get_arg(&args, "--delay").unwrap_or_else(|| "0,20,100".to_string());

    let ks = parse_list(&k_str);
    let delays = parse_list(&delay_str);

    if ks.is_empty() || delays.is_empty() {
        eprintln!("Invalid --k or --delay argument");
        return;
    }

    eprintln!("=== Async Throughput Benchmark ===\n");
    eprintln!("K values (batch sizes): {:?}", ks);
    eprintln!("Delays (ms): {:?}\n", delays);

    // Results storage: (delay_ms, k, wall_ms, tokens_per_sec, latency_fraction)
    let mut results: Vec<(u64, u64, u128, f64, f64)> = Vec::new();

    for &delay_ms in &delays {
        eprintln!("\n--- Delay {} ms ---", delay_ms);

        // Start 2 workers
        eprintln!("  Starting workers...");
        let mut worker_a = start_worker(8001, &(0..20).collect::<Vec<_>>(), delay_ms);
        let mut worker_b = start_worker(8002, &(20..40).collect::<Vec<_>>(), delay_ms);

        // Wait for both to be healthy
        eprintln!("  Waiting for workers...");
        let health_a = wait_for_worker("http://localhost:8001", Duration::from_secs(600)).await;
        let health_b = wait_for_worker("http://localhost:8002", Duration::from_secs(600)).await;

        if !health_a || !health_b {
            eprintln!("  Worker health check failed");
            let _ = worker_a.kill();
            let _ = worker_b.kill();
            continue;
        }

        eprintln!("  All workers ready");

        // Load coordinator for this delay
        let coordinator = match load_coordinator(&mpath) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("  Failed to load coordinator: {e}");
                let _ = worker_a.kill();
                let _ = worker_b.kill();
                continue;
            }
        };

        let num_layers = coordinator.config.n_layers as usize;

        // Create DistributedModel
        let worker_configs = vec![
            WorkerConfig {
                url: "http://localhost:8001".to_string(),
                expert_indices: (0..20).collect(),
            },
            WorkerConfig {
                url: "http://localhost:8002".to_string(),
                expert_indices: (20..40).collect(),
            },
        ];

        let dm = DistributedModel::new(coordinator, &worker_configs);

        // Run throughput test for each K
        for &k in &ks {
            eprintln!("  K={}", k);

            // Create K single-token sequences with token ids 49, 50, 51, ...
            let seqs: Vec<Vec<u32>> = (0..k as usize).map(|i| vec![49 + i as u32]).collect();

            // Time forward_batch
            let start = Instant::now();
            let _outputs = dm.forward_batch(&seqs).await;
            let wall_ms = start.elapsed().as_millis();

            // Compute metrics
            let wall_s = wall_ms as f64 / 1000.0;
            let tokens_per_sec = if wall_s > 0.0 { k as f64 / wall_s } else { 0.0 };

            // Latency fraction: (32 layers * delay_ms) / wall_ms
            let theoretical_min_ms = (num_layers as f64) * (delay_ms as f64);
            let latency_frac =
                if wall_ms > 0 { theoretical_min_ms / (wall_ms as f64) } else { 0.0 };

            eprintln!(
                "    wall: {}ms, tokens/s: {:.2}, latency_frac: {:.4}",
                wall_ms, tokens_per_sec, latency_frac
            );

            results.push((delay_ms, k, wall_ms, tokens_per_sec, latency_frac));
        }

        // Kill workers
        eprintln!("  Cleaning up...");
        let _ = worker_a.kill();
        let _ = worker_b.kill();
        let _ = worker_a.wait();
        let _ = worker_b.wait();
    }

    // Print results as markdown table
    eprintln!("\n=== Results ===\n");
    println!("| Delay (ms) | K | Wall (ms) | Tokens/s | Latency Fraction |");
    println!("|------------|---|-----------|----------|------------------|");

    for (delay_ms, k, wall_ms, tokens_per_sec, latency_frac) in &results {
        println!(
            "| {} | {} | {} | {:.2} | {:.4} |",
            delay_ms, k, wall_ms, tokens_per_sec, latency_frac
        );
    }
}
