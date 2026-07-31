/// Debug: compare local vs remote FFN for a single layer.
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::Duration;

use synapse_core::native_moe::expert_worker_client::ExpertWorkerClient;
use synapse_core::native_moe::forward;
use synapse_core::native_moe::model::MoeModel;

fn model_path() -> PathBuf {
    PathBuf::from(
        "/home/ksante/.ollama/models/blobs/sha256-4cbc52994d8ce56d58f3ecadcd451a5dbb2a4f1142098c6b9f030d18ee5e052b",
    )
}

fn start_worker(port: u16, experts: &[usize]) -> Child {
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

async fn wait_for_worker(url: &str) -> bool {
    let client = ExpertWorkerClient::new(url.to_string());
    let start = std::time::Instant::now();
    while start.elapsed() < Duration::from_secs(60) {
        if client.health_check().await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

#[tokio::main]
async fn main() {
    let mpath = model_path();
    if !mpath.exists() {
        eprintln!("Model not found");
        return;
    }

    eprintln!("=== FFN Debug: Local vs Remote ===\n");

    let model = MoeModel::load_all(&mpath).unwrap();
    let tokens = vec![49u32];
    let d_model = model.config.d_model as usize;

    // Embedding
    let hidden: Vec<Vec<f32>> = if let Some(ref embd) = model.token_embd {
        let d = embd.shape[0] as usize;
        let shape_vocab = embd.shape[1] as usize;
        tokens
            .iter()
            .map(|&tid| {
                let t = tid as usize % shape_vocab;
                (0..d)
                    .map(|dim| model.config.embedding_scale * embd.data[t * d + dim])
                    .collect()
            })
            .collect()
    } else {
        vec![vec![0.0f32; d_model]; 1]
    };

    // Run attention for layer 0
    let attn_out = forward::forward_layer_attention(&model, 0, hidden);
    let route = &attn_out.route;

    eprintln!("Routing: experts={:?}, scores={:.4?}", route.1, route.2);

    // Local FFN
    let layer = &model.layers[0];
    let local_ffn = if let (Some(gate), Some(up), Some(down)) =
        (&layer.gate_exps, &layer.up_exps, &layer.down_exps)
    {
        forward::expert_ffn(
            &attn_out.ffn_normed,
            gate,
            up,
            down,
            &route.1,
            &route.2,
            model.config.d_ff as usize,
        )
    } else {
        eprintln!("No expert weights!");
        return;
    };

    eprintln!("Local FFN[0][0..5] = {:?}", &local_ffn[0][..5]);
    eprintln!("Local FFN norm = {:.4}", local_ffn[0].iter().map(|x| x * x).sum::<f32>().sqrt());

    // Start workers
    eprintln!("\nStarting workers...");
    let mut workers = Vec::new();
    workers.push(start_worker(8001, &(0..20).collect::<Vec<_>>()));
    workers.push(start_worker(8002, &(20..40).collect::<Vec<_>>()));
    wait_for_worker("http://localhost:8001").await;
    wait_for_worker("http://localhost:8002").await;
    eprintln!("  Workers ready");

    // Remote FFN — send ALL experts to their respective workers at once
    // This preserves the score normalization
    let client1 = ExpertWorkerClient::new("http://localhost:8001".into());
    let client2 = ExpertWorkerClient::new("http://localhost:8002".into());

    // Group experts by worker
    let mut worker_experts: HashMap<usize, Vec<(u32, f32)>> = HashMap::new();
    for (i, &eid) in route.1.iter().enumerate() {
        let wid = if eid < 20 { 0 } else { 1 };
        worker_experts.entry(wid).or_default().push((eid, route.2[i]));
    }

    let mut remote_ffn = vec![0.0f32; d_model];
    for (wid, experts) in &worker_experts {
        let client = if *wid == 0 { &client1 } else { &client2 };
        let ids: Vec<u32> = experts.iter().map(|(id, _)| *id).collect();
        let scores: Vec<f32> = experts.iter().map(|(_, s)| *s).collect();

        match client
            .compute_ffn(0, attn_out.ffn_normed[0].clone(), ids.clone(), scores.clone())
            .await
        {
            Ok(output) => {
                eprintln!("  Worker {wid}: experts={:?}, output norm={:.4}", ids, output.iter().map(|x| x * x).sum::<f32>().sqrt());
                for d in 0..d_model {
                    remote_ffn[d] += output[d];
                }
            }
            Err(e) => eprintln!("  Worker {wid} failed: {e}"),
        }
    }

    eprintln!("Remote FFN[0..5] = {:?}", &remote_ffn[..5]);
    eprintln!("Remote FFN norm = {:.4}", remote_ffn.iter().map(|x| x * x).sum::<f32>().sqrt());

    // Compare
    let dot: f32 = local_ffn[0].iter().zip(remote_ffn.iter()).map(|(a, b)| a * b).sum();
    let norm_local: f32 = local_ffn[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_remote: f32 = remote_ffn.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos_sim = dot / (norm_local * norm_remote);

    eprintln!("\nCosine similarity: {:.6}", cos_sim);

    if cos_sim > 0.99 {
        eprintln!("PASS: FFN outputs match");
    } else {
        eprintln!("FAIL: FFN outputs diverge");
        // Debug: check individual expert outputs
        eprintln!("\nDebug: comparing individual experts...");
        for (i, &eid) in route.1.iter().enumerate().take(2) {
            let score = route.2[i];
            eprintln!("  Expert {eid}: score={:.4}", score);
        }
    }

    for mut w in workers {
        let _ = w.kill();
    }
}
