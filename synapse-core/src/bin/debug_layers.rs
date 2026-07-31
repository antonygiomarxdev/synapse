/// Debug: compare monolithic vs distributed forward layer by layer.
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
    let expert_strs: Vec<String> = experts.iter().map(|e| e.to_string()).collect();
    Command::new("cargo")
        .args(["run", "--release", "--bin", "expert_worker", "--", path.to_str().unwrap(), "--port", &port.to_string()])
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
        if client.health_check().await { return true; }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[tokio::main]
async fn main() {
    let mpath = model_path();
    if !mpath.exists() { eprintln!("Model not found"); return; }

    eprintln!("=== Layer-by-Layer Debug ===\n");

    let model = MoeModel::load_all(&mpath).unwrap();
    let tokens = vec![49u32];
    let d_model = model.config.d_model as usize;
    let n_tokens = tokens.len();

    // Embedding
    let embd_hidden: Vec<Vec<f32>> = if let Some(ref embd) = model.token_embd {
        let d = embd.shape[0] as usize;
        let sv = embd.shape[1] as usize;
        tokens.iter().map(|&tid| {
            let t = tid as usize % sv;
            (0..d).map(|dim| model.config.embedding_scale * embd.data[t * d + dim]).collect()
        }).collect()
    } else {
        vec![vec![0.0f32; d_model]; n_tokens]
    };

    // Start workers
    let mut workers = Vec::new();
    workers.push(start_worker(8001, &(0..20).collect::<Vec<_>>()));
    workers.push(start_worker(8002, &(20..40).collect::<Vec<_>>()));
    wait_for_worker("http://localhost:8001").await;
    wait_for_worker("http://localhost:8002").await;

    let client1 = ExpertWorkerClient::new("http://localhost:8001".into());
    let client2 = ExpertWorkerClient::new("http://localhost:8002".into());

    // Run first 3 layers, comparing monolithic vs distributed
    let max_layers = 3;
    let mut mono_hidden = embd_hidden.clone();
    let mut dist_hidden = embd_hidden.clone();

    for layer_idx in 0..max_layers {
        eprintln!("--- Layer {layer_idx} ---");

        // Monolithic: run full layer (attention + FFN)
        let mono_attn = forward::forward_layer_attention(&model, layer_idx, mono_hidden.clone());
        let mono_ffn = forward::expert_ffn(
            &mono_attn.ffn_normed,
            model.layers[layer_idx].gate_exps.as_ref().unwrap(),
            model.layers[layer_idx].up_exps.as_ref().unwrap(),
            model.layers[layer_idx].down_exps.as_ref().unwrap(),
            &mono_attn.route.1,
            &mono_attn.route.2,
            model.config.d_ff as usize,
        );
        mono_hidden = forward::combine_ffn_residual(&mono_attn.residual2, &mono_ffn, model.config.residual_scale);

        // Distributed: run attention locally, dispatch FFN to workers
        let dist_attn = forward::forward_layer_attention(&model, layer_idx, dist_hidden.clone());

        // Check attention output matches
        let attn_cos = cosine(&mono_attn.ffn_normed[0], &dist_attn.ffn_normed[0]);
        eprintln!("  Attention hidden cos_sim: {:.6}", attn_cos);

        // Check routing matches
        eprintln!("  Mono route: {:?}", &mono_attn.route.1[..3]);
        eprintln!("  Dist route: {:?}", &dist_attn.route.1[..3]);

        // Normalize and dispatch FFN — send all experts per worker at once
        let score_sum: f32 = dist_attn.route.2.iter().sum();
        let norm_scores: Vec<f32> = if score_sum > 1e-6 {
            dist_attn.route.2.iter().map(|s| s / score_sum).collect()
        } else {
            dist_attn.route.2.clone()
        };

        // Group experts by worker
        let mut worker_experts: std::collections::HashMap<usize, Vec<(u32, f32)>> = std::collections::HashMap::new();
        for (i, &eid) in dist_attn.route.1.iter().enumerate() {
            let wid = if eid < 20 { 0 } else { 1 };
            worker_experts.entry(wid).or_default().push((eid, norm_scores[i]));
        }

        let mut remote_ffn = vec![0.0f32; d_model];
        for (wid, experts) in &worker_experts {
            let client = if *wid == 0 { &client1 } else { &client2 };
            let ids: Vec<u32> = experts.iter().map(|(id, _)| *id).collect();
            let scores: Vec<f32> = experts.iter().map(|(_, s)| *s).collect();
            match client.compute_ffn(layer_idx, dist_attn.ffn_normed[0].clone(), ids, scores).await {
                Ok(output) => { for d in 0..d_model { remote_ffn[d] += output[d]; } }
                Err(e) => eprintln!("  Worker {wid} failed: {e}"),
            }
        }

        let ffn_cos = cosine(&mono_ffn[0], &remote_ffn);
        eprintln!("  FFN output cos_sim: {:.6}", ffn_cos);

        dist_hidden = forward::combine_ffn_residual(&dist_attn.residual2, &[remote_ffn.clone()], model.config.residual_scale);

        let hidden_cos = cosine(&mono_hidden[0], &dist_hidden[0]);
        eprintln!("  After residual cos_sim: {:.6}", hidden_cos);
        eprintln!("  Mono hidden norm: {:.4}", mono_hidden[0].iter().map(|x| x * x).sum::<f32>().sqrt());
        eprintln!("  Dist hidden norm: {:.4}", dist_hidden[0].iter().map(|x| x * x).sum::<f32>().sqrt());
    }

    for mut w in workers { let _ = w.kill(); }
}
