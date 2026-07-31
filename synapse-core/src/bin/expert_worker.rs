use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use synapse_core::native_moe::expert_shard::ExpertShard;

#[derive(Deserialize)]
struct FfnRequest {
    layer: usize,
    hidden: Vec<f32>,
    expert_ids: Vec<u32>,
    expert_scores: Vec<f32>,
}

#[derive(Serialize)]
struct FfnResponse {
    output: Vec<f32>,
}

struct WorkerState {
    /// Layer index → expert shard for that layer
    shards: HashMap<usize, ExpertShard>,
}

async fn handle_ffn(
    State(state): State<Arc<WorkerState>>,
    Json(req): Json<FfnRequest>,
) -> Result<Json<FfnResponse>, StatusCode> {
    let shard = match state.shards.get(&req.layer) {
        Some(s) => s,
        None => return Err(StatusCode::NOT_FOUND),
    };
    let output =
        shard.expert_ffn(&req.hidden, &req.expert_ids, &req.expert_scores);
    Ok(Json(FfnResponse { output }))
}

async fn handle_health() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: expert_worker <model.gguf> <expert_indices...> [--port PORT] [--layers N]"
        );
        eprintln!(
            "Example: expert_worker model.gguf 0 1 2 3 4 --port 8001 --layers 32"
        );
        std::process::exit(1);
    }

    let model_path = PathBuf::from(&args[1]);

    let mut port = 8001u16;
    let mut n_layers = 32usize;
    let mut indices = Vec::new();

    let mut i = 2;
    while i < args.len() {
        if args[i] == "--port" {
            i += 1;
            port = args[i].parse().expect("port must be a number");
        } else if args[i] == "--layers" {
            i += 1;
            n_layers = args[i].parse().expect("layers must be a number");
        } else {
            indices.push(args[i].parse().expect("expert index must be a number"));
        }
        i += 1;
    }

    // Model config for granite3.1-moe:3b
    let d_model = 1536;
    let d_ff = 512;

    eprintln!("Loading experts {indices:?} for {n_layers} layers...");
    eprintln!("  model: {}", model_path.display());
    eprintln!("  d_model: {d_model}, d_ff: {d_ff}");

    let mut shards = HashMap::new();
    for layer in 0..n_layers {
        let shard = ExpertShard::load(
            &model_path,
            layer,
            &indices,
            d_model,
            d_ff,
        )
        .expect("failed to load expert shard");
        shards.insert(layer, shard);
    }

    eprintln!(
        "  loaded {} experts per layer, {} layers total",
        indices.len(),
        n_layers
    );

    let state = Arc::new(WorkerState { shards });

    let app = Router::new()
        .route("/ffn", post(handle_ffn))
        .route("/health", axum::routing::get(handle_health))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    eprintln!("Listening on {addr}");

    let listener = TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
