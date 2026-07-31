use std::path::PathBuf;
use std::sync::Arc;

use axum::{extract::State, http::StatusCode, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use synapse_core::native_moe::expert_shard::ExpertShard;

#[derive(Deserialize)]
struct FfnRequest {
    hidden: Vec<f32>,
    expert_ids: Vec<u32>,
    expert_scores: Vec<f32>,
}

#[derive(Serialize)]
struct FfnResponse {
    output: Vec<f32>,
}

struct WorkerState {
    shard: ExpertShard,
}

async fn handle_ffn(
    State(state): State<Arc<WorkerState>>,
    Json(req): Json<FfnRequest>,
) -> Result<Json<FfnResponse>, StatusCode> {
    let output =
        state.shard.expert_ffn(&req.hidden, &req.expert_ids, &req.expert_scores);
    Ok(Json(FfnResponse { output }))
}

async fn handle_health() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: expert_worker <model.gguf> <layer> <expert_indices...> [--port PORT]"
        );
        eprintln!(
            "Example: expert_worker model.gguf 0 0 1 2 3 4 5 6 7 8 9 --port 8001"
        );
        std::process::exit(1);
    }

    let model_path = PathBuf::from(&args[1]);
    let layer: usize = args[2].parse().expect("layer must be a number");

    let mut port = 8001u16;
    let mut indices = Vec::new();

    let mut i = 3;
    while i < args.len() {
        if args[i] == "--port" {
            i += 1;
            port = args[i].parse().expect("port must be a number");
        } else {
            indices.push(args[i].parse().expect("expert index must be a number"));
        }
        i += 1;
    }

    // Model config for granite3.1-moe:3b
    // TODO: read from GGUF metadata
    let d_model = 1536;
    let d_ff = 512;

    eprintln!("Loading experts {indices:?} from layer {layer}...");
    eprintln!("  model: {}", model_path.display());
    eprintln!("  d_model: {d_model}, d_ff: {d_ff}");

    let shard = ExpertShard::load(
        &model_path,
        layer,
        &indices,
        d_model,
        d_ff,
    )
    .expect("failed to load expert shard");

    eprintln!(
        "  loaded {} experts: {:?}",
        shard.experts.len(),
        shard.indices
    );

    let state = Arc::new(WorkerState { shard });

    let app = Router::new()
        .route("/ffn", post(handle_ffn))
        .route("/health", axum::routing::get(handle_health))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    eprintln!("Listening on {addr}");

    let listener = TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
