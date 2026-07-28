use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
pub struct ModelEntry {
    pub id: &'static str,
    pub name: &'static str,
    pub total_params: &'static str,
    pub active_params: &'static str,
    pub experts: u32,
    pub active_per_token: u32,
    pub context: u32,
}

pub async fn list_models() -> Json<Vec<ModelEntry>> {
    Json(vec![ModelEntry {
        id: "kimi-k3",
        name: "Kimi K3",
        total_params: "2.8T",
        active_params: "~103B",
        experts: 896,
        active_per_token: 16,
        context: 1_000_000,
    }])
}
