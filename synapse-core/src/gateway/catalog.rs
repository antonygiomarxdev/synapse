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
        id: "kimi-k27-code",
        name: "Kimi K2.7 Code",
        total_params: "~1T",
        active_params: "~30B",
        experts: 384,
        active_per_token: 32,
        context: 131072,
    }])
}
