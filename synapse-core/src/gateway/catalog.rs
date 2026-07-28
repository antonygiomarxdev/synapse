use crate::model::{Catalog, ModelEntity, ModelId};
use axum::Json;
use serde::{Deserialize, Serialize};

/// Public-facing model entry in the API response.
///
/// A subset of [`ModelEntity`] fields suitable for client consumption.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub total_params: String,
    pub experts: u32,
    pub active_per_token: u32,
    pub context_window: u64,
    pub license: String,
}

impl From<&ModelEntity> for ModelEntry {
    fn from(m: &ModelEntity) -> Self {
        Self {
            id: m.id.to_string(),
            name: m.name.clone(),
            description: m.description.clone(),
            total_params: m.total_params.clone(),
            experts: m.experts,
            active_per_token: m.active_per_token,
            context_window: m.context_window,
            license: m.license.clone(),
        }
    }
}

/// Loads the catalog from config and returns models as JSON.
pub async fn list_models() -> Json<Vec<ModelEntry>> {
    let catalog = load_catalog();
    let entries: Vec<ModelEntry> = catalog.list().iter().map(ModelEntry::from).collect();
    Json(entries)
}

/// Loads the Synapse catalog from `config/models.toml`.
fn load_catalog() -> Catalog {
    let mut catalog = Catalog::new();
    // Hardcoded for now; Task 5.3 will load from config/models.toml dynamically.
    let kimi = ModelEntity::new(
        ModelId::new("kimi-k3").unwrap(),
        "Kimi K3".into(),
        "Moonshot AI's frontier MoE. 2.8T total params, ~103B active, 896 experts, KDA linear attention, 1M context. Open-weight (MIT modified).".into(),
        "2.8T".into(),
        896,
        16,
        1.5,
        12.0,
        1_000_000,
        "MIT".into(),
        "moonshotai/Kimi-K3".into(),
        None,
    );
    catalog.register(kimi).ok();
    catalog
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    async fn test_app() -> axum::Router {
        axum::Router::new().route("/v1/models", axum::routing::get(list_models))
    }

    #[tokio::test]
    async fn list_models_returns_200() {
        let app = test_app().await;
        let response = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn list_models_returns_non_empty_array() {
        let app = test_app().await;
        let response = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
        let models: Vec<ModelEntry> = serde_json::from_slice(&body).unwrap();
        assert!(!models.is_empty());
    }

    #[tokio::test]
    async fn kimi_k3_is_listed_with_correct_specs() {
        let app = test_app().await;
        let response = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
        let models: Vec<ModelEntry> = serde_json::from_slice(&body).unwrap();
        let kimi = models.iter().find(|m| m.id == "kimi-k3").unwrap();
        assert_eq!(kimi.experts, 896);
        assert_eq!(kimi.active_per_token, 16);
        assert_eq!(kimi.context_window, 1_000_000);
    }

    #[tokio::test]
    async fn model_entries_have_required_fields() {
        let app = test_app().await;
        let response = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
        let models: Vec<ModelEntry> = serde_json::from_slice(&body).unwrap();
        for model in &models {
            assert!(!model.id.is_empty());
            assert!(!model.name.is_empty());
            assert!(model.experts > 0);
            assert!(model.active_per_token > 0);
            assert!(!model.license.is_empty());
        }
    }
}
