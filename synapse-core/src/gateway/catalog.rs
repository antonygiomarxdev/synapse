use axum::Json;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub total_params: String,
    pub active_params: String,
    pub experts: u32,
    pub active_per_token: u32,
    pub context: u32,
}

pub async fn list_models() -> Json<Vec<ModelEntry>> {
    Json(vec![ModelEntry {
        id: "kimi-k3".into(),
        name: "Kimi K3".into(),
        total_params: "2.8T".into(),
        active_params: "~103B".into(),
        experts: 896,
        active_per_token: 16,
        context: 1_000_000,
    }])
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
        let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
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
        let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let models: Vec<ModelEntry> = serde_json::from_slice(&body).unwrap();
        assert_eq!(models[0].id, "kimi-k3");
        assert_eq!(models[0].experts, 896);
        assert_eq!(models[0].active_per_token, 16);
        assert_eq!(models[0].context, 1_000_000);
    }
}
