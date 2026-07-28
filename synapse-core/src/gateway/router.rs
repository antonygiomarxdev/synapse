use axum::{Json, http::StatusCode};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default = "default_priority")]
    pub priority: String,
    #[serde(default = "default_swarm_size")]
    pub swarm_size: u32,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
}

fn default_priority() -> String {
    "realtime".into()
}
fn default_swarm_size() -> u32 {
    5
}

pub async fn chat_completions(
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    Ok(Json(ChatResponse {
        id: "chatcmpl-0001".into(),
        object: "chat.completion".into(),
        created: 0,
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".into(),
                content: "Synapse swarm — inference endpoint stub.".into(),
            },
            finish_reason: "stop".into(),
        }],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    async fn test_app() -> axum::Router {
        axum::Router::new().route("/v1/chat/completions", axum::routing::post(chat_completions))
    }

    #[tokio::test]
    async fn chat_completions_returns_200() {
        let app = test_app().await;
        let body = serde_json::json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn chat_completions_echos_model() {
        let app = test_app().await;
        let body = serde_json::json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let resp_body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let chat: ChatResponse = serde_json::from_slice(&resp_body).unwrap();
        assert_eq!(chat.model, "kimi-k3");
    }

    #[tokio::test]
    async fn defaults_applied_without_explicit_fields() {
        let app = test_app().await;
        let body = serde_json::json!({
            "model": "kimi-k3",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}

#[cfg(test)]
mod default_tests {
    use super::*;

    #[test]
    fn default_priority_is_realtime() {
        assert_eq!(default_priority(), "realtime");
    }

    #[test]
    fn default_swarm_size_is_5() {
        assert_eq!(default_swarm_size(), 5);
    }

    #[test]
    fn default_priority_is_not_empty() {
        assert!(!default_priority().is_empty());
    }

    #[test]
    fn default_swarm_size_is_nonzero() {
        assert!(default_swarm_size() > 0);
    }
}
