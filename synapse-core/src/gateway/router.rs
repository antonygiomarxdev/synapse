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

#[derive(Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
}

#[derive(Serialize)]
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
