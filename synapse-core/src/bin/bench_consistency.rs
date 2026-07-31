use std::sync::Arc;

use chrono::Utc;

use synapse_core::scheduler::infrastructure::ollama_worker_port::{
    OllamaWorkerPort, WorkerConfig,
};
use synapse_core::scheduler::ports::WorkerPort;
use synapse_core::scheduler::task::Task;
use synapse_core::job::job::Message;
use synapse_core::job::job_id::JobId;
use synapse_core::scheduler::worker_id::WorkerId;

const MODEL: &str = "granite3.1-moe:3b";

async fn ask(
    ollama: &OllamaWorkerPort,
    wid: &WorkerId,
    prompt: &str,
) -> String {
    let task = Task::new(
        JobId::new(),
        MODEL.into(),
        Message { role: "user".into(), content: prompt.into() },
        Utc::now(),
    );
    ollama.dispatch(wid, &task).await.unwrap_or_default()
}

#[tokio::main]
async fn main() {
    let ollama_11434 = Arc::new(OllamaWorkerPort::new(vec![
        WorkerConfig {
            id: WorkerId::new("w-0"),
            model: MODEL.into(),
            base_url: "http://localhost:11434".into(),
        },
    ]));
    let ollama_11435 = Arc::new(OllamaWorkerPort::new(vec![
        WorkerConfig {
            id: WorkerId::new("w-1"),
            model: MODEL.into(),
            base_url: "http://localhost:11435".into(),
        },
    ]));

    eprintln!("=== Consistency Test ===\n");
    eprintln!("Model: {MODEL}");
    eprintln!("Instance A: :11434");
    eprintln!("Instance B: :11435\n");

    let prompts = vec![
        "What is 2+2? Reply with just the number.",
        "What is the capital of France? Reply with just the city name.",
        "What is 7*8? Reply with just the number.",
        "Say hello in Spanish. One word only.",
        "What color is the sky? One word only.",
    ];

    let mut all_match = true;

    for prompt in &prompts {
        eprintln!("Prompt: {prompt}");

        // Ask both instances 3 times each
        let mut responses_a = Vec::new();
        let mut responses_b = Vec::new();

        for _ in 0..3 {
            responses_a.push(
                ask(&ollama_11434, &WorkerId::new("w-0"), prompt).await,
            );
            responses_b.push(
                ask(&ollama_11435, &WorkerId::new("w-1"), prompt).await,
            );
        }

        // Check if A's responses are consistent with each other
        let a_consistent = responses_a.windows(2).all(|w| w[0] == w[1]);
        let b_consistent = responses_b.windows(2).all(|w| w[0] == w[1]);
        let ab_match = responses_a[0] == responses_b[0];

        eprintln!("  A (x3): {:?} {}", responses_a[0],
            if a_consistent { "(consistent)" } else { "(VARIES)" });
        eprintln!("  B (x3): {:?} {}", responses_b[0],
            if b_consistent { "(consistent)" } else { "(VARIES)" });
        eprintln!("  A==B: {}", if ab_match { "YES" } else { "NO" });

        if !ab_match {
            all_match = false;
        }
        eprintln!();
    }

    if all_match {
        eprintln!("RESULT: All responses match across instances.");
    } else {
        eprintln!("RESULT: Some responses differ (expected with LLMs - non-deterministic).");
    }

    // Now test: same prompt, same instance, multiple times - is IT consistent?
    eprintln!("\n=== Self-consistency (same instance, same prompt, 5x) ===\n");
    let test_prompt = "What is 2+2? Reply with just the number.";
    eprintln!("Prompt: {test_prompt}");
    let mut self_responses = Vec::new();
    for _ in 0..5 {
        self_responses.push(
            ask(&ollama_11434, &WorkerId::new("w-0"), test_prompt).await,
        );
    }
    eprintln!("Responses: {:?}", self_responses);
    let self_consistent = self_responses.windows(2).all(|w| w[0] == w[1]);
    eprintln!("Consistent: {}", if self_consistent { "YES" } else { "NO (LLMs are non-deterministic by default)" });
}
