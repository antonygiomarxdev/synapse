use std::sync::Arc;
use std::time::Instant;

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
const TASKS: usize = 10;

/// Serial: one request at a time to one Ollama instance.
async fn run_serial(
    ollama: &OllamaWorkerPort,
    prompts: &[String],
) -> (usize, std::time::Duration) {
    let wid = WorkerId::new("w-0");
    let start = Instant::now();
    let mut ok = 0;
    for prompt in prompts {
        let task = Task::new(
            JobId::new(),
            MODEL.into(),
            Message { role: "user".into(), content: prompt.clone() },
            Utc::now(),
        );
        if ollama.dispatch(&wid, &task).await.is_ok() {
            ok += 1;
        }
    }
    (ok, start.elapsed())
}

/// Concurrent single-instance: all requests to same Ollama.
async fn run_concurrent_single(
    ollama: Arc<OllamaWorkerPort>,
    prompts: &[String],
) -> (usize, std::time::Duration) {
    let start = Instant::now();
    let mut join_set = tokio::task::JoinSet::new();
    for prompt in prompts {
        let port = ollama.clone();
        let wid = WorkerId::new("w-0");
        let prompt = prompt.clone();
        join_set.spawn(async move {
            let task = Task::new(
                JobId::new(),
                MODEL.into(),
                Message { role: "user".into(), content: prompt },
                Utc::now(),
            );
            port.dispatch(&wid, &task).await
        });
    }
    let mut ok = 0;
    while let Some(r) = join_set.join_next().await {
        if r.ok().is_some() {
            ok += 1;
        }
    }
    (ok, start.elapsed())
}

/// Distributed: requests round-robined across TWO Ollama instances.
async fn run_distributed(
    ollama_a: Arc<OllamaWorkerPort>,
    ollama_b: Arc<OllamaWorkerPort>,
    prompts: &[String],
) -> (usize, std::time::Duration) {
    let start = Instant::now();
    let mut join_set = tokio::task::JoinSet::new();
    for (i, prompt) in prompts.iter().enumerate() {
        let port = if i % 2 == 0 {
            ollama_a.clone()
        } else {
            ollama_b.clone()
        };
        let wid = WorkerId::new("w-0");
        let prompt = prompt.clone();
        join_set.spawn(async move {
            let task = Task::new(
                JobId::new(),
                MODEL.into(),
                Message { role: "user".into(), content: prompt },
                Utc::now(),
            );
            port.dispatch(&wid, &task).await
        });
    }
    let mut ok = 0;
    while let Some(r) = join_set.join_next().await {
        if r.ok().is_some() {
            ok += 1;
        }
    }
    (ok, start.elapsed())
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

    // Health checks
    eprintln!("Checking Ollama instances...");
    match ollama_11434.health_check(&WorkerId::new("w-0")).await {
        Ok(true) => eprintln!("  :11434 OK"),
        _ => {
            eprintln!("  :11434 not running");
            std::process::exit(1);
        }
    }
    match ollama_11435.health_check(&WorkerId::new("w-1")).await {
        Ok(true) => eprintln!("  :11435 OK"),
        _ => {
            eprintln!("  :11435 not running");
            std::process::exit(1);
        }
    }

    let prompts: Vec<String> = (0..TASKS)
        .map(|i| format!("What is {i}+{i}? Reply number only."))
        .collect();

    eprintln!(
        "\nBenchmark: {TASKS} tasks, model={MODEL}\n"
    );

    // Warmup both instances
    eprintln!("Warming up...");
    for port in [&ollama_11434, &ollama_11435] {
        let task = Task::new(
            JobId::new(),
            MODEL.into(),
            Message { role: "user".into(), content: "hello".into() },
            Utc::now(),
        );
        let _ = port
            .dispatch(&WorkerId::new("w-0"), &task)
            .await;
    }
    eprintln!("  done\n");

    // Run each scenario 3 times, take median
    let runs = 3;

    // 1. Serial
    eprintln!("[1/3] Serial (1 Ollama, 1 req at a time):");
    let mut serial_times = Vec::new();
    for i in 0..runs {
        let (ok, t) = run_serial(&ollama_11434, &prompts).await;
        serial_times.push(t);
        eprintln!(
            "  run {}: {ok}/{TASKS} in {:.2}s",
            i + 1,
            t.as_secs_f64()
        );
    }
    serial_times.sort();

    // 2. Concurrent single instance
    eprintln!(
        "\n[2/3] Concurrent (1 Ollama, all at once):"
    );
    let mut single_times = Vec::new();
    for i in 0..runs {
        let (ok, t) = run_concurrent_single(
            ollama_11434.clone(),
            &prompts,
        )
        .await;
        single_times.push(t);
        eprintln!(
            "  run {}: {ok}/{TASKS} in {:.2}s",
            i + 1,
            t.as_secs_f64()
        );
    }
    single_times.sort();

    // 3. Distributed (2 Ollama instances)
    eprintln!(
        "\n[3/3] Distributed (2 Ollamas, round-robin):"
    );
    let mut dist_times = Vec::new();
    for i in 0..runs {
        let (ok, t) = run_distributed(
            ollama_11434.clone(),
            ollama_11435.clone(),
            &prompts,
        )
        .await;
        dist_times.push(t);
        eprintln!(
            "  run {}: {ok}/{TASKS} in {:.2}s",
            i + 1,
            t.as_secs_f64()
        );
    }
    dist_times.sort();

    // Summary
    let serial = serial_times[1].as_secs_f64();
    let single = single_times[1].as_secs_f64();
    let dist = dist_times[1].as_secs_f64();

    eprintln!("\n=== Results (median of {runs} runs) ===");
    eprintln!(
        "Serial:          {:.2}s  (baseline)",
        serial
    );
    eprintln!(
        "Concurrent/1GPU: {:.2}s  ({:.1}x)",
        single,
        serial / single
    );
    eprintln!(
        "Distributed/2GPU: {:.2}s  ({:.1}x)",
        dist,
        serial / dist
    );
    eprintln!(
        "\nDistributed vs Concurrent: {:.1}x",
        single / dist
    );
}
