use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;

use synapse_core::job::job::{Job, Message, Priority};
use synapse_core::job::job_status::JobStatus;
use synapse_core::job::infrastructure::InMemoryJobStore;
use synapse_core::job::ports::JobStore;
use synapse_core::scheduler::infrastructure::InMemoryTaskStore;
use synapse_core::scheduler::infrastructure::MockWorkerPort;
use synapse_core::scheduler::metrics::MetricsReport;
use synapse_core::scheduler::scheduler::Scheduler;
use synapse_core::scheduler::worker_id::WorkerId;
use synapse_core::scheduler::WorkerInfo;

const MODEL: &str = "granite3.1-moe:3b";
const JOBS: usize = 50;
const MESSAGES_PER_JOB: usize = 2;
const MAX_TICKS: usize = 200;

struct BenchResult {
    config: String,
    jobs_ok: usize,
    report: MetricsReport,
    wall_clock_ms: u128,
}

fn worker(id: &str) -> WorkerInfo {
    WorkerInfo {
        id: WorkerId::new(id),
        model: MODEL.into(),
        healthy: true,
    }
}

fn run_scenario(
    name: &str,
    mock: Arc<MockWorkerPort>,
    workers: Vec<WorkerInfo>,
    n_jobs: usize,
    fail_worker: Option<WorkerId>,
    fail_after_jobs: usize,
) -> BenchResult {
    let task_store = Arc::new(InMemoryTaskStore::new());
    let job_store = Arc::new(InMemoryJobStore::new());
    let scheduler = Scheduler::new(
        task_store,
        job_store.clone(),
        mock.clone(),
        workers,
    );

    let start = Instant::now();
    let now = Utc::now();

    let mut job_ids = Vec::new();
    for i in 0..n_jobs {
        let messages: Vec<Message> = (0..MESSAGES_PER_JOB)
            .map(|j| Message {
                role: "user".into(),
                content: format!("msg-{i}-{j}"),
            })
            .collect();
        let job = Job::submit(
            MODEL.into(),
            messages,
            Priority::Normal,
        )
        .unwrap();
        job_ids.push(job.id);
        job_store.save(&job).unwrap();
        scheduler
            .decompose(&job.id, &job.messages, &job.model, now)
            .unwrap();
    }

    let mut crashed = false;
    for _tick in 0..MAX_TICKS {
        if !crashed {
            if let Some(ref wid) = fail_worker {
                let completed = job_ids
                    .iter()
                    .filter(|id| {
                        job_store
                            .find_by_id(id)
                            .unwrap()
                            .map(|j| j.status == JobStatus::Completed)
                            .unwrap_or(false)
                    })
                    .count();
                if completed >= fail_after_jobs {
                    mock.set_failing(wid.clone());
                    crashed = true;
                }
            }
        }

        let _ = scheduler.tick(now);

        let all_terminal = job_ids.iter().all(|id| {
            job_store
                .find_by_id(id)
                .unwrap()
                .map(|j| {
                    j.status == JobStatus::Completed
                        || j.status == JobStatus::Failed
                })
                .unwrap_or(false)
        });
        if all_terminal {
            break;
        }
    }

    let wall_clock_ms = start.elapsed().as_millis();
    let mut completed = 0;
    for id in &job_ids {
        let job = job_store.find_by_id(id).unwrap().unwrap();
        if job.status == JobStatus::Completed {
            completed += 1;
        }
    }

    BenchResult {
        config: name.to_string(),
        jobs_ok: completed,
        report: scheduler.metrics.report(),
        wall_clock_ms,
    }
}

fn main() {
    eprintln!("=== V0 Benchmark ===\n");

    // Scenario 1: Single worker
    eprintln!("[1/3] Single worker baseline...");
    let mock1 = Arc::new(MockWorkerPort::new());
    let r1 = run_scenario(
        "1 worker",
        mock1,
        vec![worker("w-0")],
        JOBS,
        None,
        0,
    );

    // Scenario 2: Multi worker (2 workers)
    eprintln!("[2/3] Multi worker (2 workers)...");
    let mock2 = Arc::new(MockWorkerPort::new());
    let r2 = run_scenario(
        "2 workers",
        mock2,
        vec![worker("w-0"), worker("w-1")],
        JOBS,
        None,
        0,
    );

    // Scenario 3: Crash recovery (worker-0 fails mid-job)
    eprintln!("[3/3] Crash recovery (worker-0 fails after half)...");
    let mock3 = Arc::new(MockWorkerPort::new());
    let r3 = run_scenario(
        "2 workers + crash",
        mock3,
        vec![worker("w-0"), worker("w-1")],
        JOBS,
        Some(WorkerId::new("w-0")),
        JOBS / 2,
    );

    // Generate report
    let date = Utc::now().format("%Y-%m-%d").to_string();
    let report = generate_report(&date, &[r1, r2, r3]);

    let path = format!("docs/benchmarks/v0-{date}.md");
    std::fs::create_dir_all("docs/benchmarks").unwrap();
    std::fs::write(&path, &report).unwrap();

    eprintln!("\nReport written to {path}");
    println!("\n{report}");
}

fn generate_report(
    date: &str,
    results: &[BenchResult],
) -> String {
    let mut md = String::new();

    md.push_str(&format!(
        "# V0 Benchmark — {date}\n\n"
    ));
    md.push_str("## Configuration\n\n");
    md.push_str(&format!(
        "- **Model:** {MODEL}\n"
    ));
    md.push_str(&format!(
        "- **Jobs:** {JOBS} × {MESSAGES_PER_JOB} messages\n\n"
    ));

    md.push_str("## Results\n\n");
    md.push_str(
        "| Config | Jobs | OK% | Retries | Tokens | p50 exec | p95 exec | p50 queue | Wall |\n",
    );
    md.push_str(
        "|--------|------|-----|---------|--------|----------|----------|-----------|------|\n",
    );

    for r in results {
        let ok_pct = if r.report.total_jobs > 0 {
            r.report.success_rate * 100.0
        } else {
            0.0
        };
        let wall_s = r.wall_clock_ms as f64 / 1000.0;

        md.push_str(&format!(
            "| {} | {} | {:.0}% | {} | {} | {}ms | {}ms | {}ms | {:.1}s |\n",
            r.config,
            r.jobs_ok,
            ok_pct,
            r.report.retried_tasks,
            r.report.tokens_total,
            r.report.execution_time_p50_ms,
            r.report.execution_time_p95_ms,
            r.report.queue_time_p50_ms,
            wall_s,
        ));
    }

    md.push_str("\n## Notes\n\n");
    md.push_str(
        "- Crash scenario: worker-0 marked failing after half the jobs\n",
    );
    md.push_str(
        "- p50/p95 are per-task execution times\n",
    );

    md
}
