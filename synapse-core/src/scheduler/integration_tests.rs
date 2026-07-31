//! Integration tests for the scheduler: 50-job throughput and crash recovery.
//!
//! These tests use MockWorkerPort to avoid requiring a real Ollama instance.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Utc;

    use crate::job::job::{Job, Message, Priority};
    use crate::job::job_status::JobStatus;
    use crate::job::infrastructure::InMemoryJobStore;
    use crate::job::ports::JobStore;
    use crate::scheduler::infrastructure::{InMemoryTaskStore, MockWorkerPort};
    use crate::scheduler::scheduler::Scheduler;
    use crate::scheduler::worker_id::WorkerId;
    use crate::scheduler::WorkerInfo;

    fn workers() -> Vec<WorkerInfo> {
        vec![
            WorkerInfo { id: WorkerId::new("w-0"), model: "granite3.1-moe:3b".into(), healthy: true },
            WorkerInfo { id: WorkerId::new("w-1"), model: "qwen3:8b".into(), healthy: true },
        ]
    }

    fn make_scheduler(workers: Vec<WorkerInfo>) -> (Scheduler, Arc<MockWorkerPort>) {
        let mock = Arc::new(MockWorkerPort::new());
        let scheduler = Scheduler::new(
            Arc::new(InMemoryTaskStore::new()),
            Arc::new(InMemoryJobStore::new()),
            mock.clone(),
            workers,
        );
        (scheduler, mock)
    }

    /// Submit N jobs with 2 messages each, run tick until all complete.
    fn run_jobs(scheduler: &Scheduler, job_store: &dyn JobStore, n: usize) -> (usize, usize) {
        let now = Utc::now();
        let mut job_ids = Vec::new();

        for i in 0..n {
            let job = Job::submit(
                "granite3.1-moe:3b".into(),
                vec![
                    Message { role: "user".into(), content: format!("msg-{i}-a") },
                    Message { role: "user".into(), content: format!("msg-{i}-b") },
                ],
                Priority::Normal,
            )
            .unwrap();
            job_ids.push(job.id);
            job_store.save(&job).unwrap();
            scheduler.decompose(&job.id, &job.messages, &job.model, now).unwrap();
        }

        // Run ticks until all jobs are terminal or we hit max iterations
        let mut completed = 0;
        let mut failed = 0;
        for _ in 0..200 {
            let _ = scheduler.tick(now);
            completed = 0;
            failed = 0;
            for id in &job_ids {
                let job = job_store.find_by_id(id).unwrap().unwrap();
                match job.status {
                    JobStatus::Completed => completed += 1,
                    JobStatus::Failed => failed += 1,
                    _ => {}
                }
            }
            if completed + failed == n {
                break;
            }
        }

        (completed, failed)
    }

    #[test]
    fn fifty_jobs_all_complete_with_two_workers() {
        let (scheduler, _mock) = make_scheduler(workers());
        let job_store = scheduler.job_store.as_ref();

        let (completed, failed) = run_jobs(&scheduler, job_store, 50);

        assert_eq!(completed, 50, "all 50 jobs should complete");
        assert_eq!(failed, 0, "no jobs should fail");
        assert_eq!(scheduler.metrics.report().total_jobs, 50);
        assert_eq!(scheduler.metrics.report().completed_jobs, 50);
    }

    #[test]
    fn both_workers_receive_tasks() {
        let (scheduler, mock) = make_scheduler(workers());
        let job_store = scheduler.job_store.as_ref();

        run_jobs(&scheduler, job_store, 10);

        let log = mock.dispatch_log();
        let w0_count = log.iter().filter(|(w, _)| *w == WorkerId::new("w-0")).count();
        let w1_count = log.iter().filter(|(w, _)| *w == WorkerId::new("w-1")).count();

        assert!(w0_count > 0, "worker-0 should have received tasks");
        assert!(w1_count > 0, "worker-1 should have received tasks");
    }

    #[test]
    fn crash_recovery_worker_fails_mid_job() {
        let (scheduler, mock) = make_scheduler(workers());
        let job_store = scheduler.job_store.as_ref();
        let now = Utc::now();

        // Submit a job with 10 messages
        let job = Job::submit(
            "model".into(),
            (0..10)
                .map(|i| Message { role: "user".into(), content: format!("msg-{i}") })
                .collect(),
            Priority::Normal,
        )
        .unwrap();
        let job_id = job.id;
        job_store.save(&job).unwrap();
        scheduler.decompose(&job.id, &job.messages, &job.model, now).unwrap();

        // First tick: dispatches tasks to both workers
        let _ = scheduler.tick(now);

        // Now simulate worker-0 crash
        mock.set_failing(WorkerId::new("w-0"));

        // Run remaining ticks — worker-1 should pick up retried tasks
        for _ in 0..100 {
            let _ = scheduler.tick(now);
            let job = job_store.find_by_id(&job_id).unwrap().unwrap();
            match job.status {
                JobStatus::Completed | JobStatus::Failed => break,
                _ => {}
            }
        }

        let job = job_store.find_by_id(&job_id).unwrap().unwrap();
        assert_eq!(job.status, JobStatus::Completed, "job should complete despite worker crash");
    }

    #[test]
    fn zero_orphaned_jobs_after_permanent_failure() {
        let (scheduler, mock) = make_scheduler(workers());
        let job_store = scheduler.job_store.as_ref();
        let now = Utc::now();

        // Make all workers fail permanently
        mock.set_failing(WorkerId::new("w-0"));
        mock.set_failing(WorkerId::new("w-1"));

        let job = Job::submit(
            "model".into(),
            vec![Message { role: "user".into(), content: "test".into() }],
            Priority::Normal,
        )
        .unwrap();
        let job_id = job.id;
        job_store.save(&job).unwrap();
        scheduler.decompose(&job.id, &job.messages, &job.model, now).unwrap();

        // Run ticks until terminal
        for _ in 0..100 {
            let _ = scheduler.tick(now);
            let job = job_store.find_by_id(&job_id).unwrap().unwrap();
            match job.status {
                JobStatus::Completed | JobStatus::Failed => break,
                _ => {}
            }
        }

        let job = job_store.find_by_id(&job_id).unwrap().unwrap();
        assert_eq!(job.status, JobStatus::Failed, "job should be failed, not orphaned");
        assert_eq!(scheduler.metrics.report().failed_jobs, 1);
    }
}
