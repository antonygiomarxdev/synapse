//! Integration tests: 50-job throughput and crash recovery.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Utc;

    use crate::job::job::{Job, Message, Priority};
    use crate::job::job_status::JobStatus;
    use crate::job::infrastructure::InMemoryJobStore;
    use crate::job::ports::JobStore;
    use crate::scheduler::infrastructure::{
        InMemoryTaskStore, MockWorkerPort,
    };
    use crate::scheduler::scheduler::Scheduler;
    use crate::scheduler::task_status::TaskStatus;
    use crate::scheduler::worker_id::WorkerId;
    use crate::scheduler::WorkerInfo;

    fn workers() -> Vec<WorkerInfo> {
        vec![
            WorkerInfo {
                id: WorkerId::new("w-0"),
                model: "model".into(),
                healthy: true,
            },
            WorkerInfo {
                id: WorkerId::new("w-1"),
                model: "model".into(),
                healthy: true,
            },
        ]
    }

    fn make_scheduler(
        workers: Vec<WorkerInfo>,
    ) -> (Scheduler, Arc<MockWorkerPort>) {
        let mock = Arc::new(MockWorkerPort::new());
        let scheduler = Scheduler::new(
            Arc::new(InMemoryTaskStore::new()),
            Arc::new(InMemoryJobStore::new()),
            mock.clone(),
            workers,
        );
        (scheduler, mock)
    }

    async fn run_jobs(
        scheduler: &Scheduler,
        job_store: &dyn JobStore,
        n: usize,
    ) -> (usize, usize) {
        let now = Utc::now();
        let mut job_ids = Vec::new();

        for i in 0..n {
            let job = Job::submit(
                "model".into(),
                vec![
                    Message {
                        role: "user".into(),
                        content: format!("msg-{i}-a"),
                    },
                    Message {
                        role: "user".into(),
                        content: format!("msg-{i}-b"),
                    },
                ],
                Priority::Normal,
            )
            .unwrap();
            job_ids.push(job.id);
            job_store.save(&job).unwrap();
            scheduler
                .decompose(
                    &job.id,
                    &job.messages,
                    &job.model,
                    now,
                )
                .unwrap();
        }

        let mut completed = 0;
        let mut failed = 0;
        for _ in 0..200 {
            let _ = scheduler.tick(now).await;
            completed = 0;
            failed = 0;
            for id in &job_ids {
                let job =
                    job_store.find_by_id(id).unwrap().unwrap();
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

    #[tokio::test]
    async fn fifty_jobs_all_complete_with_two_workers() {
        let (scheduler, _mock) = make_scheduler(workers());
        let job_store = scheduler.job_store.as_ref();

        let (completed, failed) =
            run_jobs(&scheduler, job_store, 50).await;

        assert_eq!(completed, 50, "all 50 jobs should complete");
        assert_eq!(failed, 0, "no jobs should fail");
        assert_eq!(scheduler.metrics.report().total_jobs, 50);
        assert_eq!(scheduler.metrics.report().completed_jobs, 50);
    }

    #[tokio::test]
    async fn both_workers_receive_tasks() {
        let (scheduler, mock) = make_scheduler(workers());
        let job_store = scheduler.job_store.as_ref();

        run_jobs(&scheduler, job_store, 10).await;

        let log = mock.dispatch_log();
        let w0_count = log
            .iter()
            .filter(|(w, _)| *w == WorkerId::new("w-0"))
            .count();
        let w1_count = log
            .iter()
            .filter(|(w, _)| *w == WorkerId::new("w-1"))
            .count();

        assert!(w0_count > 0, "worker-0 should have tasks");
        assert!(w1_count > 0, "worker-1 should have tasks");
    }

    #[tokio::test]
    async fn crash_recovery_worker_fails_mid_job() {
        let (scheduler, mock) = make_scheduler(workers());
        let job_store = scheduler.job_store.as_ref();
        let now = Utc::now();

        let job = Job::submit(
            "model".into(),
            (0..10)
                .map(|i| Message {
                    role: "user".into(),
                    content: format!("msg-{i}"),
                })
                .collect(),
            Priority::Normal,
        )
        .unwrap();
        let job_id = job.id;
        job_store.save(&job).unwrap();
        scheduler
            .decompose(&job.id, &job.messages, &job.model, now)
            .unwrap();

        let _ = scheduler.tick(now).await;
        mock.set_failing(WorkerId::new("w-0"));

        for _ in 0..100 {
            let _ = scheduler.tick(now).await;
            let job =
                job_store.find_by_id(&job_id).unwrap().unwrap();
            match job.status {
                JobStatus::Completed | JobStatus::Failed => break,
                _ => {}
            }
        }

        let job = job_store.find_by_id(&job_id).unwrap().unwrap();
        assert_eq!(
            job.status,
            JobStatus::Completed,
            "job should complete despite crash"
        );
    }

    #[tokio::test]
    async fn zero_orphaned_jobs_after_permanent_failure() {
        let (scheduler, mock) = make_scheduler(workers());
        let job_store = scheduler.job_store.as_ref();
        let now = Utc::now();

        mock.set_failing(WorkerId::new("w-0"));
        mock.set_failing(WorkerId::new("w-1"));

        let job = Job::submit(
            "model".into(),
            vec![Message {
                role: "user".into(),
                content: "test".into(),
            }],
            Priority::Normal,
        )
        .unwrap();
        let job_id = job.id;
        job_store.save(&job).unwrap();
        scheduler
            .decompose(&job.id, &job.messages, &job.model, now)
            .unwrap();

        for _ in 0..100 {
            let _ = scheduler.tick(now).await;
            let job =
                job_store.find_by_id(&job_id).unwrap().unwrap();
            match job.status {
                JobStatus::Completed | JobStatus::Failed => break,
                _ => {}
            }
        }

        let job = job_store.find_by_id(&job_id).unwrap().unwrap();
        assert_eq!(
            job.status,
            JobStatus::Failed,
            "job should be failed, not orphaned"
        );
        assert_eq!(scheduler.metrics.report().failed_jobs, 1);
    }

    #[tokio::test]
    async fn job_with_ten_prompts_dispatches_ten_tasks() {
        let (scheduler, _mock) = make_scheduler(workers());
        let job_store = scheduler.job_store.as_ref();
        let now = Utc::now();

        let messages: Vec<Message> = (0..10)
            .map(|i| Message {
                role: "user".into(),
                content: format!("prompt-{i}"),
            })
            .collect();

        let job = Job::submit(
            "model".into(),
            messages,
            Priority::Normal,
        )
        .unwrap();
        let job_id = job.id;
        job_store.save(&job).unwrap();
        let tasks = scheduler
            .decompose(&job.id, &job.messages, &job.model, now)
            .unwrap();
        assert_eq!(tasks.len(), 10);

        scheduler.tick(now).await.unwrap();

        let tasks =
            scheduler.task_store.find_by_job_id(&job_id).unwrap();
        assert_eq!(tasks.len(), 10);
        for task in &tasks {
            assert_eq!(task.status, TaskStatus::Completed);
        }

        let job = job_store.find_by_id(&job_id).unwrap().unwrap();
        assert_eq!(job.status, JobStatus::Completed);
    }

    #[tokio::test]
    async fn same_task_id_not_executed_twice() {
        let (scheduler, mock) = make_scheduler(workers());
        let job_store = scheduler.job_store.as_ref();
        let now = Utc::now();

        let job = Job::submit(
            "model".into(),
            vec![Message {
                role: "user".into(),
                content: "unique-prompt".into(),
            }],
            Priority::Normal,
        )
        .unwrap();
        job_store.save(&job).unwrap();
        let tasks = scheduler
            .decompose(&job.id, &job.messages, &job.model, now)
            .unwrap();
        let task_id = tasks[0].id;

        scheduler.tick(now).await.unwrap();

        let log = mock.dispatch_log();
        let dispatches = log
            .iter()
            .filter(|(_, tid)| *tid == task_id.to_string())
            .count();
        assert_eq!(dispatches, 1, "dispatched exactly once");

        scheduler.tick(now).await.unwrap();
        let log_after = mock.dispatch_log();
        let dispatches_after = log_after
            .iter()
            .filter(|(_, tid)| *tid == task_id.to_string())
            .count();
        assert_eq!(dispatches_after, 1, "not re-dispatched");
    }

    #[tokio::test]
    async fn crash_recovery_within_thirty_seconds() {
        let (scheduler, mock) = make_scheduler(workers());
        let job_store = scheduler.job_store.as_ref();

        let job = Job::submit(
            "model".into(),
            (0..6)
                .map(|i| Message {
                    role: "user".into(),
                    content: format!("msg-{i}"),
                })
                .collect(),
            Priority::Normal,
        )
        .unwrap();
        let job_id = job.id;
        job_store.save(&job).unwrap();

        let start = std::time::Instant::now();
        let now = Utc::now();
        scheduler
            .decompose(&job.id, &job.messages, &job.model, now)
            .unwrap();

        scheduler.tick(now).await.unwrap();
        mock.set_failing(WorkerId::new("w-0"));

        for i in 0..100 {
            let advanced_now =
                now + chrono::Duration::seconds(i * 5);
            let _ = scheduler.tick(advanced_now).await;
            let job =
                job_store.find_by_id(&job_id).unwrap().unwrap();
            match job.status {
                JobStatus::Completed | JobStatus::Failed => break,
                _ => {}
            }
        }

        let elapsed = start.elapsed();
        let job = job_store.find_by_id(&job_id).unwrap().unwrap();
        assert_eq!(
            job.status,
            JobStatus::Completed,
            "should complete despite crash"
        );
        assert!(
            elapsed.as_secs() < 30,
            "recovery should take <30s, took {:?}",
            elapsed
        );
    }
}
