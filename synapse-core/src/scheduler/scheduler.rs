use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};

use super::metrics::MetricsCollector;
use super::ports::{TaskStore, WorkerPort};
use super::task::{Task, WorkerInfo};
use super::task_status::TaskStatus;
use crate::job::job::{Job, JobResult};
use crate::job::job_id::JobId;
use crate::job::ports::JobStore;
use crate::shared::DomainError;

/// Round-robin scheduler that dispatches tasks to known workers.
///
/// Owns the lifecycle of tasks: decomposes jobs into tasks,
/// dispatches pending tasks to workers, handles lease expiry,
/// retries failed tasks, and completes/fails jobs when all
/// their tasks reach a terminal state.
pub struct Scheduler {
    /// Task store for persisting and querying tasks.
    pub task_store: Arc<dyn TaskStore>,
    /// Job store for persisting and querying jobs.
    pub job_store: Arc<dyn JobStore>,
    worker_port: Arc<dyn WorkerPort>,
    workers: Vec<WorkerInfo>,
    next_worker_idx: std::sync::atomic::AtomicUsize,
    /// Collector for inference metrics (jobs, tasks, latencies).
    pub metrics: Arc<MetricsCollector>,
}

impl Scheduler {
    pub fn new(
        task_store: Arc<dyn TaskStore>,
        job_store: Arc<dyn JobStore>,
        worker_port: Arc<dyn WorkerPort>,
        workers: Vec<WorkerInfo>,
    ) -> Self {
        Self {
            task_store,
            job_store,
            worker_port,
            workers,
            next_worker_idx: std::sync::atomic::AtomicUsize::new(0),
            metrics: Arc::new(MetricsCollector::new()),
        }
    }

    /// Decomposes a job into one task per message.
    ///
    /// Transitions the job from Pending to Running via the JobStore port.
    /// Records a job submission in metrics.
    pub fn decompose(&self, job_id: &JobId, messages: &[crate::job::job::Message], model: &str, now: DateTime<Utc>) -> Result<Vec<Task>, DomainError> {
        self.job_store.start(job_id)?;
        self.metrics.record_job_submit();

        let tasks: Vec<Task> = messages
            .iter()
            .map(|msg| Task::new(*job_id, model.to_string(), msg.clone(), now))
            .collect();

        for task in &tasks {
            self.task_store.save(task)?;
        }

        Ok(tasks)
    }

    /// Picks the next healthy worker via round-robin.
    fn next_worker(&self) -> Option<&WorkerInfo> {
        let healthy: Vec<&WorkerInfo> = self.workers.iter().filter(|w| w.healthy).collect();
        if healthy.is_empty() {
            return None;
        }
        let idx = self.next_worker_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(healthy[idx % healthy.len()])
    }

    /// Processes one tick: dispatch pending, check leases, retry failures.
    ///
    /// Returns the number of tasks processed.
    pub fn tick(&self, now: DateTime<Utc>) -> Result<usize, DomainError> {
        let mut processed = 0;

        // 1. Dispatch pending tasks
        let pending = self.task_store.find_by_status(&TaskStatus::Pending)?;
        for mut task in pending {
            if task.is_permanently_failed() {
                continue;
            }
            self.dispatch_task(&mut task, now)?;
            processed += 1;
        }

        // 2. Check for expired leases
        let leased = self.task_store.find_by_status(&TaskStatus::Leased)?;
        for mut task in leased {
            if task.is_lease_expired(now) {
                task.fail("lease expired".into(), now)?;
                self.task_store.save(&task)?;
                self.metrics.record_task_retry();
                processed += 1;
            }
        }

        // 3. Retry failed tasks that haven't exceeded max retries
        let failed = self.task_store.find_by_status(&TaskStatus::Failed)?;
        for mut task in failed {
            if !task.is_permanently_failed() {
                task.retry(now)?;
                self.task_store.save(&task)?;
                self.metrics.record_task_retry();
                processed += 1;
            }
        }

        // 4. Check if any jobs should be finalized
        self.finalize_jobs()?;

        Ok(processed)
    }

    /// Dispatches a single task to a worker.
    ///
    /// Records queue time and execution time in metrics.
    fn dispatch_task(&self, task: &mut Task, now: DateTime<Utc>) -> Result<(), DomainError> {
        let worker = match self.next_worker() {
            Some(w) => w,
            None => return Err(DomainError::WorkerDispatchFailed {
                reason: "no healthy workers available".into(),
            }),
        };

        task.lease(worker.id.clone(), now)?;
        self.task_store.save(task)?;

        let start = Instant::now();
        match self.worker_port.dispatch(&worker.id, task) {
            Ok(_text) => {
                let exec_ms = start.elapsed().as_millis() as u64;
                self.metrics.record_task_dispatch(0, exec_ms);
                task.complete(now)?;
                self.task_store.save(task)?;
            }
            Err(_) => {
                task.fail("dispatch failed".into(), now)?;
                self.task_store.save(task)?;
            }
        }

        Ok(())
    }

    /// Checks all jobs for finalization (all tasks terminal → job complete/failed).
    ///
    /// Records job completion or failure in metrics.
    fn finalize_jobs(&self) -> Result<(), DomainError> {
        let running_jobs: Vec<Job> = {
            let all_jobs = self.job_store.list()?;
            all_jobs.into_iter().filter(|j| j.status == crate::job::job_status::JobStatus::Running).collect()
        };

        for job in running_jobs {
            let tasks = self.task_store.find_by_job_id(&job.id)?;
            if tasks.is_empty() {
                continue;
            }

            let all_completed = tasks.iter().all(|t| t.status == TaskStatus::Completed);
            let any_permanently_failed = tasks.iter().any(|t| t.is_permanently_failed());

            if all_completed {
                let total_tokens = tasks.len() as u32;
                let combined_text = tasks
                    .iter()
                    .map(|t| t.message.content.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                self.job_store.complete(&job.id, JobResult { text: combined_text, tokens: total_tokens })?;
                self.metrics.record_job_complete();
            } else if any_permanently_failed {
                self.job_store.fail(&job.id, "task retries exhausted".into())?;
                self.metrics.record_job_fail();
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::job::{Message, Priority};
    use crate::job::infrastructure::InMemoryJobStore;
    use crate::scheduler::infrastructure::{InMemoryTaskStore, MockWorkerPort};
    use crate::scheduler::worker_id::WorkerId;

    fn now() -> DateTime<Utc> {
        Utc::now()
    }

    fn make_scheduler_with_mock(workers: Vec<WorkerInfo>) -> (Scheduler, Arc<MockWorkerPort>) {
        let mock = Arc::new(MockWorkerPort::new());
        let scheduler = Scheduler::new(
            Arc::new(InMemoryTaskStore::new()),
            Arc::new(InMemoryJobStore::new()),
            mock.clone(),
            workers,
        );
        (scheduler, mock)
    }

    fn make_scheduler(workers: Vec<WorkerInfo>) -> Scheduler {
        make_scheduler_with_mock(workers).0
    }

    fn test_job() -> Job {
        Job::submit(
            "model".into(),
            vec![
                Message { role: "user".into(), content: "msg1".into() },
                Message { role: "user".into(), content: "msg2".into() },
            ],
            Priority::Normal,
        )
        .unwrap()
    }

    #[test]
    fn decompose_creates_one_task_per_message() {
        let scheduler = make_scheduler(vec![
            WorkerInfo { id: WorkerId::new("w-0"), model: "model".into(), healthy: true },
        ]);
        let job = test_job();
        scheduler.job_store.save(&job).unwrap();
        let tasks = scheduler.decompose(&job.id, &job.messages, &job.model, now()).unwrap();
        assert_eq!(tasks.len(), 2);
        let updated = scheduler.job_store.find_by_id(&job.id).unwrap().unwrap();
        assert_eq!(updated.status, crate::job::job_status::JobStatus::Running);
        assert_eq!(scheduler.metrics.report().total_jobs, 1);
    }

    #[test]
    fn decompose_saves_tasks_to_store() {
        let scheduler = make_scheduler(vec![
            WorkerInfo { id: WorkerId::new("w-0"), model: "model".into(), healthy: true },
        ]);
        let job = test_job();
        scheduler.job_store.save(&job).unwrap();
        let tasks = scheduler.decompose(&job.id, &job.messages, &job.model, now()).unwrap();
        for task in &tasks {
            let found = scheduler.task_store.find_by_id(&task.id).unwrap();
            assert!(found.is_some());
        }
    }

    #[test]
    fn tick_dispatches_pending_tasks() {
        let scheduler = make_scheduler(vec![
            WorkerInfo { id: WorkerId::new("w-0"), model: "model".into(), healthy: true },
        ]);
        let job = test_job();
        scheduler.job_store.save(&job).unwrap();
        scheduler.decompose(&job.id, &job.messages, &job.model, now()).unwrap();

        let processed = scheduler.tick(now()).unwrap();
        assert!(processed > 0);

        let tasks = scheduler.task_store.find_by_job_id(&job.id).unwrap();
        for task in &tasks {
            assert_eq!(task.status, TaskStatus::Completed);
        }
        assert_eq!(scheduler.metrics.report().total_tasks, 2);
    }

    #[test]
    fn tick_completes_job_when_all_tasks_done() {
        let scheduler = make_scheduler(vec![
            WorkerInfo { id: WorkerId::new("w-0"), model: "model".into(), healthy: true },
        ]);
        let job = test_job();
        let job_id = job.id;
        scheduler.job_store.save(&job).unwrap();
        scheduler.decompose(&job.id, &job.messages, &job.model, now()).unwrap();

        scheduler.tick(now()).unwrap();

        let updated_job = scheduler.job_store.find_by_id(&job_id).unwrap().unwrap();
        assert_eq!(updated_job.status, crate::job::job_status::JobStatus::Completed);
        assert_eq!(scheduler.metrics.report().completed_jobs, 1);
    }

    #[test]
    fn round_robin_distributes_across_workers() {
        let w0 = WorkerId::new("w-0");
        let w1 = WorkerId::new("w-1");
        let (scheduler, mock) = make_scheduler_with_mock(vec![
            WorkerInfo { id: w0.clone(), model: "model".into(), healthy: true },
            WorkerInfo { id: w1.clone(), model: "model".into(), healthy: true },
        ]);

        let job = Job::submit(
            "model".into(),
            vec![
                Message { role: "user".into(), content: "a".into() },
                Message { role: "user".into(), content: "b".into() },
                Message { role: "user".into(), content: "c".into() },
                Message { role: "user".into(), content: "d".into() },
            ],
            Priority::Normal,
        )
        .unwrap();
        scheduler.job_store.save(&job).unwrap();
        scheduler.decompose(&job.id, &job.messages, &job.model, now()).unwrap();
        scheduler.tick(now()).unwrap();

        let log = mock.dispatch_log();
        assert_eq!(log.len(), 4);
        assert_eq!(log[0].0, w0);
        assert_eq!(log[1].0, w1);
        assert_eq!(log[2].0, w0);
        assert_eq!(log[3].0, w1);
    }

    #[test]
    fn no_healthy_workers_returns_error() {
        let scheduler = make_scheduler(vec![
            WorkerInfo { id: WorkerId::new("w-0"), model: "model".into(), healthy: false },
        ]);
        let job = test_job();
        scheduler.job_store.save(&job).unwrap();
        scheduler.decompose(&job.id, &job.messages, &job.model, now()).unwrap();

        let result = scheduler.tick(now());
        assert!(matches!(result, Err(DomainError::WorkerDispatchFailed { .. })));
    }

    #[test]
    fn metrics_track_retries_on_failure() {
        let (scheduler, mock) = make_scheduler_with_mock(vec![
            WorkerInfo { id: WorkerId::new("w-0"), model: "model".into(), healthy: true },
        ]);
        mock.set_failing(WorkerId::new("w-0"));

        let job = test_job();
        scheduler.job_store.save(&job).unwrap();
        scheduler.decompose(&job.id, &job.messages, &job.model, now()).unwrap();

        let _ = scheduler.tick(now());
        let report = scheduler.metrics.report();
        assert!(report.retried_tasks > 0, "should have recorded retries");
    }
}
