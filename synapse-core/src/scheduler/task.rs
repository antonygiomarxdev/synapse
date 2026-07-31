use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::task_id::TaskId;
use super::task_status::TaskStatus;
use super::worker_id::WorkerId;
use crate::job::job::Message;
use crate::job::job_id::JobId;
use crate::shared::DomainError;

/// Maximum number of retries before a task is permanently failed.
pub const MAX_RETRIES: u32 = 3;

/// Default lease duration in seconds.
pub const LEASE_DURATION_SECS: i64 = 30;

/// A lease granted to a worker for executing a task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lease {
    pub task_id: TaskId,
    pub worker_id: WorkerId,
    pub granted_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

/// A unit of work derived from a [`Job`].
///
/// Each message in a job becomes one task. The scheduler dispatches tasks
/// to workers, manages leases, and retries on failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub job_id: JobId,
    pub model: String,
    pub message: Message,
    pub status: TaskStatus,
    pub retry_count: u32,
    pub worker_id: Option<WorkerId>,
    pub lease: Option<Lease>,
    pub failure_reason: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Task {
    /// Creates a new task in `Pending` status.
    pub fn new(
        job_id: JobId,
        model: String,
        message: Message,
        now: DateTime<Utc>,
    ) -> Self {
        Self {
            id: TaskId::new(),
            job_id,
            model,
            message,
            status: TaskStatus::Pending,
            retry_count: 0,
            worker_id: None,
            lease: None,
            failure_reason: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Grants a lease to a worker.
    pub fn lease(&mut self, worker_id: WorkerId, now: DateTime<Utc>) -> Result<(), DomainError> {
        self.status = self.status.transition(TaskStatus::Leased)?;
        self.worker_id = Some(worker_id.clone());
        self.lease = Some(Lease {
            task_id: self.id,
            worker_id,
            granted_at: now,
            expires_at: now + chrono::Duration::seconds(LEASE_DURATION_SECS),
        });
        self.updated_at = now;
        Ok(())
    }

    /// Marks the task as completed.
    pub fn complete(&mut self, now: DateTime<Utc>) -> Result<(), DomainError> {
        self.status = self.status.transition(TaskStatus::Completed)?;
        self.lease = None;
        self.updated_at = now;
        Ok(())
    }

    /// Marks the task as failed and increments retry count.
    pub fn fail(&mut self, reason: String, now: DateTime<Utc>) -> Result<(), DomainError> {
        self.status = self.status.transition(TaskStatus::Failed)?;
        self.retry_count += 1;
        self.failure_reason = Some(reason);
        self.lease = None;
        self.updated_at = now;
        Ok(())
    }

    /// Re-enqueues a failed task for retry (Failed → Pending).
    pub fn retry(&mut self, now: DateTime<Utc>) -> Result<(), DomainError> {
        self.status = self.status.transition(TaskStatus::Pending)?;
        self.worker_id = None;
        self.failure_reason = None;
        self.updated_at = now;
        Ok(())
    }

    /// Returns `true` if the lease has expired at the given time.
    pub fn is_lease_expired(&self, now: DateTime<Utc>) -> bool {
        match &self.lease {
            Some(lease) => now > lease.expires_at,
            None => false,
        }
    }

    /// Returns `true` if max retries exceeded.
    pub fn is_permanently_failed(&self) -> bool {
        self.retry_count >= MAX_RETRIES
    }
}

/// Metadata about an available worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerInfo {
    pub id: WorkerId,
    pub model: String,
    pub healthy: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> DateTime<Utc> {
        Utc::now()
    }

    fn test_task() -> Task {
        Task::new(
            JobId::new(),
            "model".into(),
            Message { role: "user".into(), content: "hi".into() },
            now(),
        )
    }

    #[test]
    fn new_task_is_pending() {
        let task = test_task();
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.retry_count, 0);
        assert!(task.worker_id.is_none());
        assert!(task.lease.is_none());
        assert!(task.failure_reason.is_none());
    }

    #[test]
    fn lease_sets_worker_and_lease_object() {
        let mut task = test_task();
        let wid = WorkerId::new("w-0");
        task.lease(wid.clone(), now()).unwrap();
        assert_eq!(task.status, TaskStatus::Leased);
        assert_eq!(task.worker_id, Some(wid));
        assert!(task.lease.is_some());
        let lease = task.lease.as_ref().unwrap();
        assert!(lease.expires_at > lease.granted_at);
    }

    #[test]
    fn lease_rejected_from_non_pending() {
        let mut task = test_task();
        task.lease(WorkerId::new("w-0"), now()).unwrap();
        let r = task.lease(WorkerId::new("w-1"), now());
        assert!(matches!(r, Err(DomainError::InvalidTaskTransition { .. })));
    }

    #[test]
    fn complete_from_leased() {
        let mut task = test_task();
        task.lease(WorkerId::new("w-0"), now()).unwrap();
        task.complete(now()).unwrap();
        assert_eq!(task.status, TaskStatus::Completed);
        assert!(task.lease.is_none());
    }

    #[test]
    fn fail_increments_retry_count_and_stores_reason() {
        let mut task = test_task();
        task.lease(WorkerId::new("w-0"), now()).unwrap();
        task.fail("timeout".into(), now()).unwrap();
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(task.retry_count, 1);
        assert_eq!(task.failure_reason.as_deref(), Some("timeout"));
    }

    #[test]
    fn retry_re_enqueues() {
        let n = now();
        let mut task = test_task();
        task.lease(WorkerId::new("w-0"), n).unwrap();
        task.fail("error".into(), n).unwrap();
        task.retry(n).unwrap();
        assert_eq!(task.status, TaskStatus::Pending);
        assert!(task.worker_id.is_none());
        assert!(task.failure_reason.is_none());
    }

    #[test]
    fn retry_rejected_from_non_failed() {
        let mut task = test_task();
        let r = task.retry(now());
        assert!(matches!(r, Err(DomainError::InvalidTaskTransition { .. })));
    }

    #[test]
    fn is_permanently_failed_after_max_retries() {
        let n = now();
        let mut task = test_task();
        for _ in 0..MAX_RETRIES {
            task.lease(WorkerId::new("w-0"), n).unwrap();
            task.fail("error".into(), n).unwrap();
            if !task.is_permanently_failed() {
                task.retry(n).unwrap();
            }
        }
        assert!(task.is_permanently_failed());
    }

    #[test]
    fn lease_not_expired_when_fresh() {
        let mut task = test_task();
        task.lease(WorkerId::new("w-0"), now()).unwrap();
        assert!(!task.is_lease_expired(now()));
    }

    #[test]
    fn no_lease_is_not_expired() {
        let task = test_task();
        assert!(!task.is_lease_expired(now()));
    }

    #[test]
    fn lease_expired_after_duration() {
        let mut task = test_task();
        let past = Utc::now() - chrono::Duration::seconds(60);
        task.lease(WorkerId::new("w-0"), past).unwrap();
        assert!(task.is_lease_expired(now()));
    }

    #[test]
    fn task_serde_roundtrip() {
        let mut task = test_task();
        task.lease(WorkerId::new("w-0"), now()).unwrap();
        let json = serde_json::to_string(&task).unwrap();
        let parsed: Task = serde_json::from_str(&json).unwrap();
        assert_eq!(task, parsed);
    }
}
