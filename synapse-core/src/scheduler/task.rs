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
    pub lease_expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Task {
    /// Creates a new task in `Pending` status.
    pub fn new(job_id: JobId, model: String, message: Message) -> Self {
        let now = Utc::now();
        Self {
            id: TaskId::new(),
            job_id,
            model,
            message,
            status: TaskStatus::Pending,
            retry_count: 0,
            worker_id: None,
            lease_expires_at: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Grants a lease to a worker.
    pub fn lease(&mut self, worker_id: WorkerId) -> Result<(), DomainError> {
        self.status = self.status.transition(TaskStatus::Leased)?;
        self.worker_id = Some(worker_id);
        self.lease_expires_at = Some(Utc::now() + chrono::Duration::seconds(LEASE_DURATION_SECS));
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Marks the task as completed.
    pub fn complete(&mut self) -> Result<(), DomainError> {
        self.status = self.status.transition(TaskStatus::Completed)?;
        self.lease_expires_at = None;
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Marks the task as failed and increments retry count.
    pub fn fail(&mut self, reason: String) -> Result<(), DomainError> {
        self.status = self.status.transition(TaskStatus::Failed)?;
        self.retry_count += 1;
        self.lease_expires_at = None;
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Re-enqueues a failed task for retry (Failed → Pending).
    pub fn retry(&mut self) -> Result<(), DomainError> {
        self.status = self.status.transition(TaskStatus::Pending)?;
        self.worker_id = None;
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Returns `true` if the lease has expired.
    pub fn is_lease_expired(&self) -> bool {
        match self.lease_expires_at {
            Some(expires) => Utc::now() > expires,
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
    use crate::job::job::Priority;

    fn test_task() -> Task {
        Task::new(
            JobId::new(),
            "model".into(),
            Message { role: "user".into(), content: "hi".into() },
        )
    }

    #[test]
    fn new_task_is_pending() {
        let task = test_task();
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.retry_count, 0);
        assert!(task.worker_id.is_none());
    }

    #[test]
    fn lease_sets_worker_and_expiry() {
        let mut task = test_task();
        let wid = WorkerId::new("w-0");
        task.lease(wid.clone()).unwrap();
        assert_eq!(task.status, TaskStatus::Leased);
        assert_eq!(task.worker_id, Some(wid));
        assert!(task.lease_expires_at.is_some());
    }

    #[test]
    fn lease_rejected_from_non_pending() {
        let mut task = test_task();
        task.lease(WorkerId::new("w-0")).unwrap();
        let r = task.lease(WorkerId::new("w-1"));
        assert!(matches!(r, Err(DomainError::InvalidTaskTransition { .. })));
    }

    #[test]
    fn complete_from_leased() {
        let mut task = test_task();
        task.lease(WorkerId::new("w-0")).unwrap();
        task.complete().unwrap();
        assert_eq!(task.status, TaskStatus::Completed);
        assert!(task.lease_expires_at.is_none());
    }

    #[test]
    fn fail_increments_retry_count() {
        let mut task = test_task();
        task.lease(WorkerId::new("w-0")).unwrap();
        task.fail("error".into()).unwrap();
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(task.retry_count, 1);
    }

    #[test]
    fn retry_re_enqueues() {
        let mut task = test_task();
        task.lease(WorkerId::new("w-0")).unwrap();
        task.fail("error".into()).unwrap();
        task.retry().unwrap();
        assert_eq!(task.status, TaskStatus::Pending);
        assert!(task.worker_id.is_none());
    }

    #[test]
    fn retry_rejected_from_non_failed() {
        let mut task = test_task();
        let r = task.retry();
        assert!(matches!(r, Err(DomainError::InvalidTaskTransition { .. })));
    }

    #[test]
    fn is_permanently_failed_after_max_retries() {
        let mut task = test_task();
        for _ in 0..MAX_RETRIES {
            task.lease(WorkerId::new("w-0")).unwrap();
            task.fail("error".into()).unwrap();
            if !task.is_permanently_failed() {
                task.retry().unwrap();
            }
        }
        assert!(task.is_permanently_failed());
    }

    #[test]
    fn lease_not_expired_when_fresh() {
        let mut task = test_task();
        task.lease(WorkerId::new("w-0")).unwrap();
        assert!(!task.is_lease_expired());
    }

    #[test]
    fn no_lease_is_not_expired() {
        let task = test_task();
        assert!(!task.is_lease_expired());
    }

    #[test]
    fn task_serde_roundtrip() {
        let mut task = test_task();
        task.lease(WorkerId::new("w-0")).unwrap();
        let json = serde_json::to_string(&task).unwrap();
        let parsed: Task = serde_json::from_str(&json).unwrap();
        assert_eq!(task, parsed);
    }
}
