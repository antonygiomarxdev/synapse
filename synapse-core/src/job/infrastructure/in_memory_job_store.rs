use std::collections::HashMap;
use std::sync::Mutex;

use crate::job::job::{Job, JobResult};
use crate::job::job_id::JobId;
use crate::job::job_status::JobStatus;
use crate::job::ports::JobStore;
use crate::shared::DomainError;

/// Thread-safe in-memory job store for V0.
///
/// Stores jobs in a `Mutex<HashMap>`. Suitable for single-node development
/// and testing. Not suitable for production multi-process deployments.
pub struct InMemoryJobStore {
    jobs: Mutex<HashMap<JobId, Job>>,
}

impl InMemoryJobStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self { jobs: Mutex::new(HashMap::new()) }
    }
}

impl Default for InMemoryJobStore {
    fn default() -> Self {
        Self::new()
    }
}

impl JobStore for InMemoryJobStore {
    fn save(&self, job: &Job) -> Result<(), DomainError> {
        let mut jobs = self.jobs.lock().map_err(|e| DomainError::StorageError {
            message: format!("lock poisoned: {e}"),
        })?;
        jobs.insert(job.id, job.clone());
        Ok(())
    }

    fn find_by_id(&self, id: &JobId) -> Result<Option<Job>, DomainError> {
        let jobs = self.jobs.lock().map_err(|e| DomainError::StorageError {
            message: format!("lock poisoned: {e}"),
        })?;
        Ok(jobs.get(id).cloned())
    }

    fn list(&self) -> Result<Vec<Job>, DomainError> {
        let jobs = self.jobs.lock().map_err(|e| DomainError::StorageError {
            message: format!("lock poisoned: {e}"),
        })?;
        Ok(jobs.values().cloned().collect())
    }

    fn start(&self, id: &JobId) -> Result<(), DomainError> {
        let mut jobs = self.jobs.lock().map_err(|e| DomainError::StorageError {
            message: format!("lock poisoned: {e}"),
        })?;
        match jobs.get_mut(id) {
            Some(job) => job.transition_to(JobStatus::Running),
            None => Err(DomainError::JobNotFound { job_id: id.to_string() }),
        }
    }

    fn complete(&self, id: &JobId, result: JobResult) -> Result<(), DomainError> {
        let mut jobs = self.jobs.lock().map_err(|e| DomainError::StorageError {
            message: format!("lock poisoned: {e}"),
        })?;
        match jobs.get_mut(id) {
            Some(job) => job.complete(result),
            None => Err(DomainError::JobNotFound { job_id: id.to_string() }),
        }
    }

    fn fail(&self, id: &JobId, reason: String) -> Result<(), DomainError> {
        let mut jobs = self.jobs.lock().map_err(|e| DomainError::StorageError {
            message: format!("lock poisoned: {e}"),
        })?;
        match jobs.get_mut(id) {
            Some(job) => job.fail(reason),
            None => Err(DomainError::JobNotFound { job_id: id.to_string() }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::job::{Message, Priority};
    use crate::job::ports::JobStore;

    fn test_job() -> Job {
        Job::submit(
            "model".into(),
            vec![Message { role: "user".into(), content: "hi".into() }],
            Priority::Normal,
        )
        .unwrap()
    }

    #[test]
    fn save_and_find() {
        let store = InMemoryJobStore::new();
        let job = test_job();
        let id = job.id;
        store.save(&job).unwrap();
        let found = store.find_by_id(&id).unwrap().unwrap();
        assert_eq!(found.id, id);
        assert_eq!(found.status, JobStatus::Pending);
    }

    #[test]
    fn save_overwrites_existing() {
        let store = InMemoryJobStore::new();
        let mut job = test_job();
        let id = job.id;
        store.save(&job).unwrap();
        job.status = JobStatus::Running;
        store.save(&job).unwrap();
        let found = store.find_by_id(&id).unwrap().unwrap();
        assert_eq!(found.status, JobStatus::Running);
    }

    #[test]
    fn find_unknown_returns_none() {
        let store = InMemoryJobStore::new();
        assert!(store.find_by_id(&JobId::new()).unwrap().is_none());
    }

    #[test]
    fn list_returns_all_saved() {
        let store = InMemoryJobStore::new();
        let j1 = test_job();
        let j2 = test_job();
        store.save(&j1).unwrap();
        store.save(&j2).unwrap();
        let all = store.list().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn list_empty_store() {
        let store = InMemoryJobStore::new();
        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn start_transitions_to_running() {
        let store = InMemoryJobStore::new();
        let job = test_job();
        let id = job.id;
        store.save(&job).unwrap();
        store.start(&id).unwrap();
        let found = store.find_by_id(&id).unwrap().unwrap();
        assert_eq!(found.status, JobStatus::Running);
    }

    #[test]
    fn start_unknown_id() {
        let store = InMemoryJobStore::new();
        let result = store.start(&JobId::new());
        assert!(matches!(result, Err(DomainError::JobNotFound { .. })));
    }

    #[test]
    fn complete_sets_result() {
        let store = InMemoryJobStore::new();
        let job = test_job();
        let id = job.id;
        store.save(&job).unwrap();
        store.start(&id).unwrap();
        store.complete(&id, JobResult { text: "done".into(), tokens: 5 }).unwrap();
        let found = store.find_by_id(&id).unwrap().unwrap();
        assert_eq!(found.status, JobStatus::Completed);
        assert!(found.result.is_some());
    }

    #[test]
    fn fail_sets_error() {
        let store = InMemoryJobStore::new();
        let job = test_job();
        let id = job.id;
        store.save(&job).unwrap();
        store.fail(&id, "timeout".into()).unwrap();
        let found = store.find_by_id(&id).unwrap().unwrap();
        assert_eq!(found.status, JobStatus::Failed);
        assert_eq!(found.error.as_deref(), Some("timeout"));
    }

    #[test]
    fn hundred_concurrent_jobs_all_retrievable() {
        let store = InMemoryJobStore::new();
        let mut ids = Vec::new();

        for _ in 0..100 {
            let job = test_job();
            ids.push(job.id);
            store.save(&job).unwrap();
        }

        assert_eq!(store.list().unwrap().len(), 100);

        for id in &ids {
            let found = store.find_by_id(id).unwrap();
            assert!(found.is_some(), "job {id} should be retrievable");
            assert_eq!(found.unwrap().status, JobStatus::Pending);
        }
    }
}
