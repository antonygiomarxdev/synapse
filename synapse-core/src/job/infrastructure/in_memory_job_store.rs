use std::collections::HashMap;
use std::sync::Mutex;

use crate::job::job::Job;
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

    fn update_status(&self, id: &JobId, status: JobStatus) -> Result<(), DomainError> {
        let mut jobs = self.jobs.lock().map_err(|e| DomainError::StorageError {
            message: format!("lock poisoned: {e}"),
        })?;
        match jobs.get_mut(id) {
            Some(job) => job.transition_to(status),
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
    fn update_status_success() {
        let store = InMemoryJobStore::new();
        let job = test_job();
        let id = job.id;
        store.save(&job).unwrap();
        store.update_status(&id, JobStatus::Running).unwrap();
        let found = store.find_by_id(&id).unwrap().unwrap();
        assert_eq!(found.status, JobStatus::Running);
    }

    #[test]
    fn update_status_unknown_id() {
        let store = InMemoryJobStore::new();
        let result = store.update_status(&JobId::new(), JobStatus::Running);
        assert!(matches!(result, Err(DomainError::JobNotFound { .. })));
    }

    #[test]
    fn update_status_rejects_invalid_transition() {
        let store = InMemoryJobStore::new();
        let job = test_job();
        let id = job.id;
        store.save(&job).unwrap();
        // Pending → Completed is invalid
        let result = store.update_status(&id, JobStatus::Completed);
        assert!(matches!(result, Err(DomainError::InvalidJobTransition { .. })));
        // Status unchanged
        let found = store.find_by_id(&id).unwrap().unwrap();
        assert_eq!(found.status, JobStatus::Pending);
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
