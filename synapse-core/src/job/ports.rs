use super::job::Job;
use super::job_id::JobId;
use super::job_status::JobStatus;
use crate::shared::DomainError;

/// Port for persisting and retrieving jobs.
///
/// Infrastructure adapters implement this with concrete storage
/// (in-memory, SQLite, etc.). The domain layer depends only on this trait.
pub trait JobStore: Send + Sync {
    /// Persists a job. Overwrites if a job with the same ID exists.
    fn save(&self, job: &Job) -> Result<(), DomainError>;

    /// Finds a job by its ID.
    fn find_by_id(&self, id: &JobId) -> Result<Option<Job>, DomainError>;

    /// Lists all jobs.
    fn list(&self) -> Result<Vec<Job>, DomainError>;

    /// Updates only the status of a job.
    fn update_status(&self, id: &JobId, status: JobStatus) -> Result<(), DomainError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::job::{Message, Priority};

    /// In-memory implementation to validate the trait contract.
    struct InMemoryStore {
        jobs: std::sync::Mutex<std::collections::HashMap<JobId, Job>>,
    }

    impl InMemoryStore {
        fn new() -> Self {
            Self { jobs: std::sync::Mutex::new(std::collections::HashMap::new()) }
        }
    }

    impl JobStore for InMemoryStore {
        fn save(&self, job: &Job) -> Result<(), DomainError> {
            let mut jobs = self.jobs.lock().unwrap();
            jobs.insert(job.id, job.clone());
            Ok(())
        }

        fn find_by_id(&self, id: &JobId) -> Result<Option<Job>, DomainError> {
            let jobs = self.jobs.lock().unwrap();
            Ok(jobs.get(id).cloned())
        }

        fn list(&self) -> Result<Vec<Job>, DomainError> {
            let jobs = self.jobs.lock().unwrap();
            Ok(jobs.values().cloned().collect())
        }

        fn update_status(&self, id: &JobId, status: JobStatus) -> Result<(), DomainError> {
            let mut jobs = self.jobs.lock().unwrap();
            match jobs.get_mut(id) {
                Some(job) => job.transition_to(status),
                None => Err(DomainError::JobNotFound { job_id: id.to_string() }),
            }
        }
    }

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
        let store = InMemoryStore::new();
        let job = test_job();
        let id = job.id;
        store.save(&job).unwrap();
        let found = store.find_by_id(&id).unwrap().unwrap();
        assert_eq!(found.id, id);
    }

    #[test]
    fn find_returns_none_for_unknown() {
        let store = InMemoryStore::new();
        let id = JobId::new();
        assert!(store.find_by_id(&id).unwrap().is_none());
    }

    #[test]
    fn list_returns_all() {
        let store = InMemoryStore::new();
        let j1 = test_job();
        let j2 = test_job();
        store.save(&j1).unwrap();
        store.save(&j2).unwrap();
        assert_eq!(store.list().unwrap().len(), 2);
    }

    #[test]
    fn update_status_changes_status() {
        let store = InMemoryStore::new();
        let job = test_job();
        let id = job.id;
        store.save(&job).unwrap();
        store.update_status(&id, JobStatus::Running).unwrap();
        let found = store.find_by_id(&id).unwrap().unwrap();
        assert_eq!(found.status, JobStatus::Running);
    }

    #[test]
    fn update_status_unknown_job() {
        let store = InMemoryStore::new();
        let id = JobId::new();
        let result = store.update_status(&id, JobStatus::Running);
        assert!(matches!(result, Err(DomainError::JobNotFound { .. })));
    }
}
