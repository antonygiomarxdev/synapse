use super::job::{Job, JobResult};
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

    /// Transitions a job from Pending to Running.
    fn start(&self, id: &JobId) -> Result<(), DomainError>;

    /// Transitions a job to Completed with a result.
    fn complete(&self, id: &JobId, result: JobResult) -> Result<(), DomainError>;

    /// Transitions a job to Failed with a reason.
    fn fail(&self, id: &JobId, reason: String) -> Result<(), DomainError>;
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

        fn start(&self, id: &JobId) -> Result<(), DomainError> {
            let mut jobs = self.jobs.lock().unwrap();
            match jobs.get_mut(id) {
                Some(job) => job.transition_to(JobStatus::Running),
                None => Err(DomainError::JobNotFound { job_id: id.to_string() }),
            }
        }

        fn complete(&self, id: &JobId, result: crate::job::job::JobResult) -> Result<(), DomainError> {
            let mut jobs = self.jobs.lock().unwrap();
            match jobs.get_mut(id) {
                Some(job) => job.complete(result),
                None => Err(DomainError::JobNotFound { job_id: id.to_string() }),
            }
        }

        fn fail(&self, id: &JobId, reason: String) -> Result<(), DomainError> {
            let mut jobs = self.jobs.lock().unwrap();
            match jobs.get_mut(id) {
                Some(job) => job.fail(reason),
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
    fn start_transitions_to_running() {
        let store = InMemoryStore::new();
        let job = test_job();
        let id = job.id;
        store.save(&job).unwrap();
        store.start(&id).unwrap();
        let found = store.find_by_id(&id).unwrap().unwrap();
        assert_eq!(found.status, JobStatus::Running);
    }

    #[test]
    fn start_unknown_job() {
        let store = InMemoryStore::new();
        let id = JobId::new();
        let result = store.start(&id);
        assert!(matches!(result, Err(DomainError::JobNotFound { .. })));
    }

    #[test]
    fn complete_sets_result() {
        let store = InMemoryStore::new();
        let job = test_job();
        let id = job.id;
        store.save(&job).unwrap();
        store.start(&id).unwrap();
        store.complete(&id, crate::job::job::JobResult { text: "done".into(), tokens: 5 }).unwrap();
        let found = store.find_by_id(&id).unwrap().unwrap();
        assert_eq!(found.status, JobStatus::Completed);
        assert!(found.result.is_some());
    }

    #[test]
    fn fail_sets_error() {
        let store = InMemoryStore::new();
        let job = test_job();
        let id = job.id;
        store.save(&job).unwrap();
        store.fail(&id, "timeout".into()).unwrap();
        let found = store.find_by_id(&id).unwrap().unwrap();
        assert_eq!(found.status, JobStatus::Failed);
        assert_eq!(found.error.as_deref(), Some("timeout"));
    }
}
