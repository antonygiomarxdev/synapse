use super::task::Task;
use super::task_id::TaskId;
use super::task_status::TaskStatus;
use super::worker_id::WorkerId;
use crate::shared::DomainError;

/// Port for persisting and retrieving tasks.
pub trait TaskStore: Send + Sync {
    fn save(&self, task: &Task) -> Result<(), DomainError>;
    fn find_by_id(&self, id: &TaskId) -> Result<Option<Task>, DomainError>;
    fn find_by_status(&self, status: &TaskStatus) -> Result<Vec<Task>, DomainError>;
    fn find_by_job_id(&self, job_id: &crate::job::job_id::JobId) -> Result<Vec<Task>, DomainError>;
}

/// Port for dispatching tasks to inference workers.
///
/// Infrastructure adapters implement this with real workers (vLLM, Ollama)
/// or mock workers for testing. All methods are async to allow concurrent
/// dispatch across multiple workers.
#[async_trait::async_trait]
pub trait WorkerPort: Send + Sync {
    /// Dispatches a task to a worker and returns the generated text.
    async fn dispatch(&self, worker_id: &WorkerId, task: &Task) -> Result<String, DomainError>;

    /// Checks if the given worker is healthy and reachable.
    async fn health_check(&self, worker_id: &WorkerId) -> Result<bool, DomainError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::job::{Message, Priority};
    use crate::job::job_id::JobId;
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct InMemoryTaskStore {
        tasks: Mutex<HashMap<TaskId, Task>>,
    }

    impl InMemoryTaskStore {
        fn new() -> Self {
            Self { tasks: Mutex::new(HashMap::new()) }
        }
    }

    impl TaskStore for InMemoryTaskStore {
        fn save(&self, task: &Task) -> Result<(), DomainError> {
            self.tasks.lock().unwrap().insert(task.id, task.clone());
            Ok(())
        }

        fn find_by_id(&self, id: &TaskId) -> Result<Option<Task>, DomainError> {
            Ok(self.tasks.lock().unwrap().get(id).cloned())
        }

        fn find_by_status(&self, status: &TaskStatus) -> Result<Vec<Task>, DomainError> {
            Ok(self.tasks.lock().unwrap().values().filter(|t| t.status == *status).cloned().collect())
        }

        fn find_by_job_id(&self, job_id: &JobId) -> Result<Vec<Task>, DomainError> {
            Ok(self.tasks.lock().unwrap().values().filter(|t| t.job_id == *job_id).cloned().collect())
        }
    }

    fn test_task() -> Task {
        Task::new(JobId::new(), "model".into(), Message { role: "user".into(), content: "hi".into() }, chrono::Utc::now())
    }

    #[test]
    fn save_and_find() {
        let store = InMemoryTaskStore::new();
        let task = test_task();
        let id = task.id;
        store.save(&task).unwrap();
        assert!(store.find_by_id(&id).unwrap().is_some());
    }

    #[test]
    fn find_by_status_filters() {
        let store = InMemoryTaskStore::new();
        let t1 = test_task();
        let mut t2 = test_task();
        t2.status = TaskStatus::Completed;
        store.save(&t1).unwrap();
        store.save(&t2).unwrap();
        let pending = store.find_by_status(&TaskStatus::Pending).unwrap();
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn find_by_job_id() {
        let store = InMemoryTaskStore::new();
        let job_id = JobId::new();
        let mut t1 = test_task();
        t1.job_id = job_id;
        let mut t2 = test_task();
        t2.job_id = job_id;
        store.save(&t1).unwrap();
        store.save(&t2).unwrap();
        let tasks = store.find_by_job_id(&job_id).unwrap();
        assert_eq!(tasks.len(), 2);
    }
}
