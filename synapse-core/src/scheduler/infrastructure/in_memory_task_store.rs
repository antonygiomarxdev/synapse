use std::collections::HashMap;
use std::sync::Mutex;

use crate::job::job_id::JobId;
use crate::scheduler::ports::TaskStore;
use crate::scheduler::task::Task;
use crate::scheduler::task_id::TaskId;
use crate::scheduler::task_status::TaskStatus;
use crate::shared::DomainError;

/// Thread-safe in-memory task store for V0.
pub struct InMemoryTaskStore {
    tasks: Mutex<HashMap<TaskId, Task>>,
}

impl InMemoryTaskStore {
    pub fn new() -> Self {
        Self { tasks: Mutex::new(HashMap::new()) }
    }
}

impl Default for InMemoryTaskStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskStore for InMemoryTaskStore {
    fn save(&self, task: &Task) -> Result<(), DomainError> {
        let mut tasks = self.tasks.lock().map_err(|e| DomainError::StorageError {
            message: format!("lock poisoned: {e}"),
        })?;
        tasks.insert(task.id, task.clone());
        Ok(())
    }

    fn find_by_id(&self, id: &TaskId) -> Result<Option<Task>, DomainError> {
        let tasks = self.tasks.lock().map_err(|e| DomainError::StorageError {
            message: format!("lock poisoned: {e}"),
        })?;
        Ok(tasks.get(id).cloned())
    }

    fn find_by_status(&self, status: &TaskStatus) -> Result<Vec<Task>, DomainError> {
        let tasks = self.tasks.lock().map_err(|e| DomainError::StorageError {
            message: format!("lock poisoned: {e}"),
        })?;
        Ok(tasks.values().filter(|t| t.status == *status).cloned().collect())
    }

    fn find_by_job_id(&self, job_id: &JobId) -> Result<Vec<Task>, DomainError> {
        let tasks = self.tasks.lock().map_err(|e| DomainError::StorageError {
            message: format!("lock poisoned: {e}"),
        })?;
        Ok(tasks.values().filter(|t| t.job_id == *job_id).cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::job::Message;
    use chrono::Utc;

    fn test_task() -> Task {
        Task::new(JobId::new(), "model".into(), Message { role: "user".into(), content: "hi".into() }, Utc::now())
    }

    #[test]
    fn save_and_find() {
        let store = InMemoryTaskStore::new();
        let task = test_task();
        let id = task.id;
        store.save(&task).unwrap();
        let found = store.find_by_id(&id).unwrap().unwrap();
        assert_eq!(found.id, id);
    }

    #[test]
    fn find_unknown_returns_none() {
        let store = InMemoryTaskStore::new();
        assert!(store.find_by_id(&TaskId::new()).unwrap().is_none());
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
        let completed = store.find_by_status(&TaskStatus::Completed).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(completed.len(), 1);
    }

    #[test]
    fn find_by_job_id() {
        let store = InMemoryTaskStore::new();
        let job_id = JobId::new();
        let mut t = test_task();
        t.job_id = job_id;
        store.save(&t).unwrap();
        let tasks = store.find_by_job_id(&job_id).unwrap();
        assert_eq!(tasks.len(), 1);
    }
}
