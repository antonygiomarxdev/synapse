use std::sync::Mutex;

use crate::scheduler::ports::WorkerPort;
use crate::scheduler::task::Task;
use crate::scheduler::worker_id::WorkerId;
use crate::shared::DomainError;

/// Configurable mock worker for testing.
///
/// By default, dispatch succeeds with "mock response". Call `set_failures`
/// to make specific workers fail.
pub struct MockWorkerPort {
    responses: Mutex<Vec<(WorkerId, String)>>,
    failures: Mutex<Vec<WorkerId>>,
}

impl MockWorkerPort {
    pub fn new() -> Self {
        Self {
            responses: Mutex::new(Vec::new()),
            failures: Mutex::new(Vec::new()),
        }
    }

    /// Makes the given worker fail on dispatch.
    pub fn set_failing(&self, worker_id: WorkerId) {
        self.failures.lock().unwrap().push(worker_id);
    }

    /// Returns all dispatch calls made (worker_id, task_id).
    pub fn dispatch_log(&self) -> Vec<(WorkerId, String)> {
        self.responses.lock().unwrap().clone()
    }
}

impl Default for MockWorkerPort {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerPort for MockWorkerPort {
    fn dispatch(&self, worker_id: &WorkerId, task: &Task) -> Result<String, DomainError> {
        let failures = self.failures.lock().unwrap();
        if failures.iter().any(|w| w == worker_id) {
            return Err(DomainError::WorkerDispatchFailed {
                reason: format!("worker {worker_id} is failing"),
            });
        }

        let mut log = self.responses.lock().unwrap();
        log.push((worker_id.clone(), task.id.to_string()));

        Ok(format!("mock response for task {}", task.id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::job::Message;
    use crate::job::job_id::JobId;

    fn test_task() -> Task {
        Task::new(JobId::new(), "model".into(), Message { role: "user".into(), content: "hi".into() })
    }

    #[test]
    fn dispatch_succeeds_by_default() {
        let port = MockWorkerPort::new();
        let task = test_task();
        let result = port.dispatch(&WorkerId::new("w-0"), &task);
        assert!(result.is_ok());
    }

    #[test]
    fn dispatch_fails_when_configured() {
        let port = MockWorkerPort::new();
        port.set_failing(WorkerId::new("w-0"));
        let task = test_task();
        let result = port.dispatch(&WorkerId::new("w-0"), &task);
        assert!(matches!(result, Err(DomainError::WorkerDispatchFailed { .. })));
    }

    #[test]
    fn dispatch_log_records_calls() {
        let port = MockWorkerPort::new();
        let task = test_task();
        port.dispatch(&WorkerId::new("w-0"), &task).unwrap();
        let log = port.dispatch_log();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].0, WorkerId::new("w-0"));
    }
}
