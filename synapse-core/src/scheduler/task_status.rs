use serde::{Deserialize, Serialize};

use crate::shared::DomainError;

/// Lifecycle state of a [`Task`].
///
/// Valid transitions:
/// - `Pending → Leased` (dispatched to worker)
/// - `Leased → Completed` (worker returned result)
/// - `Leased → Failed` (worker error or timeout)
/// - `Failed → Pending` (retry — re-enqueue for dispatch)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Leased,
    Completed,
    Failed,
}

impl TaskStatus {
    pub fn can_transition_to(&self, next: &TaskStatus) -> bool {
        matches!(
            (self, next),
            (Self::Pending, Self::Leased)
                | (Self::Leased, Self::Completed)
                | (Self::Leased, Self::Failed)
                | (Self::Failed, Self::Pending)
        )
    }

    pub fn transition(&self, next: TaskStatus) -> Result<TaskStatus, DomainError> {
        if self.can_transition_to(&next) {
            Ok(next)
        } else {
            Err(DomainError::InvalidTaskTransition {
                from: self.to_string(),
                to: next.to_string(),
            })
        }
    }
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Leased => write!(f, "leased"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_to_leased() {
        assert!(TaskStatus::Pending.can_transition_to(&TaskStatus::Leased));
    }

    #[test]
    fn leased_to_completed() {
        assert!(TaskStatus::Leased.can_transition_to(&TaskStatus::Completed));
    }

    #[test]
    fn leased_to_failed() {
        assert!(TaskStatus::Leased.can_transition_to(&TaskStatus::Failed));
    }

    #[test]
    fn failed_to_pending_retry() {
        assert!(TaskStatus::Failed.can_transition_to(&TaskStatus::Pending));
    }

    #[test]
    fn pending_cannot_go_to_completed() {
        assert!(!TaskStatus::Pending.can_transition_to(&TaskStatus::Completed));
    }

    #[test]
    fn completed_is_terminal() {
        assert!(!TaskStatus::Completed.can_transition_to(&TaskStatus::Pending));
        assert!(!TaskStatus::Completed.can_transition_to(&TaskStatus::Leased));
    }

    #[test]
    fn transition_returns_new_status() {
        let s = TaskStatus::Pending.transition(TaskStatus::Leased).unwrap();
        assert_eq!(s, TaskStatus::Leased);
    }

    #[test]
    fn transition_rejects_invalid() {
        let r = TaskStatus::Pending.transition(TaskStatus::Completed);
        assert!(matches!(r, Err(DomainError::InvalidTaskTransition { .. })));
    }

    #[test]
    fn display_format() {
        assert_eq!(TaskStatus::Pending.to_string(), "pending");
        assert_eq!(TaskStatus::Leased.to_string(), "leased");
    }

    #[test]
    fn serde_roundtrip() {
        let statuses = [TaskStatus::Pending, TaskStatus::Leased, TaskStatus::Completed, TaskStatus::Failed];
        for s in &statuses {
            let json = serde_json::to_string(s).unwrap();
            let parsed: TaskStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*s, parsed);
        }
    }
}
