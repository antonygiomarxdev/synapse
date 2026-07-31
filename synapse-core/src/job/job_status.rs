use serde::{Deserialize, Serialize};

use crate::shared::DomainError;

/// Lifecycle state of a [`Job`].
///
/// Valid transitions:
/// - `Pending → Running`
/// - `Pending → Failed`
/// - `Running → Completed`
/// - `Running → Failed`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl JobStatus {
    /// Returns `true` if transitioning to `next` is a valid state move.
    pub fn can_transition_to(&self, next: &JobStatus) -> bool {
        matches!(
            (self, next),
            (Self::Pending, Self::Running)
                | (Self::Pending, Self::Failed)
                | (Self::Running, Self::Completed)
                | (Self::Running, Self::Failed)
        )
    }

    /// Validates and applies a transition. Returns the new status or an error.
    pub fn transition(&self, next: JobStatus) -> Result<JobStatus, DomainError> {
        if self.can_transition_to(&next) {
            Ok(next)
        } else {
            Err(DomainError::InvalidJobTransition {
                from: self.to_string(),
                to: next.to_string(),
            })
        }
    }
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_can_transition_to_running() {
        assert!(JobStatus::Pending.can_transition_to(&JobStatus::Running));
    }

    #[test]
    fn pending_can_transition_to_failed() {
        assert!(JobStatus::Pending.can_transition_to(&JobStatus::Failed));
    }

    #[test]
    fn running_can_transition_to_completed() {
        assert!(JobStatus::Running.can_transition_to(&JobStatus::Completed));
    }

    #[test]
    fn running_can_transition_to_failed() {
        assert!(JobStatus::Running.can_transition_to(&JobStatus::Failed));
    }

    #[test]
    fn completed_cannot_transition() {
        assert!(!JobStatus::Completed.can_transition_to(&JobStatus::Running));
        assert!(!JobStatus::Completed.can_transition_to(&JobStatus::Failed));
        assert!(!JobStatus::Completed.can_transition_to(&JobStatus::Pending));
    }

    #[test]
    fn failed_cannot_transition() {
        assert!(!JobStatus::Failed.can_transition_to(&JobStatus::Running));
        assert!(!JobStatus::Failed.can_transition_to(&JobStatus::Completed));
        assert!(!JobStatus::Failed.can_transition_to(&JobStatus::Pending));
    }

    #[test]
    fn pending_cannot_transition_to_completed() {
        assert!(!JobStatus::Pending.can_transition_to(&JobStatus::Completed));
    }

    #[test]
    fn transition_returns_new_status() {
        let status = JobStatus::Pending.transition(JobStatus::Running).unwrap();
        assert_eq!(status, JobStatus::Running);
    }

    #[test]
    fn transition_rejects_invalid() {
        let result = JobStatus::Completed.transition(JobStatus::Running);
        assert!(matches!(
            result,
            Err(DomainError::InvalidJobTransition { .. })
        ));
    }

    #[test]
    fn display_format() {
        assert_eq!(JobStatus::Pending.to_string(), "pending");
        assert_eq!(JobStatus::Running.to_string(), "running");
        assert_eq!(JobStatus::Completed.to_string(), "completed");
        assert_eq!(JobStatus::Failed.to_string(), "failed");
    }

    #[test]
    fn serde_roundtrip() {
        let statuses = [
            JobStatus::Pending,
            JobStatus::Running,
            JobStatus::Completed,
            JobStatus::Failed,
        ];
        for status in &statuses {
            let json = serde_json::to_string(status).unwrap();
            let parsed: JobStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*status, parsed);
        }
    }

    #[test]
    fn serde_uses_snake_case() {
        assert_eq!(serde_json::to_string(&JobStatus::Pending).unwrap(), "\"pending\"");
        assert_eq!(serde_json::to_string(&JobStatus::Failed).unwrap(), "\"failed\"");
    }
}
