use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::job_id::JobId;
use super::job_status::JobStatus;
use crate::shared::DomainError;

/// An OpenAI-compatible message in a job request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Result produced when a job completes successfully.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobResult {
    pub text: String,
    pub tokens: u32,
}

/// Priority level for job scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Low,
    Normal,
    High,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Normal => write!(f, "normal"),
            Self::High => write!(f, "high"),
        }
    }
}

impl std::str::FromStr for Priority {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(Self::Low),
            "normal" => Ok(Self::Normal),
            "high" => Ok(Self::High),
            _ => Err(format!("invalid priority: {s}")),
        }
    }
}

/// The Job aggregate root.
///
/// Identity fields (`model`, `messages`, `priority`) are immutable after creation.
/// Only `status`, `result`, and `error` can change through validated transitions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Job {
    pub id: JobId,
    pub model: String,
    pub messages: Vec<Message>,
    pub priority: Priority,
    pub status: JobStatus,
    pub result: Option<JobResult>,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Job {
    /// Creates a new job in `Pending` status.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidJob`] if model is empty or messages is empty.
    pub fn submit(model: String, messages: Vec<Message>, priority: Priority) -> Result<Self, DomainError> {
        if model.is_empty() {
            return Err(DomainError::InvalidJob { reason: "model must not be empty".into() });
        }
        if messages.is_empty() {
            return Err(DomainError::InvalidJob { reason: "messages must not be empty".into() });
        }
        for (i, msg) in messages.iter().enumerate() {
            if msg.role.is_empty() {
                return Err(DomainError::InvalidJob {
                    reason: format!("messages[{i}].role must not be empty"),
                });
            }
            if msg.content.is_empty() {
                return Err(DomainError::InvalidJob {
                    reason: format!("messages[{i}].content must not be empty"),
                });
            }
        }

        let now = Utc::now();
        Ok(Self {
            id: JobId::new(),
            model,
            messages,
            priority,
            status: JobStatus::Pending,
            result: None,
            error: None,
            created_at: now,
            updated_at: now,
        })
    }

    /// Transitions the job to a new status.
    pub fn transition_to(&mut self, next: JobStatus) -> Result<(), DomainError> {
        self.status = self.status.transition(next)?;
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Marks the job as completed with a result.
    pub fn complete(&mut self, result: JobResult) -> Result<(), DomainError> {
        self.transition_to(JobStatus::Completed)?;
        self.result = Some(result);
        Ok(())
    }

    /// Marks the job as failed with a reason.
    pub fn fail(&mut self, reason: String) -> Result<(), DomainError> {
        self.transition_to(JobStatus::Failed)?;
        self.error = Some(reason);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_messages() -> Vec<Message> {
        vec![Message { role: "user".into(), content: "Hello".into() }]
    }

    #[test]
    fn submit_creates_pending_job() {
        let job = Job::submit("model".into(), valid_messages(), Priority::Normal).unwrap();
        assert_eq!(job.status, JobStatus::Pending);
        assert!(job.result.is_none());
        assert!(job.error.is_none());
    }

    #[test]
    fn submit_rejects_empty_model() {
        let result = Job::submit("".into(), valid_messages(), Priority::Normal);
        assert!(matches!(result, Err(DomainError::InvalidJob { .. })));
    }

    #[test]
    fn submit_rejects_empty_messages() {
        let result = Job::submit("model".into(), vec![], Priority::Normal);
        assert!(matches!(result, Err(DomainError::InvalidJob { .. })));
    }

    #[test]
    fn submit_rejects_empty_role() {
        let msgs = vec![Message { role: "".into(), content: "hi".into() }];
        let result = Job::submit("model".into(), msgs, Priority::Normal);
        assert!(matches!(result, Err(DomainError::InvalidJob { .. })));
    }

    #[test]
    fn submit_rejects_empty_content() {
        let msgs = vec![Message { role: "user".into(), content: "".into() }];
        let result = Job::submit("model".into(), msgs, Priority::Normal);
        assert!(matches!(result, Err(DomainError::InvalidJob { .. })));
    }

    #[test]
    fn submit_sets_default_priority() {
        let job = Job::submit("model".into(), valid_messages(), Priority::default()).unwrap();
        assert_eq!(job.priority, Priority::Normal);
    }

    #[test]
    fn transition_pending_to_running() {
        let mut job = Job::submit("model".into(), valid_messages(), Priority::Normal).unwrap();
        job.transition_to(JobStatus::Running).unwrap();
        assert_eq!(job.status, JobStatus::Running);
    }

    #[test]
    fn transition_running_to_completed() {
        let mut job = Job::submit("model".into(), valid_messages(), Priority::Normal).unwrap();
        job.transition_to(JobStatus::Running).unwrap();
        job.transition_to(JobStatus::Completed).unwrap();
        assert_eq!(job.status, JobStatus::Completed);
    }

    #[test]
    fn transition_rejects_invalid() {
        let mut job = Job::submit("model".into(), valid_messages(), Priority::Normal).unwrap();
        let result = job.transition_to(JobStatus::Completed);
        assert!(matches!(result, Err(DomainError::InvalidJobTransition { .. })));
        assert_eq!(job.status, JobStatus::Pending);
    }

    #[test]
    fn complete_sets_result() {
        let mut job = Job::submit("model".into(), valid_messages(), Priority::Normal).unwrap();
        job.transition_to(JobStatus::Running).unwrap();
        job.complete(JobResult { text: "hi".into(), tokens: 1 }).unwrap();
        assert_eq!(job.status, JobStatus::Completed);
        assert!(job.result.is_some());
        assert_eq!(job.result.as_ref().unwrap().text, "hi");
    }

    #[test]
    fn fail_sets_error() {
        let mut job = Job::submit("model".into(), valid_messages(), Priority::Normal).unwrap();
        job.fail("timeout".into()).unwrap();
        assert_eq!(job.status, JobStatus::Failed);
        assert_eq!(job.error.as_deref(), Some("timeout"));
    }

    #[test]
    fn fail_from_running() {
        let mut job = Job::submit("model".into(), valid_messages(), Priority::Normal).unwrap();
        job.transition_to(JobStatus::Running).unwrap();
        job.fail("crash".into()).unwrap();
        assert_eq!(job.status, JobStatus::Failed);
    }

    #[test]
    fn cannot_fail_twice() {
        let mut job = Job::submit("model".into(), valid_messages(), Priority::Normal).unwrap();
        job.fail("first".into()).unwrap();
        let result = job.fail("second".into());
        assert!(matches!(result, Err(DomainError::InvalidJobTransition { .. })));
    }

    #[test]
    fn priority_display() {
        assert_eq!(Priority::Low.to_string(), "low");
        assert_eq!(Priority::Normal.to_string(), "normal");
        assert_eq!(Priority::High.to_string(), "high");
    }

    #[test]
    fn priority_from_str() {
        assert_eq!("low".parse::<Priority>().unwrap(), Priority::Low);
        assert_eq!("normal".parse::<Priority>().unwrap(), Priority::Normal);
        assert_eq!("high".parse::<Priority>().unwrap(), Priority::High);
        assert_eq!("HIGH".parse::<Priority>().unwrap(), Priority::High);
        assert!("urgent".parse::<Priority>().is_err());
    }

    #[test]
    fn job_serde_roundtrip() {
        let mut job = Job::submit("model".into(), valid_messages(), Priority::High).unwrap();
        job.transition_to(JobStatus::Running).unwrap();
        job.complete(JobResult { text: "done".into(), tokens: 5 }).unwrap();

        let json = serde_json::to_string(&job).unwrap();
        let parsed: Job = serde_json::from_str(&json).unwrap();
        assert_eq!(job, parsed);
    }
}
