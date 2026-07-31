use std::fmt;
use std::str::FromStr;

use uuid::Uuid;

/// Typed identifier for [`Job`] aggregates.
///
/// Wraps a UUID v4 to prevent mixing job IDs with other entity IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(Uuid);

impl JobId {
    /// Creates a new random job ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Returns the inner UUID.
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for JobId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Uuid::parse_str(s).map(Self).map_err(|e| format!("invalid JobId: {e}"))
    }
}

impl From<Uuid> for JobId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl serde::Serialize for JobId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for JobId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Uuid::parse_str(&s).map(Self).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_job_id_is_unique() {
        let a = JobId::new();
        let b = JobId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn display_roundtrip() {
        let id = JobId::new();
        let s = id.to_string();
        let parsed: JobId = s.parse().unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn from_str_rejects_invalid() {
        let result: Result<JobId, _> = "not-a-uuid".parse();
        assert!(result.is_err());
    }

    #[test]
    fn as_uuid_returns_inner() {
        let uuid = Uuid::new_v4();
        let id = JobId::from(uuid);
        assert_eq!(id.as_uuid(), uuid);
    }

    #[test]
    fn serialize_is_string() {
        let id = JobId::new();
        let json = serde_json::to_string(&id).unwrap();
        assert!(json.starts_with('"'));
        assert!(json.ends_with('"'));
    }

    #[test]
    fn deserialize_roundtrip() {
        let id = JobId::new();
        let json = serde_json::to_string(&id).unwrap();
        let parsed: JobId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn default_is_unique() {
        let a = JobId::default();
        let b = JobId::default();
        assert_ne!(a, b);
    }
}
