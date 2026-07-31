use std::fmt;
use std::str::FromStr;

use uuid::Uuid;

/// Typed identifier for [`Task`] instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(Uuid);

impl TaskId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for TaskId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Uuid::parse_str(s).map(Self).map_err(|e| format!("invalid TaskId: {e}"))
    }
}

impl From<Uuid> for TaskId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl serde::Serialize for TaskId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for TaskId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Uuid::parse_str(&s).map(Self).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_unique() {
        assert_ne!(TaskId::new(), TaskId::new());
    }

    #[test]
    fn display_roundtrip() {
        let id = TaskId::new();
        let parsed: TaskId = id.to_string().parse().unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn from_str_rejects_invalid() {
        assert!("bad".parse::<TaskId>().is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let id = TaskId::new();
        let json = serde_json::to_string(&id).unwrap();
        let parsed: TaskId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, parsed);
    }
}
