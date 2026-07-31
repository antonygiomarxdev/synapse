use std::fmt;

/// Typed identifier for inference workers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkerId(String);

impl WorkerId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl serde::Serialize for WorkerId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> serde::Deserialize<'de> for WorkerId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        assert_eq!(WorkerId::new("w-0").to_string(), "w-0");
    }

    #[test]
    fn equality() {
        assert_eq!(WorkerId::new("a"), WorkerId::new("a"));
        assert_ne!(WorkerId::new("a"), WorkerId::new("b"));
    }

    #[test]
    fn serde_roundtrip() {
        let id = WorkerId::new("worker-1");
        let json = serde_json::to_string(&id).unwrap();
        let parsed: WorkerId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, parsed);
    }
}
