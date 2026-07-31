/// Transport port trait for worker communication.
///
/// Defines the interface for sending messages to workers.
/// Infrastructure adapters (TCP, Unix socket) implement this trait.
use crate::shared::DomainError;

/// Port for sending messages to workers and receiving responses.
#[async_trait::async_trait]
pub trait TransportPort: Send + Sync {
    /// Send a message to the worker and receive a response.
    async fn send(&self, message: &[u8]) -> Result<Vec<u8>, DomainError>;
}
