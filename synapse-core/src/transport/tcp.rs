/// TCP transport for multi-machine worker communication.
///
/// Replaces Unix sockets with TCP for cross-machine deployment.
/// Workers listen on TCP ports, coordinator connects via TCP.
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::shared::DomainError;

/// Configuration for a TCP worker endpoint.
#[derive(Debug, Clone)]
pub struct TcpWorkerConfig {
    /// Worker ID.
    pub worker_id: String,
    /// Address to connect to (or listen on).
    pub addr: SocketAddr,
}

/// TCP-based worker port for multi-machine deployment.
///
/// Connects to workers over TCP instead of Unix sockets.
pub struct TcpTransport {
    config: TcpWorkerConfig,
}

impl TcpTransport {
    /// Create a new TCP transport for the given worker.
    pub fn new(config: TcpWorkerConfig) -> Self {
        Self { config }
    }

    /// Send a message to the worker and receive a response.
    pub async fn send(&self, message: &[u8]) -> Result<Vec<u8>, DomainError> {
        let mut stream = TcpStream::connect(self.config.addr)
            .await
            .map_err(|e| DomainError::WorkerDispatchFailed {
                reason: format!("TCP connect failed: {e}"),
            })?;

        // Write length-prefixed message
        let len = message.len() as u32;
        stream.write_all(&len.to_be_bytes()).await.map_err(|e| {
            DomainError::WorkerDispatchFailed {
                reason: format!("TCP write length failed: {e}"),
            }
        })?;
        stream.write_all(message).await.map_err(|e| {
            DomainError::WorkerDispatchFailed {
                reason: format!("TCP write message failed: {e}"),
            }
        })?;

        // Read length-prefixed response
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await.map_err(|e| {
            DomainError::WorkerDispatchFailed {
                reason: format!("TCP read length failed: {e}"),
            }
        })?;
        let resp_len = u32::from_be_bytes(len_buf) as usize;

        let mut resp_buf = vec![0u8; resp_len];
        stream.read_exact(&mut resp_buf).await.map_err(|e| {
            DomainError::WorkerDispatchFailed {
                reason: format!("TCP read response failed: {e}"),
            }
        })?;

        Ok(resp_buf)
    }

    /// Get the worker address.
    pub fn addr(&self) -> SocketAddr {
        self.config.addr
    }
}

/// TCP listener for worker-side.
///
/// Listens for incoming connections and handles requests.
pub struct TcpWorkerListener {
    listener: TcpListener,
}

impl TcpWorkerListener {
    /// Bind to the given address and start listening.
    pub async fn bind(addr: SocketAddr) -> Result<Self, DomainError> {
        let listener = TcpListener::bind(addr).await.map_err(|e| {
            DomainError::WorkerDispatchFailed {
                reason: format!("TCP bind failed: {e}"),
            }
        })?;
        Ok(Self { listener })
    }

    /// Get the bound address (useful when binding to port 0).
    pub fn local_addr(&self) -> Result<SocketAddr, DomainError> {
        self.listener.local_addr().map_err(|e| {
            DomainError::WorkerDispatchFailed {
                reason: format!("TCP local_addr failed: {e}"),
            }
        })
    }

    /// Accept a single connection, read message, call handler, send response.
    pub async fn accept_one<F>(&self, handler: F) -> Result<(), DomainError>
    where
        F: Fn(Vec<u8>) -> Vec<u8>,
    {
        let (mut stream, _addr) = self.listener.accept().await.map_err(|e| {
            DomainError::WorkerDispatchFailed {
                reason: format!("TCP accept failed: {e}"),
            }
        })?;

        // Read length-prefixed message
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await.map_err(|e| {
            DomainError::WorkerDispatchFailed {
                reason: format!("TCP read length failed: {e}"),
            }
        })?;
        let msg_len = u32::from_be_bytes(len_buf) as usize;

        let mut msg_buf = vec![0u8; msg_len];
        stream.read_exact(&mut msg_buf).await.map_err(|e| {
            DomainError::WorkerDispatchFailed {
                reason: format!("TCP read message failed: {e}"),
            }
        })?;

        // Call handler
        let response = handler(msg_buf);

        // Write length-prefixed response
        let len = response.len() as u32;
        stream.write_all(&len.to_be_bytes()).await.map_err(|e| {
            DomainError::WorkerDispatchFailed {
                reason: format!("TCP write length failed: {e}"),
            }
        })?;
        stream.write_all(&response).await.map_err(|e| {
            DomainError::WorkerDispatchFailed {
                reason: format!("TCP write response failed: {e}"),
            }
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn test_addr() -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 0)
    }

    #[tokio::test]
    async fn tcp_transport_connect_fails_when_no_listener() {
        let config = TcpWorkerConfig {
            worker_id: "test".into(),
            addr: "127.0.0.1:1".parse().unwrap(), // Port 1 - no listener
        };
        let transport = TcpTransport::new(config);
        let result = transport.send(b"hello").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn tcp_transport_send_and_receive() {
        // Start listener
        let listener = TcpWorkerListener::bind(test_addr()).await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Spawn handler
        let handle = tokio::spawn(async move {
            listener.accept_one(|msg| {
                // Echo back with prefix
                let mut response = b"echo: ".to_vec();
                response.extend_from_slice(&msg);
                response
            }).await.unwrap();
        });

        // Give listener time to start
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Connect and send
        let config = TcpWorkerConfig {
            worker_id: "test".into(),
            addr,
        };
        let transport = TcpTransport::new(config);
        let response = transport.send(b"hello").await.unwrap();

        assert_eq!(response, b"echo: hello");

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn tcp_transport_multiple_messages() {
        // Start listener
        let listener = TcpWorkerListener::bind(test_addr()).await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Spawn handler that accepts multiple connections
        let handle = tokio::spawn(async move {
            for _ in 0..3 {
                listener.accept_one(|msg| {
                    let mut response = b"response: ".to_vec();
                    response.extend_from_slice(&msg);
                    response
                }).await.unwrap();
            }
        });

        // Give listener time to start
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Send 3 messages
        for i in 0..3 {
            let config = TcpWorkerConfig {
                worker_id: "test".into(),
                addr,
            };
            let transport = TcpTransport::new(config);
            let msg = format!("message {i}");
            let response = transport.send(msg.as_bytes()).await.unwrap();
            assert_eq!(response, format!("response: message {i}").as_bytes());
        }

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn tcp_worker_config_clone() {
        let config = TcpWorkerConfig {
            worker_id: "test".into(),
            addr: test_addr(),
        };
        let cloned = config.clone();
        assert_eq!(cloned.worker_id, "test");
        assert_eq!(cloned.addr, test_addr());
    }

    #[tokio::test]
    async fn tcp_transport_returns_addr() {
        let config = TcpWorkerConfig {
            worker_id: "test".into(),
            addr: test_addr(),
        };
        let transport = TcpTransport::new(config);
        assert_eq!(transport.addr(), test_addr());
    }
}
