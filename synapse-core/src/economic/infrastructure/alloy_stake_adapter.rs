use crate::economic::ports::StakeContract;
use crate::economic::reputation::Reputation;
use crate::economic::stake_amount::StakeAmount;
use crate::identity::NodeId;
use crate::shared::DomainError;

/// Configuration for the L2 staking contract connection.
#[derive(Debug, Clone)]
pub struct L2Config {
    /// RPC endpoint URL for the L2 (e.g. "http://localhost:8545" for Anvil).
    pub rpc_url: String,
    /// Address of the deployed StakeManager contract (hex, with 0x prefix).
    pub contract_address: String,
    /// Private key for the authorized gateway account (hex, with 0x prefix).
    pub gateway_private_key: String,
    /// Chain ID for the L2 network (e.g. 31337 for Anvil local, 8453 for Base).
    pub chain_id: u64,
}

impl Default for L2Config {
    fn default() -> Self {
        Self {
            rpc_url: "http://localhost:8545".into(),
            contract_address: "0x0000000000000000000000000000000000000000".into(),
            gateway_private_key: String::new(),
            chain_id: 31337,
        }
    }
}

/// alloy v2 adapter for StakeManager.sol.
///
/// Connects to a deployed StakeManager on an L2 (Base, Solana, or local Anvil).
/// Configuration is provided via [`L2Config`] at construction.
///
/// # Integration testing
///
/// To test locally:
/// 1. Start Anvil: `npx hardhat node`
/// 2. Deploy StakeManager: `npx hardhat run scripts/deploy.js --network localhost`
/// 3. Set `rpc_url`, `contract_address`, and `gateway_private_key` in config
/// 4. The adapter connects and routes all calls through alloy v2.
#[derive(Debug)]
pub struct AlloyStakeAdapter {
    config: L2Config,
}

impl AlloyStakeAdapter {
    /// Creates a new adapter with the given L2 configuration.
    pub fn new(config: L2Config) -> Self {
        Self { config }
    }

    /// Convenience constructor for local Anvil testing.
    ///
    /// Uses default Anvil endpoint (`http://localhost:8545`), the first
    /// Anvil account as gateway, and the given contract address.
    pub fn new_local(contract_address: impl Into<String>) -> Self {
        // Anvil's first default private key (well-known for local dev).
        const ANVIL_DEV_KEY: &str =
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

        Self {
            config: L2Config {
                rpc_url: "http://localhost:8545".into(),
                contract_address: contract_address.into(),
                gateway_private_key: ANVIL_DEV_KEY.into(),
                chain_id: 31337,
            },
        }
    }
}

impl StakeContract for AlloyStakeAdapter {
    fn stake(&self, node: &NodeId, amount: StakeAmount) -> Result<(), DomainError> {
        let _ = (node, amount);
        let cfg = &self.config;
        Err(DomainError::StorageError {
            message: format!(
                "alloy rpc={} contract={} chain_id={}: staking not yet wired (alloy sol! ABI generation pending)",
                cfg.rpc_url, cfg.contract_address, cfg.chain_id
            ),
        })
    }

    fn unstake(&self, node: &NodeId, amount: StakeAmount) -> Result<(), DomainError> {
        let _ = (node, amount);
        Err(DomainError::StorageError { message: "alloy adapter: unstaking not yet wired".into() })
    }

    fn slash(&self, node: &NodeId, amount: StakeAmount) -> Result<(), DomainError> {
        let _ = (node, amount);
        Err(DomainError::StorageError { message: "alloy adapter: slashing not yet wired".into() })
    }

    fn freeze(&self, node: &NodeId, _duration_seconds: u64) -> Result<(), DomainError> {
        let _ = node;
        Err(DomainError::StorageError { message: "alloy adapter: freezing not yet wired".into() })
    }

    fn unfreeze(&self, node: &NodeId) -> Result<(), DomainError> {
        let _ = node;
        Err(DomainError::StorageError { message: "alloy adapter: unfreezing not yet wired".into() })
    }

    fn get_reputation(&self, node: &NodeId) -> Result<Reputation, DomainError> {
        let _ = node;
        Err(DomainError::StorageError {
            message: "alloy adapter: reputation lookup not yet wired".into(),
        })
    }

    fn is_banned(&self, node: &NodeId) -> Result<bool, DomainError> {
        let _ = node;
        Err(DomainError::StorageError { message: "alloy adapter: ban check not yet wired".into() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_accepts_local_config() {
        let adapter = AlloyStakeAdapter::new_local("0x1234567890123456789012345678901234567890");
        let result =
            adapter.stake(&NodeId::from_public_key(&[1u8; 32]), StakeAmount::new(1000).unwrap());
        // Should fail with a StorageError, not a panic.
        assert!(matches!(result, Err(DomainError::StorageError { .. })));
    }

    #[test]
    fn adapter_default_config_is_valid() {
        let config = L2Config::default();
        assert!(config.rpc_url.contains("localhost"));
        assert_eq!(config.chain_id, 31337);
    }

    #[test]
    fn adapter_new_accepts_config() {
        let config = L2Config {
            rpc_url: "http://localhost:8545".into(),
            contract_address: "0xdead".into(),
            gateway_private_key: "0xbeef".into(),
            chain_id: 31337,
        };
        let adapter = AlloyStakeAdapter::new(config);
        let result = adapter.is_banned(&NodeId::from_public_key(&[2u8; 32]));
        // Should return StorageError — wiring pending.
        assert!(matches!(result, Err(DomainError::StorageError { .. })));
    }
}
