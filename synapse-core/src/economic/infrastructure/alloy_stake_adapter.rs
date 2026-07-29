use crate::economic::ports::StakeContract;
use crate::economic::reputation::Reputation;
use crate::economic::stake_amount::StakeAmount;
use crate::identity::NodeId;
use crate::shared::DomainError;

/// alloy v2 adapter for StakeManager.sol — STUB.
///
/// Actual alloy integration deferred until L2 selection (Solana vs Base).
/// Issue: https://github.com/antonygiomarxdev/synapse/issues/3
#[derive(Debug)]
pub struct AlloyStakeAdapter;

impl AlloyStakeAdapter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AlloyStakeAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl StakeContract for AlloyStakeAdapter {
    fn stake(&self, _node: &NodeId, _amount: StakeAmount) -> Result<(), DomainError> {
        Err(DomainError::StorageError {
            message: "L2 adapter: staking not yet implemented (awaiting L2 selection)".into(),
        })
    }
    fn unstake(&self, _node: &NodeId, _amount: StakeAmount) -> Result<(), DomainError> {
        Err(DomainError::StorageError {
            message: "L2 adapter: unstaking not yet implemented (awaiting L2 selection)".into(),
        })
    }
    fn slash(&self, _node: &NodeId, _amount: StakeAmount) -> Result<(), DomainError> {
        Err(DomainError::StorageError {
            message: "L2 adapter: slashing not yet implemented (awaiting L2 selection)".into(),
        })
    }
    fn freeze(&self, _node: &NodeId, _duration_seconds: u64) -> Result<(), DomainError> {
        Err(DomainError::StorageError {
            message: "L2 adapter: freezing not yet implemented (awaiting L2 selection)".into(),
        })
    }
    fn unfreeze(&self, _node: &NodeId) -> Result<(), DomainError> {
        Err(DomainError::StorageError {
            message: "L2 adapter: unfreezing not yet implemented (awaiting L2 selection)".into(),
        })
    }
    fn get_reputation(&self, _node: &NodeId) -> Result<Reputation, DomainError> {
        Err(DomainError::StorageError {
            message: "L2 adapter: reputation lookup not yet implemented (awaiting L2 selection)"
                .into(),
        })
    }
    fn is_banned(&self, _node: &NodeId) -> Result<bool, DomainError> {
        Err(DomainError::StorageError {
            message: "L2 adapter: ban check not yet implemented (awaiting L2 selection)".into(),
        })
    }
}
