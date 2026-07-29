use crate::identity::NodeId;
use crate::shared::DomainError;

pub use crate::economic::pricing::TokensPerMillion;
pub use crate::economic::reputation::Reputation;
pub use crate::economic::stake::StakeAmount;

/// Port implemented by the L2 staking contract adapter.
pub trait StakeContract {
    fn stake(&self, node: &NodeId, amount: StakeAmount) -> Result<(), DomainError>;
    fn unstake(&self, node: &NodeId, amount: StakeAmount) -> Result<(), DomainError>;
    fn slash(&self, node: &NodeId, amount: StakeAmount) -> Result<(), DomainError>;
    fn freeze(&self, node: &NodeId, duration_seconds: u64) -> Result<(), DomainError>;
    fn unfreeze(&self, node: &NodeId) -> Result<(), DomainError>;
    fn get_reputation(&self, node: &NodeId) -> Result<Reputation, DomainError>;
    fn is_banned(&self, node: &NodeId) -> Result<bool, DomainError>;
}

/// Port implemented by the payment processing adapter.
pub trait PaymentGateway {
    fn pay(&self, node: &NodeId, price_per_million: TokensPerMillion, token_count: u64) -> Result<String, DomainError>;
    fn verify_payment(&self, tx_hash: &str) -> Result<bool, DomainError>;
}

#[cfg(test)]
mod tests {
    // Traits only — no tests yet.
}
