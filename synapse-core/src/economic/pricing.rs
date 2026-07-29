use crate::model::ExpertId;
#[cfg(test)]
use crate::model::ModelId;
use crate::shared::DomainError;
use serde::{Deserialize, Serialize};

/// Price denominated in smallest currency units per 1M tokens.
///
/// For USDC this is cents (2 decimal places). Must be non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TokensPerMillion(u64);

impl TokensPerMillion {
    /// Creates a new price value.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidPrice`] if `tokens_per_million` is zero.
    pub fn new(tokens_per_million: u64) -> Result<Self, DomainError> {
        if tokens_per_million == 0 {
            return Err(DomainError::InvalidPrice { reason: "must be non-zero".into() });
        }
        Ok(Self(tokens_per_million))
    }

    /// The raw price per million tokens.
    pub fn tokens_per_million(self) -> u64 {
        self.0
    }
}

/// Total cost of an expert route, summing all expert prices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteCost {
    prices: Vec<TokensPerMillion>,
}

impl RouteCost {
    /// Creates a [`RouteCost`] from individual expert prices.
    pub fn new(prices: Vec<TokensPerMillion>) -> Self {
        Self { prices }
    }

    /// The total cost across all experts in the route.
    ///
    /// Constructs [`TokensPerMillion`] directly from the summed internal value.
    /// This is safe because we are within the same module and `TokensPerMillion`'s
    /// tuple field is a private same-crate detail — callers must use
    /// [`TokensPerMillion::new`] for validation.
    pub fn total(&self) -> TokensPerMillion {
        let sum: u64 = self.prices.iter().map(|p| p.0).sum();
        TokensPerMillion(sum)
    }
}

/// Selects the cheapest valid expert route.
///
/// Each expert may offer multiple price points (for different replicas).
/// This function picks the minimum price for each expert and returns
/// experts in input order with the total cost.
///
/// Returns `None` for empty input. Returns `Some` with zero total cost
/// if all experts have empty price lists (though in practice this should
/// not happen — the DHT filters experts with no published price).
pub fn cheapest_route(
    experts: &[(ExpertId, Vec<TokensPerMillion>)],
) -> Option<(Vec<ExpertId>, TokensPerMillion)> {
    if experts.is_empty() {
        return None;
    }

    let mut route_experts = Vec::with_capacity(experts.len());
    let mut total_cost: u64 = 0;

    for (expert_id, prices) in experts {
        route_experts.push(expert_id.clone());
        if let Some(min_price) = prices.iter().min_by_key(|p| p.0) {
            total_cost += min_price.0;
        }
    }

    Some((route_experts, TokensPerMillion(total_cost)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_per_million_rejects_zero() {
        let err = TokensPerMillion::new(0).unwrap_err();
        assert_eq!(err.to_string(), "invalid price: must be non-zero");
    }

    #[test]
    fn tokens_per_million_accepts_one() {
        let price = TokensPerMillion::new(1).unwrap();
        assert_eq!(price.tokens_per_million(), 1);
    }

    #[test]
    fn tokens_per_million_comparison() {
        let cheap = TokensPerMillion::new(100).unwrap();
        let expensive = TokensPerMillion::new(200).unwrap();
        assert!(cheap < expensive);
    }

    #[test]
    fn route_cost_sums_prices() {
        let a = TokensPerMillion::new(10).unwrap();
        let b = TokensPerMillion::new(20).unwrap();
        let c = TokensPerMillion::new(30).unwrap();
        let cost = RouteCost::new(vec![a, b, c]);
        assert_eq!(cost.total().tokens_per_million(), 60);
    }

    #[test]
    fn route_cost_empty_returns_zero() {
        let cost = RouteCost::new(vec![]);
        assert_eq!(cost.total().tokens_per_million(), 0);
    }

    #[test]
    fn cheapest_route_picks_lowest_sum() {
        let mixtral = ModelId::new("mixtral-8x7b").unwrap();
        let e0 = ExpertId::new_unchecked(mixtral.clone(), 0);
        let e1 = ExpertId::new_unchecked(mixtral.clone(), 1);
        let e2 = ExpertId::new_unchecked(mixtral.clone(), 2);

        let p5 = TokensPerMillion::new(5).unwrap();
        let p10 = TokensPerMillion::new(10).unwrap();
        let p100 = TokensPerMillion::new(100).unwrap();

        let experts = &[
            (e0.clone(), vec![p5, p100]),
            (e1.clone(), vec![p10, p5]),
            (e2.clone(), vec![p100, p10]),
        ];

        let (route, cost) = cheapest_route(experts).unwrap();
        assert_eq!(cost.tokens_per_million(), 20);
        assert_eq!(route.len(), 3);
    }

    #[test]
    fn cheapest_route_empty_input_returns_none() {
        let result = cheapest_route(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn cheapest_route_single_expert() {
        let mixtral = ModelId::new("mixtral-8x7b").unwrap();
        let e0 = ExpertId::new_unchecked(mixtral, 0);
        let p10 = TokensPerMillion::new(10).unwrap();
        let (route, cost) = cheapest_route(&[(e0, vec![p10])]).unwrap();
        assert_eq!(cost.tokens_per_million(), 10);
        assert_eq!(route.len(), 1);
    }
}
