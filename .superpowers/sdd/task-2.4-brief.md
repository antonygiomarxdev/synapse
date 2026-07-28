### Task 2.4: DAG Swarm Domain

**Files:**
- Create: `synapse-core/src/swarm/dag.rs`
- Modify: `synapse-core/src/shared/domain_error.rs` — add `InvalidRoute`
- Modify: `synapse-core/src/swarm/mod.rs` — re-export `DagRoute`

**Interfaces:**
- Consumes: `ExpertId`, `ModelId`
- Produces: `DagRoute { model: ModelId, steps: Vec<ExpertId> }`
- Produces: `DagRoute::new(model: ModelId, steps: Vec<ExpertId>) -> Result<Self, DomainError>`
- Produces: `DagRoute::len(&self) -> usize`, `DagRoute::is_empty(&self) -> bool`
- Produces: `DagRoute::expert_dependency_graph(steps: &[ExpertId]) -> Result<HashMap<ExpertId, Vec<ExpertId>>, DomainError>`

- [ ] **Step 1: Add InvalidRoute error variant**

Modify `synapse-core/src/shared/domain_error.rs`:

```rust
#[error("invalid route: {reason}")]
InvalidRoute { reason: String },
```

Add test:

```rust
#[test]
fn invalid_route_display() {
    let err = DomainError::InvalidRoute { reason: "empty steps".into() };
    assert_eq!(err.to_string(), "invalid route: empty steps");
}
```

Run: `cargo test shared::domain_error -p synapse-core`
Expected: PASS.

- [ ] **Step 2: Write failing DagRoute tests**

Create `synapse-core/src/swarm/dag.rs` with this test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ExpertId, ModelId};

    fn model() -> ModelId {
        ModelId::new("mixtral-8x7b").unwrap()
    }

    fn expert(index: u32) -> ExpertId {
        ExpertId::new(model(), index, 8).unwrap()
    }

    #[test]
    fn route_rejects_empty_steps() {
        let result = DagRoute::new(model(), vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn route_rejects_mixed_models() {
        let kimi = ModelId::new("kimi-k3").unwrap();
        let steps = vec![expert(0), ExpertId::new(kimi, 1, 896).unwrap()];
        let result = DagRoute::new(model(), steps);
        assert!(result.is_err());
    }

    #[test]
    fn valid_route_has_steps() {
        let route = DagRoute::new(model(), vec![expert(0), expert(3), expert(7)]).unwrap();
        assert_eq!(route.len(), 3);
        assert_eq!(route.model().as_str(), "mixtral-8x7b");
    }

    #[test]
    fn dependency_graph_links_consecutive_experts() {
        let route = DagRoute::new(model(), vec![expert(0), expert(3), expert(7)]).unwrap();
        let graph = route.dependency_graph();
        assert_eq!(graph.get(&expert(0)), Some(&vec![expert(3)]));
        assert_eq!(graph.get(&expert(3)), Some(&vec![expert(7)]));
        assert_eq!(graph.get(&expert(7)), Some(&vec![]));
    }
}
```

Run: `cargo test swarm::dag::tests::route_rejects_empty_steps -p synapse-core`
Expected: FAIL with `DagRoute` not found.

- [ ] **Step 3: Implement DagRoute**

Add the implementation to `synapse-core/src/swarm/dag.rs`:

```rust
use crate::model::{ExpertId, ModelId};
use crate::shared::DomainError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A directed path through the expert graph for a single request.
///
/// Each step activates one expert. The path is ordered: step N feeds
/// hidden states into step N+1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DagRoute {
    model: ModelId,
    steps: Vec<ExpertId>,
}

impl DagRoute {
    /// Creates a DAG route.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::InvalidRoute`] if the route is empty or
    /// if steps belong to different models.
    pub fn new(model: ModelId, steps: Vec<ExpertId>) -> Result<Self, DomainError> {
        if steps.is_empty() {
            return Err(DomainError::InvalidRoute {
                reason: "route must contain at least one expert step".into(),
            });
        }
        if steps.iter().any(|e| e.model != model) {
            return Err(DomainError::InvalidRoute {
                reason: "all route steps must belong to the same model".into(),
            });
        }
        Ok(Self { model, steps })
    }

    /// The model this route executes.
    pub fn model(&self) -> &ModelId {
        &self.model
    }

    /// Ordered expert steps.
    pub fn steps(&self) -> &[ExpertId] {
        &self.steps
    }

    /// Number of expert activations.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// True if the route has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Builds a simple dependency graph where each expert depends on the
    /// next expert in the route. The final expert has no dependencies.
    pub fn dependency_graph(&self) -> HashMap<ExpertId, Vec<ExpertId>> {
        let mut graph = HashMap::new();
        for (i, expert) in self.steps.iter().enumerate() {
            let deps = if i + 1 < self.steps.len() {
                vec![self.steps[i + 1].clone()]
            } else {
                vec![]
            };
            graph.insert(expert.clone(), deps);
        }
        graph
    }
}
```

- [ ] **Step 4: Re-export DagRoute**

Modify `synapse-core/src/swarm/mod.rs`:

```rust
pub use dag::DagRoute;
```

Run: `cargo test swarm::dag -p synapse-core`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add synapse-core/src/shared/domain_error.rs synapse-core/src/swarm/dag.rs synapse-core/src/swarm/mod.rs
git commit -m "feat(swarm): add DAG route value object with expert dependency graph"
```

---

