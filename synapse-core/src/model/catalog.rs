use super::{ModelEntity, ModelId};
use crate::shared::{DomainError, DomainEvent};
use uuid::Uuid;

/// The curated catalog of Synapse-compatible models.
///
/// The [`Catalog`] is an aggregate root that enforces registration
/// invariants: no duplicate model IDs, and all models must pass
/// structural validation.
///
/// In V1, the catalog is curated by Synapse Inc. Community proposals
/// are accepted via GitHub PR.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    models: Vec<ModelEntity>,
}

impl Catalog {
    /// Creates an empty catalog.
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    /// Registers a model in the catalog.
    ///
    /// # Errors
    ///
    /// Returns [`DomainError::DuplicateModel`] if a model with the
    /// same [`ModelId`] is already registered.
    pub fn register(&mut self, model: ModelEntity) -> Result<Vec<DomainEvent>, DomainError> {
        if self.models.iter().any(|m| m.id == model.id) {
            return Err(DomainError::DuplicateModel { model_id: model.id.to_string() });
        }

        let event = DomainEvent::ModelAdded {
            event_id: Uuid::new_v4(),
            model_id: model.id.to_string(),
            experts: model.experts,
            active_per_token: model.active_per_token,
        };

        self.models.push(model);
        Ok(vec![event])
    }

    /// Returns all registered models.
    pub fn list(&self) -> &[ModelEntity] {
        &self.models
    }

    /// Finds a model by its [`ModelId`].
    pub fn find(&self, id: &ModelId) -> Option<&ModelEntity> {
        self.models.iter().find(|m| m.id == *id)
    }

    /// Removes a model from the catalog.
    ///
    /// Returns `None` if no model with the given ID was registered.
    pub fn remove(&mut self, id: &ModelId) -> Option<Vec<DomainEvent>> {
        let pos = self.models.iter().position(|m| m.id == *id)?;
        self.models.remove(pos);
        Some(vec![DomainEvent::ModelRemoved { event_id: Uuid::new_v4(), model_id: id.to_string() }])
    }

    /// Returns the number of registered models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns `true` if the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model(id: &str, experts: u32, active: u32) -> ModelEntity {
        ModelEntity::new(
            ModelId::new(id).unwrap(),
            format!("{id} Name"),
            String::new(),
            String::new(),
            experts,
            active,
            0.0,
            0.0,
            0,
            String::new(),
            String::new(),
            None,
        )
    }

    #[test]
    fn new_catalog_is_empty() {
        let catalog = Catalog::new();
        assert!(catalog.is_empty());
        assert_eq!(catalog.len(), 0);
    }

    #[test]
    fn register_adds_model() {
        let mut catalog = Catalog::new();
        let model = make_model("kimi-k3", 896, 16);
        let events = catalog.register(model).unwrap();
        assert_eq!(catalog.len(), 1);
        assert!(!events.is_empty());
        assert!(matches!(events[0], DomainEvent::ModelAdded { .. }));
    }

    #[test]
    fn register_duplicate_rejected() {
        let mut catalog = Catalog::new();
        catalog.register(make_model("kimi-k3", 896, 16)).unwrap();
        let err = catalog.register(make_model("kimi-k3", 896, 16)).unwrap_err();
        assert!(matches!(err, DomainError::DuplicateModel { .. }));
    }

    #[test]
    fn list_returns_all_models() {
        let mut catalog = Catalog::new();
        catalog.register(make_model("a", 8, 2)).unwrap();
        catalog.register(make_model("b", 64, 8)).unwrap();
        assert_eq!(catalog.list().len(), 2);
    }

    #[test]
    fn find_by_id() {
        let mut catalog = Catalog::new();
        let id = ModelId::new("kimi-k3").unwrap();
        catalog.register(make_model("kimi-k3", 896, 16)).unwrap();
        let found = catalog.find(&id).unwrap();
        assert_eq!(found.experts, 896);
    }

    #[test]
    fn non_empty_catalog_is_not_empty() {
        let mut catalog = Catalog::new();
        catalog.register(make_model("kimi-k3", 896, 16)).unwrap();
        assert!(!catalog.is_empty());
        assert_eq!(catalog.len(), 1);
    }

    #[test]
    fn find_unknown_returns_none() {
        let catalog = Catalog::new();
        let id = ModelId::new("unknown").unwrap();
        assert!(catalog.find(&id).is_none());
    }

    #[test]
    fn remove_existing_model() {
        let mut catalog = Catalog::new();
        let id = ModelId::new("to-remove").unwrap();
        catalog.register(make_model("to-remove", 8, 2)).unwrap();
        assert_eq!(catalog.len(), 1);

        let events = catalog.remove(&id).unwrap();
        assert_eq!(catalog.len(), 0);
        assert!(matches!(events[0], DomainEvent::ModelRemoved { .. }));
    }

    #[test]
    fn remove_unknown_returns_none() {
        let mut catalog = Catalog::new();
        let id = ModelId::new("unknown").unwrap();
        assert!(catalog.remove(&id).is_none());
    }

    #[test]
    fn register_multiple_models() {
        let mut catalog = Catalog::new();
        catalog.register(make_model("a", 8, 2)).unwrap();
        catalog.register(make_model("b", 64, 6)).unwrap();
        catalog.register(make_model("c", 896, 16)).unwrap();
        assert_eq!(catalog.len(), 3);
    }

    #[test]
    fn default_catalog_is_empty() {
        let catalog = Catalog::default();
        assert!(catalog.is_empty());
    }
}
