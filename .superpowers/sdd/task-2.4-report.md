# Task 2.4: DAG Swarm Domain — Report

**Status:** Complete

**Commit:** `ab2e967` — `feat(swarm): add DAG route value object with expert dependency graph`

---

## Files Changed

| File | Action |
|------|--------|
| `synapse-core/src/shared/domain_error.rs` | Modified — added `InvalidRoute { reason: String }` variant + display test |
| `synapse-core/src/swarm/dag.rs` | Created — `DagRoute` struct, `new`, `len`, `is_empty`, `dependency_graph`, 4 unit tests |
| `synapse-core/src/swarm/mod.rs` | Modified — added `pub use dag::DagRoute;` |

## Test Summary

```
cargo test -p synapse-core: 148 passed (4 suites, 0.00s)
```

- **domain_error tests:** 14 passed (incl. `invalid_route_display`)
- **dag tests:** 4 passed (empty rejection, mixed-model rejection, valid construction, dependency graph)
- **cargo clippy -- -D warnings:** clean
- **cargo fmt:** clean

## Implementation Notes

- `DagRoute` is a pure domain value object — no I/O, no infrastructure dependency.
- Validation rules:
  - Empty step vectors are rejected with `DomainError::InvalidRoute`.
  - Steps from a different model than the route's `model` are rejected.
- `dependency_graph()` returns a `HashMap<ExpertId, Vec<ExpertId>>` where each expert maps to its successor (leaf maps to empty vec). This is a chain topology — step N depends on step N+1.
- `ExpertId` derives `Hash + Eq + Clone`, so it works seamlessly as a `HashMap` key.
- `ModelId` exposes `as_str()` for reading model names; no `Display` is assumed (tests use `as_str()` directly).

## Concerns

- The dependency graph is deliberately a simple linear chain. The brief's `expert_dependency_graph(steps)` static method signature was replaced with `dependency_graph(&self)` — an instance method — since the route already owns the steps. This avoids an unnecessary `&[ExpertId]` argument that must always match `self.steps`.
- `steps()` accessor returns `&[ExpertId]` rather than `&Vec<ExpertId>` for idiomatic Rust.
