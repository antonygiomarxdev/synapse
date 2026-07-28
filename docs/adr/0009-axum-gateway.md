# axum for HTTP Gateway

The gateway runs axum 0.8, the Tokio team's HTTP framework. It handles OpenAI-compatible REST endpoints (`GET /v1/models`, `POST /v1/chat/completions`), middleware (auth, CORS, rate limiting, logging), and integrates natively with the same tokio runtime used by libp2p.

**Why axum:** It's built on hyper/tokio by the Tokio team — the same runtime ecosystem as libp2p. This means axum handlers and libp2p event loops coexist in the same tokio runtime without bridging or context switching. Tower ecosystem provides production middleware (CORS via `CorsLayer`, tracing via `TraceLayer`, auth via `AuthLayer`). The extractor pattern (`Json<ModelEntry>`, `Result<Json<T>, StatusCode>`) makes handlers ergonomic and easily testable via `tower::ServiceExt::oneshot()`.

**Why not actix-web:** Actix uses its own runtime (actix-rt). Running alongside libp2p (tokio) requires bridging two runtimes — possible but adds complexity for no benefit. axum integrates natively.

**Why not warp:** Dead ecosystem. axum has more active development and wider adoption.

**V2+ streaming:** Server-sent events for token streaming are natively supported via axum's `StreamBody` and `Event` types. No framework change needed.
