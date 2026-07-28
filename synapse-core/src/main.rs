use synapse_core::gateway;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let app = gateway::api::build_router();

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    println!("Synapse Gateway listening on http://0.0.0.0:8000");

    axum::serve(listener, app).await.unwrap();
}
