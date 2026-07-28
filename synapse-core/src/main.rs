#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    synapse_core::gateway::api::serve(synapse_core::gateway::api::DEFAULT_BIND_ADDR).await;
}
