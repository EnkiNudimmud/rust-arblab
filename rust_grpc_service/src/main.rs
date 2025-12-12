use rust_grpc_service::run_server;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    run_server("127.0.0.1", 50051).await?;

    Ok(())
}
