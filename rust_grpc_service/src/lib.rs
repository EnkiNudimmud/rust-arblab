pub mod meanrev {
    tonic::include_proto!("meanrev");
}

use tonic::transport::Server;
use tracing::info;
use std::net::SocketAddr;

mod service;
use service::MeanRevServiceImpl;

pub async fn run_server(host: &str, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}", host, port).parse::<SocketAddr>()?;
    info!("Starting MeanRev gRPC server on {}", addr);

    let service = MeanRevServiceImpl::new();

    Server::builder()
        .add_service(meanrev::mean_rev_service_server::MeanRevServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
