use tonic::{transport::Server, Request, Response, Status};
use tokio_stream::wrappers::ReceiverStream;
use std::net::SocketAddr;
use log::{info, error};

// Generated proto code
pub mod hft {
    tonic::include_proto!("hft");
}

mod algorithms;
mod services;
use services::{TradingServiceImpl};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let addr: SocketAddr = "[::1]:50051".parse()?;
    let trading_service = TradingServiceImpl::new();
    
    info!("🚀 HFT gRPC Server starting on {}", addr);
    info!("📊 Services: Trading, Portfolio, Optimization");
    info!("⚡ Low-latency Rust-Python communication enabled");
    
    Server::builder()
        .add_service(hft::trading_service_server::TradingServiceServer::new(trading_service))
        .serve(addr)
        .await?;
    
    Ok(())
}
