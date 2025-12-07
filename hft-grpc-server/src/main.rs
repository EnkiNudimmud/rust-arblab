use tonic::transport::Server;
use std::net::SocketAddr;
use log::info;

// Generated proto code
pub mod hft {
    tonic::include_proto!("hft");
}

mod algorithms;
mod services;
mod finance;
mod pair_discovery;

use services::{TradingServiceImpl};
use pair_discovery::{PairDiscoveryServiceImpl, proto::pair_discovery_service_server};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let addr: SocketAddr = "[::1]:50051".parse()?;
    let trading_service = TradingServiceImpl::new();
    let pair_discovery_service = PairDiscoveryServiceImpl::default();
    
    info!("🚀 HFT gRPC Server starting on {}", addr);
    info!("📊 Services: Trading, Portfolio, Optimization, Pair Discovery");
    info!("⚡ Low-latency Rust-Python communication enabled");
    info!("🔬 Optimal Control: HJB solver, OU estimation, Cointegration tests");
    
    Server::builder()
        .add_service(hft::trading_service_server::TradingServiceServer::new(trading_service))
        .add_service(pair_discovery_service_server::PairDiscoveryServiceServer::new(pair_discovery_service))
        .serve(addr)
        .await?;
    
    Ok(())
}
