pub mod trading {
    include!(concat!(env!("OUT_DIR"), "/hft.rs"));
}

pub mod regime {
    include!(concat!(env!("OUT_DIR"), "/regime.rs"));
}

pub mod drift {
    include!(concat!(env!("OUT_DIR"), "/drift.rs"));
}

pub mod superspace {
    include!(concat!(env!("OUT_DIR"), "/superspace.rs"));
}

pub mod pair_discovery {
    include!(concat!(env!("OUT_DIR"), "/pair_discovery.rs"));
}

pub mod options {
    include!(concat!(env!("OUT_DIR"), "/options.rs"));
}

pub mod meanrev {
    tonic::include_proto!("meanrev");
}

use tonic::transport::Server;
use tracing::info;
use std::net::SocketAddr;

mod service;
mod models;
mod regime_service;
mod drift_service;
mod superspace_service;
mod options_service;
mod trading_service;

use service::MeanRevServiceImpl;
use regime_service::MyRegimeService;
use drift_service::MyDriftService;
use superspace_service::MySuperspaceService;
use options_service::MyOptionsService;
use trading_service::TradingServiceImpl;

pub async fn run_server(host: &str, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}", host, port).parse::<SocketAddr>()?;
    info!("Starting HFT gRPC server on {}", addr);

    let trading_service = TradingServiceImpl::new();
    let meanrev_service = MeanRevServiceImpl::new();
    let regime_service = MyRegimeService::default();
    let drift_service = MyDriftService::default();
    let superspace_service = MySuperspaceService::default();
    let options_service = MyOptionsService::default();

    Server::builder()
        .add_service(trading::trading_service_server::TradingServiceServer::new(trading_service))
        .add_service(meanrev::mean_rev_service_server::MeanRevServiceServer::new(meanrev_service))
        .add_service(regime::regime_service_server::RegimeServiceServer::new(regime_service))
        .add_service(drift::drift_service_server::DriftServiceServer::new(drift_service))
        .add_service(superspace::superspace_service_server::SuperspaceServiceServer::new(superspace_service))
        .add_service(options::options_service_server::OptionsServiceServer::new(options_service))
        .serve(addr)
        .await?;

    Ok(())
}
