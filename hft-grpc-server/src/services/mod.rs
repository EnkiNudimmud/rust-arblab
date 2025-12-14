use tonic::{Request, Response, Status};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use log::info;
use std::collections::HashMap;

use crate::hft::{
    trading_service_server::TradingService,
    *,
};
use crate::algorithms::{meanrev, portfolio, hmm, sparse};

pub struct TradingServiceImpl {
    // Add state here if needed (e.g., market data cache)
}

impl TradingServiceImpl {
    pub fn new() -> Self {
        Self {}
    }
}

#[tonic::async_trait]
impl TradingService for TradingServiceImpl {
    async fn calculate_mean_reversion(
        &self,
        request: Request<MeanReversionRequest>,
    ) -> Result<Response<MeanReversionResponse>, Status> {
        let req = request.into_inner();
        info!("Mean reversion request: {} prices, threshold={}", 
              req.prices.len(), req.threshold);
        
        if req.prices.is_empty() {
            return Err(Status::invalid_argument("Prices array is empty"));
        }
        
        let lookback = req.lookback as usize;
        
        if req.prices.len() < lookback {
            return Err(Status::invalid_argument("Not enough data for lookback period"));
        }
        
        // Use real algorithm
        let result = meanrev::calculate_mean_reversion(&req.prices, lookback, req.threshold);
        
        let response = MeanReversionResponse {
            signal: result.signal,
            zscore: result.zscore,
            entry_signal: result.entry_signal,
            exit_signal: result.exit_signal,
            metrics: result.metrics,
        };
        
        Ok(Response::new(response))
    }
    
    async fn optimize_portfolio(
        &self,
        request: Request<PortfolioOptimizationRequest>,
    ) -> Result<Response<PortfolioOptimizationResponse>, Status> {
        let req = request.into_inner();
        info!("Portfolio optimization: {} assets, method={}", 
              req.prices.len(), req.method);
        
        if req.prices.is_empty() {
            return Err(Status::invalid_argument("No price data provided"));
        }
        
        // Convert PriceVector to Vec<Vec<f64>>
        let prices: Vec<Vec<f64>> = req.prices.iter()
            .map(|pv| pv.prices.clone())
            .collect();
        
        // Extract parameters
        let risk_free_rate = req.parameters.get("risk_free_rate").copied().unwrap_or(0.02);
        let target_return = req.parameters.get("target_return").copied();
        
        // Use real algorithm
        let result = portfolio::optimize_portfolio(
            &prices,
            &req.method,
            risk_free_rate,
            target_return,
        );
        
        let response = PortfolioOptimizationResponse {
            weights: result.weights,
            expected_return: result.expected_return,
            volatility: result.volatility,
            sharpe_ratio: result.sharpe_ratio,
            metrics: result.metrics,
        };
        
        Ok(Response::new(response))
    }
    
    async fn detect_regime(
        &self,
        request: Request<RegimeDetectionRequest>,
    ) -> Result<Response<RegimeDetectionResponse>, Status> {
        let req = request.into_inner();
        info!("Regime detection: {} observations, {} regimes", 
              req.returns.len(), req.n_regimes);
        
        // Placeholder implementation
        // TODO: Integrate HMM from rust_core
        let response = RegimeDetectionResponse {
            current_regime: 0,
            regime_probabilities: vec![0.7, 0.2, 0.1],
            regimes: vec![
                RegimeParameters {
                    mean: 0.001,
                    volatility: 0.02,
                    persistence: 0.9,
                },
            ],
        };
        
        Ok(Response::new(response))
    }
    
    type StreamMarketDataStream = ReceiverStream<Result<MarketDataUpdate, Status>>;
    
    async fn stream_market_data(
        &self,
        request: Request<StreamRequest>,
    ) -> Result<Response<Self::StreamMarketDataStream>, Status> {
        let req = request.into_inner();
        info!("Market data stream: {:?} from {}", req.symbols, req.exchange);
        
        let (tx, rx) = mpsc::channel(128);
        
        // Spawn task to stream data
        tokio::spawn(async move {
            for i in 0..10 {
                for symbol in &req.symbols {
                    let update = MarketDataUpdate {
                        symbol: symbol.clone(),
                        timestamp: chrono::Utc::now().timestamp_millis(),
                        bid: 100.0 + i as f64,
                        ask: 100.1 + i as f64,
                        mid: 100.05 + i as f64,
                        volume: 1000.0,
                    };
                    
                    if tx.send(Ok(update)).await.is_err() {
                        break;
                    }
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(req.interval_ms as u64)).await;
            }
        });
        
        Ok(Response::new(ReceiverStream::new(rx)))
    }
    
    async fn get_order_book(
        &self,
        request: Request<OrderBookRequest>,
    ) -> Result<Response<OrderBookResponse>, Status> {
        let req = request.into_inner();
        info!("Order book request: {} on {}", req.symbol, req.exchange);
        
        // Placeholder
        let response = OrderBookResponse {
            symbol: req.symbol,
            timestamp: chrono::Utc::now().timestamp_millis(),
            bids: vec![],
            asks: vec![],
        };
        
        Ok(Response::new(response))
    }
    
    async fn run_hmm(
        &self,
        request: Request<HmmRequest>,
    ) -> Result<Response<HmmResponse>, Status> {
        let req = request.into_inner();
        info!("HMM: {} observations, {} states", 
              req.observations.len(), req.n_states);
        
        if req.observations.is_empty() {
            return Err(Status::invalid_argument("No observations provided"));
        }
        
        // Use real HMM algorithm
        let result = hmm::fit_hmm(
            &req.observations,
            req.n_states as usize,
            req.max_iterations as usize,
            req.tolerance,
        );
        
        let response = HmmResponse {
            state_probabilities: result.state_probabilities,
            transition_matrix: result.transition_matrix,
            emission_means: result.emission_means,
            emission_stds: result.emission_stds,
            log_likelihood: result.log_likelihood,
            converged: result.converged,
        };
        
        Ok(Response::new(response))
    }
    
    async fn run_mcmc(
        &self,
        request: Request<McmcRequest>,
    ) -> Result<Response<McmcResponse>, Status> {
        let req = request.into_inner();
        info!("MCMC: {} iterations", req.n_iterations);
        
        // TODO: Integrate actual MCMC from rust_connector
        let response = McmcResponse {
            samples: vec![],
            best_params: HashMap::new(),
            best_score: 0.0,
            acceptance_rate: 30,
        };
        
        Ok(Response::new(response))
    }
    
    async fn calculate_sparse_portfolio(
        &self,
        request: Request<SparsePortfolioRequest>,
    ) -> Result<Response<SparsePortfolioResponse>, Status> {
        let req = request.into_inner();
        info!("Sparse portfolio: {} assets, method={}", 
              req.prices.len(), req.method);
        
        if req.prices.is_empty() {
            return Err(Status::invalid_argument("No price data provided"));
        }
        
        // Convert PriceVector to Vec<Vec<f64>>
        let prices: Vec<Vec<f64>> = req.prices.iter()
            .map(|pv| pv.prices.clone())
            .collect();
        
        // Use alpha as lookback period (or default to 20)
        let lookback = if req.alpha > 0.0 { req.alpha as usize } else { 20 };
        
        // Use real sparse portfolio algorithm
        let result = sparse::sparse_meanrev_portfolio(
            &prices,
            req.lambda,
            lookback,
        );
        
        let response = SparsePortfolioResponse {
            weights: result.weights,
            n_assets_selected: result.n_assets_selected,
            objective_value: result.objective_value,
            metrics: result.metrics,
        };
        
        Ok(Response::new(response))
    }
    
    async fn box_tao_decomposition(
        &self,
        request: Request<BoxTaoRequest>,
    ) -> Result<Response<BoxTaoResponse>, Status> {
        let req = request.into_inner();
        info!("Box-Tao decomposition: {} assets", req.prices.len());
        
        // TODO: Integrate Box-Tao from rust_connector
        let response = BoxTaoResponse {
            low_rank: Some(Matrix {
                rows: 10,
                cols: 5,
                data: vec![0.0; 50],
            }),
            sparse: Some(Matrix {
                rows: 10,
                cols: 5,
                data: vec![0.0; 50],
            }),
            noise: Some(Matrix {
                rows: 10,
                cols: 5,
                data: vec![0.0; 50],
            }),
            iterations: 100,
            converged: true,
        };
        
        Ok(Response::new(response))
    }
}
