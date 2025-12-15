use tonic::{Request, Response, Status};
use crate::hft::trading::*;
use crate::models;

pub struct TradingServiceImpl;

impl TradingServiceImpl {
    pub fn new() -> Self {
        TradingServiceImpl
    }
}

#[tonic::async_trait]
impl trading_service_server::TradingService for TradingServiceImpl {
    async fn calculate_mean_reversion(
        &self,
        request: Request<MeanReversionRequest>,
    ) -> Result<Response<MeanReversionResponse>, Status> {
        let req = request.into_inner();
        
        if req.prices.is_empty() {
            return Err(Status::invalid_argument("Prices array is empty"));
        }
        
        let lookback = req.lookback as usize;
        if req.prices.len() < lookback {
            return Err(Status::invalid_argument("Not enough data for lookback period"));
        }

        // Calculate z-score based on rolling statistics
        let n = req.prices.len();
        let last_price = req.prices[n - 1];
        
        let start_idx = if n > lookback { n - lookback } else { 0 };
        let lookback_prices = &req.prices[start_idx..];
        
        let mean = lookback_prices.iter().sum::<f64>() / lookback_prices.len() as f64;
        let variance = lookback_prices.iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>() / lookback_prices.len() as f64;
        let std_dev = variance.sqrt();

        let zscore = if std_dev > 0.0 {
            (last_price - mean) / std_dev
        } else {
            0.0
        };

        let signal = if zscore.abs() > req.threshold { zscore } else { 0.0 };
        let entry_signal = zscore < -req.threshold;
        let exit_signal = zscore > req.threshold;

        let mut metrics = std::collections::HashMap::new();
        metrics.insert("mean".to_string(), mean);
        metrics.insert("std_dev".to_string(), std_dev);
        metrics.insert("lookback".to_string(), lookback as f64);

        Ok(Response::new(MeanReversionResponse {
            signal,
            zscore,
            entry_signal,
            exit_signal,
            metrics,
        }))
    }

    async fn optimize_portfolio(
        &self,
        request: Request<PortfolioOptimizationRequest>,
    ) -> Result<Response<PortfolioOptimizationResponse>, Status> {
        let req = request.into_inner();
        
        if req.prices.is_empty() {
            return Err(Status::invalid_argument("No price data provided"));
        }

        // Extract price vectors
        let mut price_data: Vec<Vec<f64>> = Vec::new();
        let mut symbols: Vec<String> = Vec::new();
        
        for price_vec in &req.prices {
            if price_vec.prices.is_empty() {
                return Err(Status::invalid_argument("Empty price vector"));
            }
            price_data.push(price_vec.prices.clone());
            symbols.push(price_vec.symbol.clone());
        }

        let n_assets = price_data.len();
        
        // Simple equal weight for now
        let weights = vec![1.0 / n_assets as f64; n_assets];
        
        // Calculate returns
        let mut returns = Vec::new();
        for prices in &price_data {
            if prices.len() < 2 {
                return Err(Status::invalid_argument("Need at least 2 price points"));
            }
            for i in 1..prices.len() {
                returns.push((prices[i] - prices[i-1]) / prices[i-1]);
            }
        }

        let expected_return = if !returns.is_empty() {
            returns.iter().sum::<f64>() / returns.len() as f64
        } else {
            0.0
        };

        let variance = returns.iter()
            .map(|r| (r - expected_return).powi(2))
            .sum::<f64>() / returns.len().max(1) as f64;
        let volatility = variance.sqrt();

        let risk_free_rate = req.parameters.get("risk_free_rate").copied().unwrap_or(0.02);
        let sharpe_ratio = if volatility > 0.0 {
            (expected_return - risk_free_rate) / volatility
        } else {
            0.0
        };

        let mut metrics = std::collections::HashMap::new();
        metrics.insert("n_assets".to_string(), n_assets as f64);
        metrics.insert("method".to_string(), if req.method == "markowitz" { 1.0 } else { 0.0 });

        Ok(Response::new(PortfolioOptimizationResponse {
            weights,
            expected_return,
            volatility,
            sharpe_ratio,
            metrics,
        }))
    }

    async fn detect_regime(
        &self,
        request: Request<RegimeDetectionRequest>,
    ) -> Result<Response<RegimeDetectionResponse>, Status> {
        let req = request.into_inner();
        
        if req.returns.is_empty() {
            return Err(Status::invalid_argument("Returns array is empty"));
        }

        let n_regimes = req.n_regimes as usize;
        if n_regimes < 2 {
            return Err(Status::invalid_argument("Need at least 2 regimes"));
        }

        // Simplified HMM - classify based on volatility
        let mean_ret = req.returns.iter().sum::<f64>() / req.returns.len() as f64;
        let variance = req.returns.iter()
            .map(|r| (r - mean_ret).powi(2))
            .sum::<f64>() / req.returns.len() as f64;
        let volatility = variance.sqrt();

        // Determine regime based on current volatility vs average
        let last_ret = req.returns[req.returns.len() - 1];
        let current_regime = if last_ret.abs() > volatility { 1 } else { 0 };
        
        let mut regime_probs = vec![0.0; n_regimes];
        regime_probs[current_regime as usize] = 0.7;
        for i in 0..n_regimes {
            if i != current_regime as usize {
                regime_probs[i] = 0.3 / (n_regimes - 1) as f64;
            }
        }

        let mut regimes = Vec::new();
        for i in 0..n_regimes {
            regimes.push(RegimeParameters {
                mean: mean_ret * (1.0 + i as f64 * 0.1),
                volatility: volatility * (1.0 + i as f64 * 0.2),
                persistence: 0.8,
            });
        }

        Ok(Response::new(RegimeDetectionResponse {
            current_regime: current_regime as i32,
            regime_probabilities: regime_probs,
            regimes,
        }))
    }

    type StreamMarketDataStream = tokio_stream::wrappers::ReceiverStream<Result<MarketDataUpdate, Status>>;

    async fn stream_market_data(
        &self,
        _request: Request<StreamRequest>,
    ) -> Result<Response<Self::StreamMarketDataStream>, Status> {
        let (_tx, rx) = tokio::sync::mpsc::channel(10);
        
        // For now, return empty stream
        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn get_order_book(
        &self,
        _request: Request<OrderBookRequest>,
    ) -> Result<Response<OrderBookResponse>, Status> {
        Err(Status::unimplemented("Order book not implemented"))
    }

    async fn run_hmm(
        &self,
        _request: Request<HmmRequest>,
    ) -> Result<Response<HmmResponse>, Status> {
        Err(Status::unimplemented("HMM not implemented"))
    }

    async fn run_mcmc(
        &self,
        _request: Request<McmcRequest>,
    ) -> Result<Response<McmcResponse>, Status> {
        Err(Status::unimplemented("MCMC not implemented"))
    }

    async fn calculate_sparse_portfolio(
        &self,
        _request: Request<SparsePortfolioRequest>,
    ) -> Result<Response<SparsePortfolioResponse>, Status> {
        Err(Status::unimplemented("Sparse portfolio not implemented"))
    }

    async fn box_tao_decomposition(
        &self,
        _request: Request<BoxTaoRequest>,
    ) -> Result<Response<BoxTaoResponse>, Status> {
        Err(Status::unimplemented("Box-Tao decomposition not implemented"))
    }
}
