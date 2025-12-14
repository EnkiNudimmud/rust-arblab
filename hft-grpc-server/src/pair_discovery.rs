//! Pair Discovery gRPC Service Implementation
//!
//! High-performance pair discovery using Rust backend with Tokio async runtime

use tonic::{Request, Response, Status};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use optimizr::optimal_control::{HJBSolver, HJBConfig};
use crate::finance::*;

// Import generated protobuf code
pub mod proto {
    tonic::include_proto!("pair_discovery");
}

use proto::{
    pair_discovery_service_server::PairDiscoveryService,
    *,
};

#[derive(Debug, Default)]
pub struct PairDiscoveryServiceImpl;

#[tonic::async_trait]
impl PairDiscoveryService for PairDiscoveryServiceImpl {
    /// Test a single pair
    async fn test_pair(
        &self,
        request: Request<PairTestRequest>,
    ) -> std::result::Result<Response<PairTestResponse>, Status> {
        let req = request.into_inner();
        
        // Validate inputs
        if req.prices1.len() != req.prices2.len() {
            return Err(Status::invalid_argument("Price arrays must have same length"));
        }
        
        if req.prices1.len() < 20 {
            return Err(Status::invalid_argument("Need at least 20 data points"));
        }
        
        let significance = if req.significance > 0.0 { req.significance } else { 0.05 };
        let min_hurst = if req.min_hurst > 0.0 { req.min_hurst } else { 0.45 };
        let transaction_cost = if req.transaction_cost > 0.0 { req.transaction_cost } else { 0.001 };
        
        // Run analysis (blocking, but fast)
        let result = tokio::task::spawn_blocking(move || {
            test_pair_impl(
                &req.symbol1,
                &req.symbol2,
                &req.prices1,
                &req.prices2,
                significance,
                min_hurst,
                transaction_cost,
            )
        })
        .await
        .map_err(|e| Status::internal(format!("Task join error: {}", e)))?;
        
        Ok(Response::new(result))
    }
    
    /// Batch discovery with streaming results
    type DiscoverPairsStream = ReceiverStream<std::result::Result<PairDiscoveryUpdate, Status>>;
    
    async fn discover_pairs(
        &self,
        request: Request<BatchDiscoveryRequest>,
    ) -> std::result::Result<Response<Self::DiscoverPairsStream>, Status> {
        let req = request.into_inner();
        
        let (tx, rx) = mpsc::channel(128);
        
        // Spawn background task for discovery
        tokio::spawn(async move {
            if let Err(e) = discover_pairs_impl(req, tx).await {
                log::error!("Discovery error: {}", e);
            }
        });
        
        Ok(Response::new(ReceiverStream::new(rx)))
    }
    
    /// Solve HJB equation
    async fn solve_hjb(
        &self,
        request: Request<HjbRequest>,
    ) -> std::result::Result<Response<HjbResponse>, Status> {
        let req = request.into_inner();
        
        let config = HJBConfig {
            kappa: req.kappa,
            theta: req.theta,
            sigma: req.sigma,
            rho: if req.rho > 0.0 { req.rho } else { 0.04 },
            transaction_cost: if req.transaction_cost > 0.0 { req.transaction_cost } else { 0.001 },
            n_points: if req.n_points > 0 { req.n_points as usize } else { 200 },
            max_iter: if req.max_iter > 0 { req.max_iter as usize } else { 2000 },
            tolerance: if req.tolerance > 0.0 { req.tolerance } else { 1e-6 },
            n_std: if req.n_std > 0.0 { req.n_std } else { 4.0 },
        };
        
        let result = tokio::task::spawn_blocking(move || {
            let solver = HJBSolver::new(config)
                .map_err(|e| Status::invalid_argument(format!("{}", e)))?;
            
            let hjb_result = solver.solve()
                .map_err(|e| Status::internal(format!("{}", e)))?;
            
            Ok::<_, Status>(HjbResponse {
                lower_boundary: hjb_result.lower_boundary,
                upper_boundary: hjb_result.upper_boundary,
                residual: hjb_result.residual,
                iterations: hjb_result.iterations as i32,
                x_grid: hjb_result.x.to_vec(),
                value_function: hjb_result.value.to_vec(),
            })
        })
        .await
        .map_err(|e| Status::internal(format!("Task error: {}", e)))??;
        
        Ok(Response::new(result))
    }
    
    /// Estimate OU parameters
    async fn estimate_ou(
        &self,
        request: Request<OuEstimationRequest>,
    ) -> std::result::Result<Response<OuEstimationResponse>, Status> {
        let req = request.into_inner();
        
        let dt = if req.dt > 0.0 { req.dt } else { 1.0 / 252.0 };
        
        let result = tokio::task::spawn_blocking(move || {
            let params = if req.use_mle {
                estimate_ou_params_mle(&req.spread, dt)
            } else {
                estimate_ou_params(&req.spread, dt)
            }
            .map_err(|e| Status::invalid_argument(format!("{}", e)))?;
            
            Ok::<_, Status>(OuEstimationResponse {
                kappa: params.kappa,
                theta: params.theta,
                sigma: params.sigma,
                half_life: params.half_life,
            })
        })
        .await
        .map_err(|e| Status::internal(format!("Task error: {}", e)))??;
        
        Ok(Response::new(result))
    }
    
    /// Test cointegration
    async fn test_cointegration(
        &self,
        request: Request<CointegrationRequest>,
    ) -> std::result::Result<Response<CointegrationResponse>, Status> {
        let req = request.into_inner();
        
        let significance = if req.significance > 0.0 { req.significance } else { 0.05 };
        
        let result = tokio::task::spawn_blocking(move || {
            let coint_result = engle_granger_test(&req.y, &req.x, significance)
                .map_err(|e| Status::invalid_argument(format!("{}", e)))?;
            
            Ok::<_, Status>(CointegrationResponse {
                beta: coint_result.beta,
                adf_statistic: coint_result.adf_statistic,
                p_value: coint_result.p_value,
                is_cointegrated: coint_result.is_cointegrated,
                spread: coint_result.spread,
            })
        })
        .await
        .map_err(|e| Status::internal(format!("Task error: {}", e)))??;
        
        Ok(Response::new(result))
    }
    
    /// Calculate Hurst exponent
    async fn calculate_hurst(
        &self,
        request: Request<HurstRequest>,
    ) -> std::result::Result<Response<HurstResponse>, Status> {
        let req = request.into_inner();
        
        let max_lag = if req.max_lag > 0 { req.max_lag as usize } else { 20 };
        
        let result = tokio::task::spawn_blocking(move || {
            let hurst = if req.use_dfa {
                let min_window = max_lag;
                let max_window = req.series.len() / 4;
                hurst_dfa(&req.series, min_window, max_window)
            } else {
                hurst_exponent(&req.series, max_lag)
            }
            .map_err(|e| Status::invalid_argument(format!("{}", e)))?;
            
            Ok::<_, Status>(HurstResponse { hurst })
        })
        .await
        .map_err(|e| Status::internal(format!("Task error: {}", e)))??;
        
        Ok(Response::new(result))
    }
    
    /// Backtest strategy
    async fn backtest_strategy(
        &self,
        request: Request<BacktestRequest>,
    ) -> std::result::Result<Response<BacktestResponse>, Status> {
        let req = request.into_inner();
        
        let transaction_cost = if req.transaction_cost > 0.0 { req.transaction_cost } else { 0.001 };
        
        let result = tokio::task::spawn_blocking(move || {
            let backtest_result = backtest_optimal_switching(
                &req.spread,
                req.lower_bound,
                req.upper_bound,
                transaction_cost,
            )
            .map_err(|e| Status::invalid_argument(format!("{}", e)))?;
            
            Ok::<_, Status>(BacktestResponse {
                total_return: backtest_result.total_return,
                sharpe_ratio: backtest_result.sharpe_ratio,
                max_drawdown: backtest_result.max_drawdown,
                num_trades: backtest_result.num_trades as i32,
                win_rate: backtest_result.win_rate,
                pnl: backtest_result.pnl,
                avg_holding_period: backtest_result.avg_holding_period,
                profit_factor: backtest_result.profit_factor,
            })
        })
        .await
        .map_err(|e| Status::internal(format!("Task error: {}", e)))??;
        
        Ok(Response::new(result))
    }
}

/// Test a single pair (internal implementation)
fn test_pair_impl(
    symbol1: &str,
    symbol2: &str,
    prices1: &[f64],
    prices2: &[f64],
    significance: f64,
    min_hurst: f64,
    transaction_cost: f64,
) -> PairTestResponse {
    // 1. Cointegration test
    let coint_result = match engle_granger_test(prices1, prices2, significance) {
        Ok(r) => r,
        Err(e) => {
            return PairTestResponse {
                success: false,
                symbol1: symbol1.to_string(),
                symbol2: symbol2.to_string(),
                error: format!("Cointegration test failed: {}", e),
                ..Default::default()
            };
        }
    };
    
    if !coint_result.is_cointegrated {
        return PairTestResponse {
            success: false,
            symbol1: symbol1.to_string(),
            symbol2: symbol2.to_string(),
            error: "Not cointegrated".to_string(),
            ..Default::default()
        };
    }
    
    // 2. Hurst exponent
    let hurst = match hurst_exponent(&coint_result.spread, 20) {
        Ok(h) => h,
        Err(e) => {
            return PairTestResponse {
                success: false,
                symbol1: symbol1.to_string(),
                symbol2: symbol2.to_string(),
                error: format!("Hurst calculation failed: {}", e),
                ..Default::default()
            };
        }
    };
    
    if hurst.is_nan() || hurst > min_hurst {
        return PairTestResponse {
            success: false,
            symbol1: symbol1.to_string(),
            symbol2: symbol2.to_string(),
            error: format!("Not mean-reverting (Hurst: {:.3})", hurst),
            ..Default::default()
        };
    }
    
    // 3. OU parameters
    let ou_params = match estimate_ou_params(&coint_result.spread, 1.0 / 252.0) {
        Ok(p) => p,
        Err(e) => {
            return PairTestResponse {
                success: false,
                symbol1: symbol1.to_string(),
                symbol2: symbol2.to_string(),
                error: format!("OU estimation failed: {}", e),
                ..Default::default()
            };
        }
    };
    
    if ou_params.kappa <= 0.0 || ou_params.kappa.is_nan() {
        return PairTestResponse {
            success: false,
            symbol1: symbol1.to_string(),
            symbol2: symbol2.to_string(),
            error: format!("Invalid kappa: {}", ou_params.kappa),
            ..Default::default()
        };
    }
    
    // 4. HJB solver
    let hjb_config = HJBConfig {
        kappa: ou_params.kappa,
        theta: ou_params.theta,
        sigma: ou_params.sigma,
        rho: 0.04,
        transaction_cost,
        n_points: 200,
        max_iter: 2000,
        tolerance: 1e-6,
        n_std: 4.0,
    };
    
    let solver = match HJBSolver::new(hjb_config) {
        Ok(s) => s,
        Err(e) => {
            return PairTestResponse {
                success: false,
                symbol1: symbol1.to_string(),
                symbol2: symbol2.to_string(),
                error: format!("HJB solver creation failed: {}", e),
                ..Default::default()
            };
        }
    };
    
    let hjb_result = match solver.solve() {
        Ok(r) => r,
        Err(e) => {
            return PairTestResponse {
                success: false,
                symbol1: symbol1.to_string(),
                symbol2: symbol2.to_string(),
                error: format!("HJB solve failed: {}", e),
                ..Default::default()
            };
        }
    };
    
    // 5. Backtest
    let backtest_result = match backtest_optimal_switching(
        &coint_result.spread,
        hjb_result.lower_boundary,
        hjb_result.upper_boundary,
        transaction_cost,
    ) {
        Ok(r) => r,
        Err(e) => {
            return PairTestResponse {
                success: false,
                symbol1: symbol1.to_string(),
                symbol2: symbol2.to_string(),
                error: format!("Backtest failed: {}", e),
                ..Default::default()
            };
        }
    };
    
    // Calculate scores
    let coint_score = 1.0 - coint_result.p_value;
    let meanrev_score = (0.5 - hurst) / 0.2;
    let profit_score = backtest_result.total_return.max(0.0);
    let combined_score = coint_score * meanrev_score * (1.0 + profit_score);
    
    PairTestResponse {
        success: true,
        symbol1: symbol1.to_string(),
        symbol2: symbol2.to_string(),
        beta: coint_result.beta,
        p_value: coint_result.p_value,
        is_cointegrated: coint_result.is_cointegrated,
        hurst,
        kappa: ou_params.kappa,
        theta: ou_params.theta,
        sigma: ou_params.sigma,
        half_life: ou_params.half_life,
        lower_boundary: hjb_result.lower_boundary,
        upper_boundary: hjb_result.upper_boundary,
        total_return: backtest_result.total_return,
        sharpe_ratio: backtest_result.sharpe_ratio,
        max_drawdown: backtest_result.max_drawdown,
        num_trades: backtest_result.num_trades as i32,
        win_rate: backtest_result.win_rate,
        coint_score,
        meanrev_score,
        profit_score,
        combined_score,
        error: String::new(),
    }
}

/// Batch discovery implementation
async fn discover_pairs_impl(
    req: BatchDiscoveryRequest,
    tx: mpsc::Sender<std::result::Result<PairDiscoveryUpdate, Status>>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let significance = if req.significance > 0.0 { req.significance } else { 0.05 };
    let min_hurst = if req.min_hurst > 0.0 { req.min_hurst } else { 0.45 };
    let transaction_cost = if req.transaction_cost > 0.0 { req.transaction_cost } else { 0.001 };
    
    let n_symbols = req.symbols.len();
    let mut pairs_to_test = Vec::new();
    
    // Generate all unique pairs
    for i in 0..n_symbols {
        for j in (i + 1)..n_symbols {
            pairs_to_test.push((i, j));
            if req.max_pairs > 0 && pairs_to_test.len() >= req.max_pairs as usize {
                break;
            }
        }
        if req.max_pairs > 0 && pairs_to_test.len() >= req.max_pairs as usize {
            break;
        }
    }
    
    let total_pairs = pairs_to_test.len();
    let symbols = Arc::new(req.symbols);
    let pairs_tested = Arc::new(Mutex::new(0));
    let pairs_found = Arc::new(Mutex::new(0));
    
    // Parallel processing with Rayon
    let results: Vec<_> = pairs_to_test
        .into_par_iter()
        .map(|(i, j)| {
            let sym1 = &symbols[i];
            let sym2 = &symbols[j];
            
            let result = test_pair_impl(
                &sym1.symbol,
                &sym2.symbol,
                &sym1.prices,
                &sym2.prices,
                significance,
                min_hurst,
                transaction_cost,
            );
            
            // Update counters
            {
                let mut tested = pairs_tested.lock().unwrap();
                *tested += 1;
                if result.success {
                    let mut found = pairs_found.lock().unwrap();
                    *found += 1;
                }
            }
            
            result
        })
        .collect();
    
    // Send results
    let mut tested = 0;
    let mut found = 0;
    
    for result in results {
        tested += 1;
        if result.success {
            found += 1;
        }
        
        let update = PairDiscoveryUpdate {
            pairs_tested: tested,
            pairs_found: found,
            progress: tested as f64 / total_pairs as f64,
            result: Some(result),
            is_final: tested == total_pairs as i32,
        };
        
        if tx.send(Ok(update)).await.is_err() {
            break;
        }
    }
    
    Ok(())
}
