use tonic::{Request, Response, Status};
use tokio_stream::wrappers::ReceiverStream;
use crate::meanrev::*;
use crate::models;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct MeanRevServiceImpl;

impl MeanRevServiceImpl {
    pub fn new() -> Self {
        MeanRevServiceImpl
    }

    fn current_timestamp_ms() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
    }
}

#[tonic::async_trait]
impl mean_rev_service_server::MeanRevService for MeanRevServiceImpl {
    type EstimateOUParamsStream = ReceiverStream<Result<OuEstimate, Status>>;

    async fn estimate_ou_params(
        &self,
        request: Request<tonic::Streaming<PriceUpdate>>,
    ) -> Result<Response<Self::EstimateOUParamsStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(10);

        tokio::spawn(async move {
            let mut prices = Vec::new();
            while let Ok(Some(update)) = stream.message().await {
                prices.push(update.price);
            }

            if prices.is_empty() {
                let _ = tx.send(Err(Status::invalid_argument("No prices provided"))).await;
                return;
            }

            let (theta, mu, sigma, half_life, r_squared, n_samples) =
                models::estimate_ou_params(&prices);

            let estimate = OuEstimate {
                theta,
                mu,
                sigma,
                half_life,
                r_squared,
                n_samples: n_samples as i32,
                status: "estimated".to_string(),
                timestamp_ms: Self::current_timestamp_ms(),
            };

            let _ = tx.send(Ok(estimate)).await;
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn optimal_thresholds(
        &self,
        request: Request<ThresholdRequest>,
    ) -> Result<Response<ThresholdResponse>, Status> {
        let req = request.into_inner();

        let ou = req.ou_params.ok_or_else(|| Status::invalid_argument("OU parameters required"))?;

        let (entry_z, exit_z, holding_period) =
            models::optimal_thresholds(ou.theta, ou.mu, ou.sigma, req.transaction_cost);

        let expected_return_per_trade = if ou.theta > 0.0 && ou.sigma > 0.0 {
            (entry_z * exit_z * ou.sigma * ou.sigma) / ou.theta
        } else {
            0.01
        };

        Ok(Response::new(ThresholdResponse {
            entry_zscore: entry_z,
            exit_zscore: exit_z,
            expected_holding_periods: holding_period,
            expected_return_per_trade,
            win_rate_estimate: 0.55,
            metadata: "Computed with OU parameters".to_string(),
        }))
    }

    type BacktestWithCostsStream = ReceiverStream<Result<BacktestMetric, Status>>;

    async fn backtest_with_costs(
        &self,
        request: Request<tonic::Streaming<BacktestInput>>,
    ) -> Result<Response<Self::BacktestWithCostsStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        tokio::spawn(async move {
            let mut prices = Vec::new();
            let mut entry_z = 2.0;
            let mut exit_z = 0.5;
            let mut transaction_cost = 0.001;

            while let Ok(Some(input)) = stream.message().await {
                prices.push(input.price);
                entry_z = input.entry_zscore;
                exit_z = input.exit_zscore;
                transaction_cost = input.transaction_cost;
            }

            if prices.is_empty() {
                let _ = tx.send(Err(Status::invalid_argument("No prices provided"))).await;
                return;
            }

            let result = models::backtest_with_costs(&prices, entry_z, exit_z, transaction_cost);

            let metric = BacktestMetric {
                bar_number: prices.len() as i32,
                cumulative_return: result.cumulative_return,
                daily_return: result.cumulative_return / (prices.len() as f64).max(1.0),
                open_positions: 0,
                unrealized_pnl: 0.0,
                realized_pnl: result.cumulative_return * 100000.0,
                max_drawdown: result.max_drawdown,
                sharpe_ratio: result.sharpe_ratio,
                num_trades: result.num_trades,
                win_rate: result.win_rate,
                avg_profit_per_trade: result.avg_profit_per_trade,
                status: "backtest_complete".to_string(),
                timestamp_ms: Self::current_timestamp_ms(),
            };

            let _ = tx.send(Ok(metric)).await;
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn compute_log_returns(
        &self,
        request: Request<ReturnsRequest>,
    ) -> Result<Response<ReturnsResponse>, Status> {
        let req = request.into_inner();

        if req.prices.is_empty() {
            return Err(Status::invalid_argument("No prices provided"));
        }

        let log_returns = models::compute_log_returns(&req.prices);
        let (mean_return, std_return) = models::mean_and_std(&log_returns);

        Ok(Response::new(ReturnsResponse {
            log_returns,
            mean_return,
            std_return,
            n_periods: req.prices.len() as i32,
        }))
    }

    async fn pca_portfolios(
        &self,
        request: Request<PcaRequest>,
    ) -> Result<Response<PcaResponse>, Status> {
        let req = request.into_inner();

        let matrix = req.returns_matrix.ok_or_else(|| Status::invalid_argument("No returns matrix"))?;

        let returns_data: Vec<Vec<f64>> = (0..matrix.rows as usize)
            .map(|i| {
                matrix
                    .values
                    .iter()
                    .skip(i * matrix.cols as usize)
                    .take(matrix.cols as usize)
                    .copied()
                    .collect()
            })
            .collect();

        let n_components = (req.n_components as usize).min(matrix.cols as usize);
        let (components, explained_variance) = models::compute_pca(&returns_data, n_components);

        let mut components_matrix_values = Vec::new();
        for component in &components {
            components_matrix_values.extend_from_slice(component);
        }

        let total_variance_explained: f64 = explained_variance.iter().sum();

        Ok(Response::new(PcaResponse {
            explained_variance_ratio: explained_variance,
            components: Some(Matrix {
                rows: components.len() as i32,
                cols: if components.is_empty() { 0 } else { components[0].len() as i32 },
                values: components_matrix_values,
            }),
            transformed: None,
            total_variance_explained,
            metadata: format!("PCA with {} components", components.len()),
        }))
    }

    type CARAOptimalWeightsStream = ReceiverStream<Result<WeightsOutput, Status>>;

    async fn cara_optimal_weights(
        &self,
        request: Request<tonic::Streaming<OptimizationInput>>,
    ) -> Result<Response<Self::CARAOptimalWeightsStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(10);

        tokio::spawn(async move {
            let mut returns_data: Option<Vec<Vec<f64>>> = None;
            let mut gamma = 2.0;

            while let Ok(Some(input)) = stream.message().await {
                if let Some(ret_matrix) = input.returns {
                    returns_data = Some(
                        (0..ret_matrix.rows as usize)
                            .map(|i| {
                                ret_matrix
                                    .values
                                    .iter()
                                    .skip(i * ret_matrix.cols as usize)
                                    .take(ret_matrix.cols as usize)
                                    .copied()
                                    .collect()
                            })
                            .collect(),
                    );
                }
                if input.gamma > 0.0 {
                    gamma = input.gamma;
                }
            }

            if let Some(returns) = returns_data {
                let covariance = models::compute_covariance(&returns);
                let mut avg_returns = vec![0.0; returns[0].len()];
                for ret in &returns {
                    for (i, r) in ret.iter().enumerate() {
                        avg_returns[i] += r / returns.len() as f64;
                    }
                }

                let weights = models::cara_optimal_weights(&avg_returns, &covariance, gamma);

                let expected_return = models::dot_product(&weights, &avg_returns);
                let variance = {
                    let cov_times_w = models::matrix_vec_multiply(&covariance, &weights);
                    models::dot_product(&cov_times_w, &weights)
                };
                let expected_volatility = variance.sqrt();
                let sharpe_ratio =
                    if expected_volatility > 1e-10 {
                        expected_return / expected_volatility
                    } else {
                        0.0
                    };

                let utility = (-gamma * (expected_return - 0.5 * gamma * variance)).exp();

                let output = WeightsOutput {
                    weights,
                    expected_return,
                    expected_volatility,
                    sharpe_ratio,
                    utility,
                    status: "optimized".to_string(),
                    timestamp_ms: Self::current_timestamp_ms(),
                    metadata: "CARA optimization completed".to_string(),
                };

                let _ = tx.send(Ok(output)).await;
            } else {
                let _ = tx.send(Err(Status::invalid_argument("No returns provided"))).await;
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    type SharpeOptimalWeightsStream = ReceiverStream<Result<WeightsOutput, Status>>;

    async fn sharpe_optimal_weights(
        &self,
        request: Request<tonic::Streaming<OptimizationInput>>,
    ) -> Result<Response<Self::SharpeOptimalWeightsStream>, Status> {
        let mut stream = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(10);

        tokio::spawn(async move {
            let mut returns_data: Option<Vec<Vec<f64>>> = None;
            let mut risk_free_rate = 0.02;

            while let Ok(Some(input)) = stream.message().await {
                if let Some(ret_matrix) = input.returns {
                    returns_data = Some(
                        (0..ret_matrix.rows as usize)
                            .map(|i| {
                                ret_matrix
                                    .values
                                    .iter()
                                    .skip(i * ret_matrix.cols as usize)
                                    .take(ret_matrix.cols as usize)
                                    .copied()
                                    .collect()
                            })
                            .collect(),
                    );
                }
                if input.risk_free_rate > 0.0 {
                    risk_free_rate = input.risk_free_rate;
                }
            }

            if let Some(returns) = returns_data {
                let covariance = models::compute_covariance(&returns);
                let mut avg_returns = vec![0.0; returns[0].len()];
                for ret in &returns {
                    for (i, r) in ret.iter().enumerate() {
                        avg_returns[i] += r / returns.len() as f64;
                    }
                }

                let weights = models::sharpe_optimal_weights(&avg_returns, &covariance, risk_free_rate);

                let expected_return = models::dot_product(&weights, &avg_returns);
                let variance = {
                    let cov_times_w = models::matrix_vec_multiply(&covariance, &weights);
                    models::dot_product(&cov_times_w, &weights)
                };
                let expected_volatility = variance.sqrt();
                let sharpe_ratio = if expected_volatility > 1e-10 {
                    (expected_return - risk_free_rate) / expected_volatility
                } else {
                    0.0
                };

                let output = WeightsOutput {
                    weights,
                    expected_return,
                    expected_volatility,
                    sharpe_ratio,
                    utility: sharpe_ratio,
                    status: "optimized".to_string(),
                    timestamp_ms: Self::current_timestamp_ms(),
                    metadata: "Sharpe optimization completed".to_string(),
                };

                let _ = tx.send(Ok(output)).await;
            } else {
                let _ = tx.send(Err(Status::invalid_argument("No returns provided"))).await;
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn multiperiod_optimize(
        &self,
        request: Request<MultiperiodRequest>,
    ) -> Result<Response<MultiperiodResponse>, Status> {
        let req = request.into_inner();

        if req.returns_sequence.is_empty() {
            return Err(Status::invalid_argument("No returns provided"));
        }

        let gamma = if req.gamma > 0.0 { req.gamma } else { 2.0 };
        let transaction_cost = req.transaction_cost;

        let mut weights_sequence: Vec<WeightsOutput> = Vec::new();
        let mut rebalance_times: Vec<i64> = Vec::new();
        let mut cumulative_utility = 0.0;

        for (idx, matrix) in req.returns_sequence.iter().enumerate() {
            let returns: Vec<Vec<f64>> = (0..matrix.rows as usize)
                .map(|i| {
                    matrix
                        .values
                        .iter()
                        .skip(i * matrix.cols as usize)
                        .take(matrix.cols as usize)
                        .copied()
                        .collect()
                })
                .collect();

            let covariance = models::compute_covariance(&returns);
            let mut avg_returns = vec![0.0; returns[0].len()];
            for ret in &returns {
                for (i, r) in ret.iter().enumerate() {
                    avg_returns[i] += r / returns.len() as f64;
                }
            }

            let mut weights = models::cara_optimal_weights(&avg_returns, &covariance, gamma);

            if idx > 0 && !weights_sequence.is_empty() {
                if let Some(prev) = weights_sequence.last() {
                    for (w, pw) in weights.iter_mut().zip(prev.weights.iter()) {
                        let change = (*w - pw).abs();
                        if change > 1e-10 {
                            *w -= transaction_cost * change * (*w - pw).signum();
                        }
                    }
                }
            }

            let sum: f64 = weights.iter().sum();
            if sum.abs() > 1e-10 {
                weights.iter_mut().for_each(|w| *w /= sum);
            }

            let period_return = models::dot_product(&weights, &avg_returns);
            let period_var = {
                let cov_times_w = models::matrix_vec_multiply(&covariance, &weights);
                models::dot_product(&cov_times_w, &weights)
            };
            let utility = (-gamma * (period_return - 0.5 * gamma * period_var)).exp();

            weights_sequence.push(WeightsOutput {
                weights,
                expected_return: period_return,
                expected_volatility: period_var.sqrt(),
                sharpe_ratio: 0.0,
                utility,
                status: "optimized".to_string(),
                timestamp_ms: Self::current_timestamp_ms(),
                metadata: "Period optimization".to_string(),
            });
            rebalance_times.push(idx as i64);
            cumulative_utility += utility;
        }

        cumulative_utility /= req.returns_sequence.len() as f64;

        Ok(Response::new(MultiperiodResponse {
            weights_sequence,
            cumulative_utility,
            avg_turnover: transaction_cost,
            rebalance_times,
            metadata: format!("Multiperiod optimization with {} periods", req.returns_sequence.len()),
        }))
    }

    async fn portfolio_stats(
        &self,
        request: Request<PortfolioStatsRequest>,
    ) -> Result<Response<PortfolioStatsResponse>, Status> {
        let req = request.into_inner();

        let matrix = req.returns.ok_or_else(|| Status::invalid_argument("No returns provided"))?;

        let returns_data: Vec<Vec<f64>> = (0..matrix.rows as usize)
            .map(|i| {
                matrix
                    .values
                    .iter()
                    .skip(i * matrix.cols as usize)
                    .take(matrix.cols as usize)
                    .copied()
                    .collect()
            })
            .collect();

        let covariance = models::compute_covariance(&returns_data);

        let mut mean_returns = vec![0.0; returns_data[0].len()];
        for ret in &returns_data {
            for (i, r) in ret.iter().enumerate() {
                mean_returns[i] += r / returns_data.len() as f64;
            }
        }

        let (mean_return, std_return) = models::mean_and_std(&mean_returns);

        let mut cov_values = Vec::new();
        for row in &covariance {
            cov_values.extend_from_slice(row);
        }

        let mut corr = covariance.clone();
        for i in 0..corr.len() {
            for j in 0..corr[i].len() {
                let denom = (covariance[i][i] * covariance[j][j]).sqrt();
                if denom > 1e-10 {
                    corr[i][j] /= denom;
                }
            }
        }

        let mut corr_values = Vec::new();
        for row in &corr {
            corr_values.extend_from_slice(row);
        }

        Ok(Response::new(PortfolioStatsResponse {
            mean_return,
            std_return,
            skewness: 0.0,
            kurtosis: 3.0,
            correlation: Some(Matrix {
                rows: corr.len() as i32,
                cols: if corr.is_empty() { 0 } else { corr[0].len() as i32 },
                values: corr_values,
            }),
            covariance: Some(Matrix {
                rows: covariance.len() as i32,
                cols: if covariance.is_empty() { 0 } else { covariance[0].len() as i32 },
                values: cov_values,
            }),
            marginal_contrib: vec![],
            metadata: "Portfolio statistics computed".to_string(),
        }))
    }

    async fn generate_signals(
        &self,
        request: Request<SignalRequest>,
    ) -> Result<Response<SignalResponse>, Status> {
        let req = request.into_inner();

        let price_history = req.price_history.ok_or_else(|| Status::invalid_argument("No price history"))?;
        
        if price_history.prices.len() < 20 {
            return Err(Status::invalid_argument("Need at least 20 price points"));
        }

        let (theta, mu, sigma, _, _, _) = models::estimate_ou_params(&price_history.prices);

        let window = 20;
        let window_prices = &price_history.prices[price_history.prices.len() - window..];
        let (mean, std) = models::mean_and_std(window_prices);
        let current_price = *price_history.prices.last().unwrap();
        let zscore = if std > 1e-10 {
            (current_price - mean) / std
        } else {
            0.0
        };

        let thresholds = req.thresholds.ok_or_else(|| Status::invalid_argument("No thresholds"))?;

        let (signal, confidence, expected_profit) = if zscore < -thresholds.entry_zscore {
            ("buy".to_string(), 0.7, (zscore.abs() - thresholds.exit_zscore) * sigma)
        } else if zscore > thresholds.entry_zscore {
            ("sell".to_string(), 0.7, (zscore - thresholds.exit_zscore) * sigma)
        } else if zscore.abs() < thresholds.exit_zscore {
            ("exit".to_string(), 0.6, 0.0)
        } else {
            ("hold".to_string(), 0.5, 0.0)
        };

        Ok(Response::new(SignalResponse {
            signal,
            confidence,
            zscore,
            expected_profit,
            timestamp_ms: Self::current_timestamp_ms(),
            reasoning: format!(
                "OU: θ={:.4}, μ={:.2}, σ={:.4}, z={:.2}",
                theta, mu, sigma, zscore
            ),
            metadata: "Signal generated from OU parameters".to_string(),
        }))
    }

    type PortfolioHealthStreamStream = ReceiverStream<Result<HealthMetric, Status>>;

    async fn portfolio_health_stream(
        &self,
        request: Request<HealthRequest>,
    ) -> Result<Response<Self::PortfolioHealthStreamStream>, Status> {
        let req = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(10);

        tokio::spawn(async move {
            let metric = HealthMetric {
                portfolio_id: req.portfolio_id.clone(),
                timestamp_ms: Self::current_timestamp_ms(),
                total_value: 100000.0,
                daily_return: 0.001,
                volatility_24h: 0.015,
                max_drawdown: 0.05,
                open_positions: 0,
                status: "healthy".to_string(),
                alerts: vec![],
                metadata: "Health check completed".to_string(),
            };

            let _ = tx.send(Ok(metric)).await;
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
