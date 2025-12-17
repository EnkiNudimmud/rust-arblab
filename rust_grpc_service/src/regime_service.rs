use tonic::{Request, Response, Status};
use crate::regime::regime_service_server::RegimeService;
use crate::regime::{
    CalibrationRequest, PortfolioConfig, PortfolioResult, SimulationRequest,
    SimulationResponse, RegimeEstimateRequest, RegimeEstimateResponse,
};
use rust_core::regime_portfolio::{
    RegimeSwitchingPortfolio, PortfolioConfig as CoreConfig,
    calibrate_from_data,
};

#[derive(Debug, Default)]
pub struct MyRegimeService;

#[tonic::async_trait]
impl RegimeService for MyRegimeService {
    async fn calibrate(
        &self,
        request: Request<CalibrationRequest>,
    ) -> Result<Response<PortfolioConfig>, Status> {
        let req = request.into_inner();
        let returns = req.returns;
        let regimes: Option<Vec<usize>> = if req.known_regimes.is_empty() {
            None
        } else {
            Some(req.known_regimes.iter().map(|&r| r as usize).collect())
        };

        // Call rust_core logic
        let config = calibrate_from_data(&returns, regimes.as_deref());

        // Convert CoreConfig to ProtoConfig
        Ok(Response::new(PortfolioConfig {
            gamma: config.gamma,
            time_horizon: config.time_horizon,
            rho: config.rho,
            transaction_cost: config.transaction_cost,
            r_bull: config.r_bull,
            mu_bull: config.mu_bull,
            sigma_bull: config.sigma_bull,
            lambda_bull: config.lambda_bull,
            jump_mean_bull: config.jump_mean_bull,
            jump_std_bull: config.jump_std_bull,
            r_normal: config.r_normal,
            mu_normal: config.mu_normal,
            sigma_normal: config.sigma_normal,
            lambda_normal: config.lambda_normal,
            jump_mean_normal: config.jump_mean_normal,
            jump_std_normal: config.jump_std_normal,
            r_bear: config.r_bear,
            mu_bear: config.mu_bear,
            sigma_bear: config.sigma_bear,
            lambda_bear: config.lambda_bear,
            jump_mean_bear: config.jump_mean_bear,
            jump_std_bear: config.jump_std_bear,
            q_bull_normal: config.q_bull_normal,
            q_bull_bear: config.q_bull_bear,
            q_normal_bull: config.q_normal_bull,
            q_normal_bear: config.q_normal_bear,
            q_bear_bull: config.q_bear_bull,
            q_bear_normal: config.q_bear_normal,
        }))
    }

    async fn optimize(
        &self,
        request: Request<PortfolioConfig>,
    ) -> Result<Response<PortfolioResult>, Status> {
        let proto_config = request.into_inner();
        
        let config = CoreConfig {
            gamma: proto_config.gamma,
            time_horizon: proto_config.time_horizon,
            rho: proto_config.rho,
            transaction_cost: proto_config.transaction_cost,
            r_bull: proto_config.r_bull,
            mu_bull: proto_config.mu_bull,
            sigma_bull: proto_config.sigma_bull,
            lambda_bull: proto_config.lambda_bull,
            jump_mean_bull: proto_config.jump_mean_bull,
            jump_std_bull: proto_config.jump_std_bull,
            r_normal: proto_config.r_normal,
            mu_normal: proto_config.mu_normal,
            sigma_normal: proto_config.sigma_normal,
            lambda_normal: proto_config.lambda_normal,
            jump_mean_normal: proto_config.jump_mean_normal,
            jump_std_normal: proto_config.jump_std_normal,
            r_bear: proto_config.r_bear,
            mu_bear: proto_config.mu_bear,
            sigma_bear: proto_config.sigma_bear,
            lambda_bear: proto_config.lambda_bear,
            jump_mean_bear: proto_config.jump_mean_bear,
            jump_std_bear: proto_config.jump_std_bear,
            q_bull_normal: proto_config.q_bull_normal,
            q_bull_bear: proto_config.q_bull_bear,
            q_normal_bull: proto_config.q_normal_bull,
            q_normal_bear: proto_config.q_normal_bear,
            q_bear_bull: proto_config.q_bear_bull,
            q_bear_normal: proto_config.q_bear_normal,
        };

        let portfolio = RegimeSwitchingPortfolio::new(config.clone());
        
        match portfolio.optimize() {
            Ok(result) => {
                Ok(Response::new(PortfolioResult {
                    wealth: result.wealth.to_vec(),
                    values: result.values.iter().flat_map(|v| v.to_vec()).collect(),
                    portfolio_weights: result.portfolio_weights.iter().flat_map(|w| w.to_vec()).collect(),
                    stationary_probs: result.stationary_probs.to_vec(),
                    expected_weight: result.expected_weight,
                    initial_value: result.initial_value,
                    iterations: result.iterations as i32,
                    residual: result.residual,
                    config: Some(proto_config),
                }))
            }
            Err(e) => Err(Status::internal(format!("Optimization failed: {}", e))),
        }
    }

    async fn simulate_path(
        &self,
        _request: Request<SimulationRequest>,
    ) -> Result<Response<SimulationResponse>, Status> {
        // Mock implementation for now as we haven't exposed simulation in rust_core extensively
        // Or reconstruct objects to call rust_core simulation
        // For brevity, using placeholder or simplistic logic if complex reconstruction needed
        Err(Status::unimplemented("Simulation not yet fully ported"))
    }

    async fn estimate_regime(
        &self,
        request: Request<RegimeEstimateRequest>,
    ) -> Result<Response<RegimeEstimateResponse>, Status> {
        let req = request.into_inner();
        let regime = RegimeSwitchingPortfolio::estimate_regime(&req.returns);
        
        let regime_str = match regime {
            rust_core::regime_portfolio::MarketRegime::Bull => "Bull",
            rust_core::regime_portfolio::MarketRegime::Normal => "Normal",
            rust_core::regime_portfolio::MarketRegime::Bear => "Bear",
        };

        Ok(Response::new(RegimeEstimateResponse {
            regime: regime_str.to_string(),
        }))
    }
}
