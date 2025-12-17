use tonic::{Request, Response, Status};
use crate::drift::drift_service_server::DriftService;
use crate::drift::{
    ChoiceRequest, PortfolioResult, LiquidationRequest, LiquidationResult,
    TransitionRequest, TransitionResult, RiskRequest, RiskMetric,
};
use rust_core::portfolio_drift_uncertainty::{
    portfolio_choice_robust, optimal_liquidation, portfolio_transition,
    value_at_risk_robust, expected_shortfall_robust,
};

#[derive(Debug, Default)]
pub struct MyDriftService;

#[tonic::async_trait]
impl DriftService for MyDriftService {
    async fn portfolio_choice(
        &self,
        request: Request<ChoiceRequest>,
    ) -> Result<Response<PortfolioResult>, Status> {
        let req = request.into_inner();
        let n_assets = req.expected_returns.len();
        
        let result = portfolio_choice_robust(
            &req.expected_returns,
            &req.covariance,
            req.risk_aversion,
            req.drift_uncertainty,
            n_assets,
        );

        Ok(Response::new(PortfolioResult {
            weights: result.weights,
            expected_return: result.expected_return,
            worst_case_return: result.worst_case_return,
            variance: result.variance,
            utility: result.utility,
        }))
    }

    async fn optimal_liquidation(
        &self,
        request: Request<LiquidationRequest>,
    ) -> Result<Response<LiquidationResult>, Status> {
        let req = request.into_inner();
        
        let result = optimal_liquidation(
            req.initial_position,
            req.time_horizon,
            req.n_steps as usize,
            req.drift,
            req.drift_uncertainty,
            req.volatility,
            req.temporary_impact,
            req.permanent_impact,
            req.risk_aversion,
        );

        Ok(Response::new(LiquidationResult {
            trading_schedule: result.trading_schedule,
            trading_rates: result.trading_rates,
            expected_cost: result.expected_cost,
            worst_case_cost: result.worst_case_cost,
            times: result.times,
        }))
    }

    async fn transition(
        &self,
        request: Request<TransitionRequest>,
    ) -> Result<Response<TransitionResult>, Status> {
        let req = request.into_inner();
        
        let result = portfolio_transition(
            &req.initial_weights,
            &req.target_weights,
            req.time_horizon,
            req.n_steps as usize,
            &req.expected_returns,
            &req.covariance,
            req.drift_uncertainty,
            req.transaction_cost,
            req.risk_aversion,
        );

        // Map nested vectors to ArrayWrapper
        let trajectory = result.trajectory.into_iter()
            .map(|v| crate::drift::ArrayWrapper { values: v })
            .collect();

        let trading_rates = result.trading_rates.into_iter()
            .map(|v| crate::drift::ArrayWrapper { values: v })
            .collect();

        Ok(Response::new(TransitionResult {
            trajectory,
            trading_rates,
            expected_cost: result.expected_cost,
            worst_case_cost: result.worst_case_cost,
            times: result.times,
        }))
    }

    async fn calculate_va_r(
        &self,
        request: Request<RiskRequest>,
    ) -> Result<Response<RiskMetric>, Status> {
        let req = request.into_inner();
        
        let val = value_at_risk_robust(
            &req.weights,
            &req.expected_returns,
            &req.covariance,
            req.drift_uncertainty,
            req.confidence_level,
            req.time_horizon,
        );

        Ok(Response::new(RiskMetric { value: val }))
    }

    async fn calculate_c_va_r(
        &self,
        request: Request<RiskRequest>,
    ) -> Result<Response<RiskMetric>, Status> {
        let req = request.into_inner();
        
        let val = expected_shortfall_robust(
            &req.weights,
            &req.expected_returns,
            &req.covariance,
            req.drift_uncertainty,
            req.confidence_level,
            req.time_horizon,
        );

        Ok(Response::new(RiskMetric { value: val }))
    }
}
