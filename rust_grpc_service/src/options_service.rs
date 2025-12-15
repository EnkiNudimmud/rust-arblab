use tonic::{Request, Response, Status};
use crate::hft::options::options_service_server::OptionsService;
use crate::hft::options::{
    OptionParams, Greeks, HedgingRequest, HedgingResponse,
};
use rust_core::options::{BlackScholesOption, DeltaHedgingStrategy};

#[derive(Debug, Default)]
pub struct MyOptionsService;

// Helper to convert proto params to core option
fn to_core_option(p: &OptionParams) -> BlackScholesOption {
    BlackScholesOption::new(
        p.spot,
        p.strike,
        p.time_to_expiry,
        p.risk_free_rate,
        p.dividend_yield,
        p.volatility,
        p.is_call,
    )
}

#[tonic::async_trait]
impl OptionsService for MyOptionsService {
    async fn calculate_greeks(
        &self,
        request: Request<OptionParams>,
    ) -> Result<Response<Greeks>, Status> {
        let req = request.into_inner();
        let opt = to_core_option(&req);
        let g = opt.greeks();

        Ok(Response::new(Greeks {
            price: opt.price(),
            delta: g.delta,
            gamma: g.gamma,
            theta: g.theta,
            vega: g.vega,
            rho: g.rho,
        }))
    }

    async fn simulate_hedging(
        &self,
        request: Request<HedgingRequest>,
    ) -> Result<Response<HedgingResponse>, Status> {
        let req = request.into_inner();
        let opt_params = req.option.ok_or(Status::invalid_argument("Missing option params"))?;
        
        let strategy = DeltaHedgingStrategy::new(
            opt_params.spot,
            opt_params.strike,
            opt_params.time_to_expiry,
            opt_params.risk_free_rate,
            opt_params.dividend_yield,
            opt_params.volatility, // implied
            req.actual_vol,
            req.hedging_vol,
            opt_params.is_call,
        );

        let (exp_profit, profit_std) = strategy.expected_profit_simulation(
            req.num_simulations as usize,
            req.num_steps as usize,
            req.drift,
        );

        let guaranteed = strategy.guaranteed_profit_actual_hedge();

        Ok(Response::new(HedgingResponse {
            expected_profit: exp_profit,
            profit_std,
            guaranteed_profit: guaranteed,
        }))
    }
}
