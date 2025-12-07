"""
Optimal Switching for Pairs Trading - Viscosity Solutions Approach

Implementation based on:
"Optimal switching for pairs trading rule: a viscosity solutions approach"

This module implements the optimal switching problem for pairs trading using:
1. Ornstein-Uhlenbeck (OU) process for the spread
2. Hamilton-Jacobi-Bellman (HJB) equations
3. Viscosity solutions for value functions
4. Four trading states: Open, Buy, Sell, Close
5. Transaction costs and optimal switching boundaries

Key Features:
- Cointegration testing with Engle-Granger and Johansen methods
- OU parameter estimation (mean-reversion speed, long-term mean, volatility)
- Finite difference methods for PDE solving
- Optimal switching boundaries computation
- Value function characterization
- Integration with sparse mean-reversion portfolios
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from scipy import optimize, linalg
from scipy.stats import norm
import warnings

# Try to import statsmodels for cointegration tests
try:
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available - cointegration tests disabled")

# Try to import optimizr for optimization
try:
    import optimizr
    OPTIMIZR_AVAILABLE = True
except ImportError:
    OPTIMIZR_AVAILABLE = False
    warnings.warn("optimizr not available - using scipy for optimization")


@dataclass
class OUParameters:
    """Ornstein-Uhlenbeck process parameters
    
    dX_t = κ(θ - X_t)dt + σ dW_t
    
    where:
    - κ (kappa): mean-reversion speed
    - θ (theta): long-term mean
    - σ (sigma): volatility
    """
    kappa: float  # Mean-reversion speed
    theta: float  # Long-term mean
    sigma: float  # Volatility
    half_life: float  # Half-life in days
    
    def __str__(self):
        return (f"OU Parameters:\n"
                f"  κ (mean-reversion speed): {self.kappa:.4f}\n"
                f"  θ (long-term mean): {self.theta:.4f}\n"
                f"  σ (volatility): {self.sigma:.4f}\n"
                f"  Half-life: {self.half_life:.2f} days")


@dataclass
class CointegrationResult:
    """Result from cointegration test"""
    is_cointegrated: bool
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    hedge_ratio: float
    method: str  # 'engle-granger' or 'johansen'
    
    def summary(self) -> str:
        """Get summary statistics"""
        cv_str = ", ".join([f"{k}: {v:.4f}" for k, v in self.critical_values.items()])
        return (
            f"Cointegration Test ({self.method}):\n"
            f"  Test Statistic: {self.test_statistic:.4f}\n"
            f"  P-value: {self.p_value:.4f}\n"
            f"  Critical Values: {cv_str}\n"
            f"  Hedge Ratio: {self.hedge_ratio:.4f}\n"
            f"  Result: {'COINTEGRATED ✓' if self.is_cointegrated else 'NOT COINTEGRATED ✗'}"
        )


@dataclass
class SwitchingBoundaries:
    """Optimal switching boundaries for pairs trading
    
    Four trading states:
    1. Open (no position)
    2. Buy (long spread: long asset 1, short asset 2)
    3. Sell (short spread: short asset 1, long asset 2)
    4. Close (exiting position)
    """
    open_to_buy: float  # Boundary: Open -> Buy (spread too low)
    open_to_sell: float  # Boundary: Open -> Sell (spread too high)
    buy_to_close: float  # Boundary: Buy -> Close (take profit/stop loss)
    sell_to_close: float  # Boundary: Sell -> Close (take profit/stop loss)
    
    # Value functions at boundaries
    V_open: np.ndarray  # Value function in open state
    V_buy: np.ndarray  # Value function in buy state
    V_sell: np.ndarray  # Value function in sell state
    
    # Grid for value functions
    spread_grid: np.ndarray
    
    def __str__(self):
        return (f"Optimal Switching Boundaries:\n"
                f"  Open -> Buy:  {self.open_to_buy:.4f}\n"
                f"  Open -> Sell: {self.open_to_sell:.4f}\n"
                f"  Buy -> Close: {self.buy_to_close:.4f}\n"
                f"  Sell -> Close: {self.sell_to_close:.4f}")


def engle_granger_cointegration(
    y: pd.Series,
    x: pd.Series,
    significance_level: float = 0.05
) -> CointegrationResult:
    """
    Engle-Granger two-step cointegration test
    
    Step 1: Estimate cointegrating relationship y_t = α + β x_t + ε_t
    Step 2: Test if residuals are stationary (ADF test)
    
    Parameters:
    -----------
    y : pd.Series
        Dependent variable (asset 1 prices)
    x : pd.Series
        Independent variable (asset 2 prices)
    significance_level : float
        Significance level for hypothesis test
        
    Returns:
    --------
    CointegrationResult
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for cointegration testing")
    
    # Align series
    common_idx = y.index.intersection(x.index)
    y_aligned = y.loc[common_idx]
    x_aligned = x.loc[common_idx]
    
    # Step 1: OLS regression to estimate hedge ratio
    from statsmodels.regression.linear_model import OLS
    import statsmodels.api as sm
    
    X = sm.add_constant(x_aligned)
    model = OLS(y_aligned, X).fit()
    hedge_ratio = model.params[1]
    
    # Compute spread (residuals)
    spread = y_aligned - hedge_ratio * x_aligned
    
    # Step 2: ADF test on residuals
    # Ensure 1D array for adfuller
    spread = np.asarray(spread).ravel()
    adf_result = adfuller(spread, maxlag=1, regression='c')
    
    test_stat = float(adf_result[0])
    p_value = float(adf_result[1])
    # Critical values are in index 4, which is a dict
    cv_dict = adf_result[4] if len(adf_result) > 4 else {}
    critical_values = {
        '1%': float(cv_dict.get('1%', 0.0)),
        '5%': float(cv_dict.get('5%', 0.0)),
        '10%': float(cv_dict.get('10%', 0.0))
    }
    
    # Determine if cointegrated
    is_cointegrated = bool(p_value < significance_level)
    
    return CointegrationResult(
        is_cointegrated=is_cointegrated,
        test_statistic=test_stat,
        p_value=p_value,
        critical_values=critical_values,
        hedge_ratio=hedge_ratio,
        method='engle-granger'
    )


def johansen_cointegration(
    prices: pd.DataFrame,
    significance_level: float = 0.05
) -> List[CointegrationResult]:
    """
    Johansen cointegration test for multiple assets
    
    Tests for cointegration among multiple time series using
    the Johansen procedure (trace test and max eigenvalue test)
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price series for multiple assets (columns = assets)
    significance_level : float
        Significance level (0.01, 0.05, or 0.10)
        
    Returns:
    --------
    List[CointegrationResult]
        Results for each cointegrating relationship
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for Johansen test")
    
    # Map significance level to index
    sig_map = {0.01: 2, 0.05: 1, 0.10: 0}
    sig_idx = sig_map.get(significance_level, 1)
    
    # Run Johansen test
    result = coint_johansen(prices, det_order=0, k_ar_diff=1)
    
    # Parse results
    cointegration_results = []
    
    # Trace test
    for i in range(len(result.lr1)):
        test_stat = result.lr1[i]
        critical_val = result.cvt[i, sig_idx]
        
        is_cointegrated = test_stat > critical_val
        
        # Get eigenvector (cointegrating vector)
        eigenvector = result.evec[:, i]
        
        coint_result = CointegrationResult(
            is_cointegrated=is_cointegrated,
            test_statistic=test_stat,
            p_value=0.0,  # Johansen doesn't provide p-values
            critical_values={
                '10%': result.cvt[i, 0],
                '5%': result.cvt[i, 1],
                '1%': result.cvt[i, 2]
            },
            hedge_ratio=eigenvector[1] / eigenvector[0] if len(eigenvector) >= 2 else 1.0,
            method='johansen-trace'
        )
        
        cointegration_results.append(coint_result)
    
    return cointegration_results


def estimate_ou_parameters(spread: pd.Series, dt: float = 1.0) -> OUParameters:
    """
    Estimate Ornstein-Uhlenbeck parameters from spread time series
    
    Uses maximum likelihood estimation (MLE) for discretized OU process:
    X_{t+Δt} = X_t + κ(θ - X_t)Δt + σ√Δt ε_t
    
    where ε_t ~ N(0,1)
    
    Parameters:
    -----------
    spread : pd.Series
        Spread time series
    dt : float
        Time step (default: 1 day)
        
    Returns:
    --------
    OUParameters
    """
    # Convert to numpy array
    X = np.asarray(spread.values, dtype=np.float64)
    n = len(X)
    
    # Compute differences
    dX = np.diff(X)
    X_lag = X[:-1]
    
    # MLE for OU parameters
    # From discretized OU: dX = κ(θ - X)dt + σ√dt ε
    
    # Step 1: Estimate θ (long-term mean) as sample mean
    theta_hat = float(np.mean(X))
    
    # Step 2: Estimate κ and σ using regression
    # dX/dt ≈ κ(θ - X) + σ/√dt ε
    # Let Y = dX/dt, A = X - θ
    # Y = -κA + σ/√dt ε
    
    Y = dX / dt
    A = X_lag - theta_hat
    
    # OLS: Y = -κA + error
    kappa_hat = -np.sum(Y * A) / np.sum(A * A)
    
    # Ensure positive mean-reversion
    kappa_hat = abs(kappa_hat)
    
    # Step 3: Estimate σ from residuals
    residuals = Y + kappa_hat * A
    sigma_hat = np.std(residuals) * np.sqrt(dt)
    
    # Compute half-life: t_half = ln(2) / κ
    half_life = np.log(2) / kappa_hat if kappa_hat > 0 else np.inf
    
    return OUParameters(
        kappa=kappa_hat,
        theta=theta_hat,
        sigma=sigma_hat,
        half_life=half_life
    )


def solve_hjb_pde(
    ou_params: OUParameters,
    transaction_cost: float,
    discount_rate: float,
    spread_min: float,
    spread_max: float,
    n_points: int = 500,
    max_iterations: int = 10000,
    tolerance: float = 1e-6
) -> SwitchingBoundaries:
    """
    Solve Hamilton-Jacobi-Bellman equations using finite difference method
    
    The value functions satisfy:
    
    1. Open state: ρV^O = max{V^O, V^B - c, V^S - c}
    2. Buy state: ρV^B = κ(θ - x)V^B_x + σ²/2 V^B_xx + x + max{V^B, V^O - c}
    3. Sell state: ρV^S = κ(θ - x)V^S_x + σ²/2 V^S_xx - x + max{V^S, V^O - c}
    
    where:
    - ρ: discount rate
    - κ, θ, σ: OU parameters
    - c: transaction cost
    - x: spread value
    
    Parameters:
    -----------
    ou_params : OUParameters
        OU process parameters
    transaction_cost : float
        Cost per transaction (as fraction of spread)
    discount_rate : float
        Discount rate ρ
    spread_min : float
        Minimum spread value for grid
    spread_max : float
        Maximum spread value for grid
    n_points : int
        Number of grid points
    max_iterations : int
        Maximum iterations for convergence
    tolerance : float
        Convergence tolerance
        
    Returns:
    --------
    SwitchingBoundaries
    """
    # Create spread grid
    x_grid = np.linspace(spread_min, spread_max, n_points)
    dx = x_grid[1] - x_grid[0]
    
    # Initialize value functions
    V_open = np.zeros(n_points)
    V_buy = np.zeros(n_points)
    V_sell = np.zeros(n_points)
    
    # Parameters
    kappa = ou_params.kappa
    theta = ou_params.theta
    sigma = ou_params.sigma
    
    # Time step for explicit scheme (must satisfy CFL condition)
    dt = min(0.5 * dx**2 / (sigma**2), 0.5 / kappa)
    
    # Coefficients for finite difference
    # Using upwind scheme for drift term, central for diffusion
    
    for iteration in range(max_iterations):
        V_open_old = V_open.copy()
        V_buy_old = V_buy.copy()
        V_sell_old = V_sell.copy()
        
        for i in range(1, n_points - 1):
            x = x_grid[i]
            
            # Drift term: κ(θ - x)
            drift = kappa * (theta - x)
            
            # Diffusion term: σ²/2
            diffusion = 0.5 * sigma**2
            
            # Finite difference approximations
            # First derivative (upwind)
            if drift > 0:
                V_buy_x = (V_buy_old[i] - V_buy_old[i-1]) / dx
                V_sell_x = (V_sell_old[i] - V_sell_old[i-1]) / dx
            else:
                V_buy_x = (V_buy_old[i+1] - V_buy_old[i]) / dx
                V_sell_x = (V_sell_old[i+1] - V_sell_old[i]) / dx
            
            # Second derivative (central)
            V_buy_xx = (V_buy_old[i+1] - 2*V_buy_old[i] + V_buy_old[i-1]) / (dx**2)
            V_sell_xx = (V_sell_old[i+1] - 2*V_sell_old[i] + V_sell_old[i-1]) / (dx**2)
            
            # Update Buy state value function
            # ρV^B = κ(θ - x)V^B_x + σ²/2 V^B_xx + x + max{V^B, V^O - c}
            rhs_buy = drift * V_buy_x + diffusion * V_buy_xx + x
            V_buy[i] = V_buy_old[i] + dt * (rhs_buy - discount_rate * V_buy_old[i])
            V_buy[i] = max(V_buy[i], V_open_old[i] - transaction_cost)
            
            # Update Sell state value function
            # ρV^S = κ(θ - x)V^S_x + σ²/2 V^S_xx - x + max{V^S, V^O - c}
            rhs_sell = drift * V_sell_x + diffusion * V_sell_xx - x
            V_sell[i] = V_sell_old[i] + dt * (rhs_sell - discount_rate * V_sell_old[i])
            V_sell[i] = max(V_sell[i], V_open_old[i] - transaction_cost)
            
            # Update Open state value function
            # ρV^O = max{V^O, V^B - c, V^S - c}
            V_open[i] = max(V_open_old[i], V_buy[i] - transaction_cost, V_sell[i] - transaction_cost)
        
        # Boundary conditions (absorbing at boundaries)
        V_open[0] = V_open[1]
        V_open[-1] = V_open[-2]
        V_buy[0] = V_buy[1]
        V_buy[-1] = V_buy[-2]
        V_sell[0] = V_sell[1]
        V_sell[-1] = V_sell[-2]
        
        # Check convergence
        diff_open = np.max(np.abs(V_open - V_open_old))
        diff_buy = np.max(np.abs(V_buy - V_buy_old))
        diff_sell = np.max(np.abs(V_sell - V_sell_old))
        max_diff = max(diff_open, diff_buy, diff_sell)
        
        if max_diff < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    # Extract switching boundaries
    # Open -> Buy: where V^B - c > V^O (spread is low, expect to rise)
    # Open -> Sell: where V^S - c > V^O (spread is high, expect to fall)
    # Buy -> Close: where V^O - c > V^B (exit long position)
    # Sell -> Close: where V^O - c > V^S (exit short position)
    
    # Find boundaries by looking for where conditions are satisfied
    open_to_buy_mask = (V_buy - transaction_cost) > V_open
    open_to_sell_mask = (V_sell - transaction_cost) > V_open
    buy_to_close_mask = (V_open - transaction_cost) > V_buy
    sell_to_close_mask = (V_open - transaction_cost) > V_sell
    
    # Get indices where conditions are first/last met
    open_to_buy_indices = np.where(open_to_buy_mask)[0]
    open_to_sell_indices = np.where(open_to_sell_mask)[0]
    buy_to_close_indices = np.where(buy_to_close_mask)[0]
    sell_to_close_indices = np.where(sell_to_close_mask)[0]
    
    # Extract boundaries (use theta as default if no crossing found)
    open_to_buy = float(x_grid[open_to_buy_indices[-1]]) if len(open_to_buy_indices) > 0 else float(theta - sigma)
    open_to_sell = float(x_grid[open_to_sell_indices[0]]) if len(open_to_sell_indices) > 0 else float(theta + sigma)
    buy_to_close = float(x_grid[buy_to_close_indices[0]]) if len(buy_to_close_indices) > 0 else float(theta + sigma)
    sell_to_close = float(x_grid[sell_to_close_indices[-1]]) if len(sell_to_close_indices) > 0 else float(theta - sigma)
    
    return SwitchingBoundaries(
        open_to_buy=open_to_buy,
        open_to_sell=open_to_sell,
        buy_to_close=buy_to_close,
        sell_to_close=sell_to_close,
        V_open=V_open,
        V_buy=V_buy,
        V_sell=V_sell,
        spread_grid=x_grid
    )


def backtest_optimal_switching(
    prices1: pd.Series,
    prices2: pd.Series,
    hedge_ratio: float,
    boundaries: SwitchingBoundaries,
    transaction_cost_bps: float = 10.0,
    initial_capital: float = 100000.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtest optimal switching strategy
    
    Parameters:
    -----------
    prices1 : pd.Series
        Asset 1 prices
    prices2 : pd.Series
        Asset 2 prices
    hedge_ratio : float
        Hedge ratio from cointegration
    boundaries : SwitchingBoundaries
        Optimal switching boundaries
    transaction_cost_bps : float
        Transaction cost in basis points
    initial_capital : float
        Initial capital
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (equity_curve, trades) - Backtest results with trades and P&L
    """
    # Align price series
    common_idx = prices1.index.intersection(prices2.index)
    p1 = prices1.loc[common_idx]
    p2 = prices2.loc[common_idx]
    
    # Compute spread
    spread = p1 - hedge_ratio * p2
    
    # Initialize state
    state = 'open'  # 'open', 'buy', 'sell'
    position_value = 0.0
    cash = initial_capital
    trades = []
    equity_curve = []
    
    # Position tracking variables
    qty1 = 0.0
    qty2 = 0.0
    entry_cost = 0.0
    entry_proceeds = 0.0
    
    tc = transaction_cost_bps / 10000.0  # Convert to decimal
    
    for t, (idx, s) in enumerate(spread.items()):
        prev_state = state
        
        # State transitions based on boundaries
        if state == 'open':
            if s <= boundaries.open_to_buy:
                # Enter long spread position
                state = 'buy'
                # Long asset 1, short asset 2
                qty1 = initial_capital / (2 * p1.iloc[t])
                qty2 = hedge_ratio * qty1
                entry_cost = qty1 * p1.iloc[t] * (1 + tc) + qty2 * p2.iloc[t] * (1 - tc)
                cash -= entry_cost
                position_value = qty1 * p1.iloc[t] - qty2 * p2.iloc[t]
                trades.append({
                    'timestamp': idx,
                    'action': 'open_long_spread',
                    'spread': s,
                    'qty1': qty1,
                    'qty2': -qty2,
                    'cost': entry_cost
                })
            elif s >= boundaries.open_to_sell:
                # Enter short spread position
                state = 'sell'
                # Short asset 1, long asset 2
                qty1 = initial_capital / (2 * p1.iloc[t])
                qty2 = hedge_ratio * qty1
                entry_proceeds = qty1 * p1.iloc[t] * (1 - tc) - qty2 * p2.iloc[t] * (1 + tc)
                cash += entry_proceeds
                position_value = -qty1 * p1.iloc[t] + qty2 * p2.iloc[t]
                trades.append({
                    'timestamp': idx,
                    'action': 'open_short_spread',
                    'spread': s,
                    'qty1': -qty1,
                    'qty2': qty2,
                    'proceeds': entry_proceeds
                })
        
        elif state == 'buy':
            # Update position value (mark to market)
            position_value = qty1 * p1.iloc[t] - qty2 * p2.iloc[t]
            
            if s >= boundaries.buy_to_close:
                # Close long spread position
                state = 'open'
                exit_proceeds = qty1 * p1.iloc[t] * (1 - tc) - qty2 * p2.iloc[t] * (1 + tc)
                cash += exit_proceeds
                pnl = exit_proceeds - entry_cost
                trades.append({
                    'timestamp': idx,
                    'action': 'close_long_spread',
                    'spread': s,
                    'proceeds': exit_proceeds,
                    'pnl': pnl
                })
                position_value = 0.0
                qty1 = 0.0
                qty2 = 0.0
                entry_cost = 0.0
        
        elif state == 'sell':
            # Update position value (mark to market)
            position_value = -qty1 * p1.iloc[t] + qty2 * p2.iloc[t]
            
            if s <= boundaries.sell_to_close:
                # Close short spread position
                state = 'open'
                exit_cost = qty1 * p1.iloc[t] * (1 + tc) + qty2 * p2.iloc[t] * (1 - tc)
                cash -= exit_cost
                pnl = entry_proceeds - exit_cost
                trades.append({
                    'timestamp': idx,
                    'action': 'close_short_spread',
                    'spread': s,
                    'cost': exit_cost,
                    'pnl': pnl
                })
                position_value = 0.0
                qty1 = 0.0
                qty2 = 0.0
                entry_proceeds = 0.0
        
        # Record equity
        total_equity = cash + position_value
        equity_curve.append({
            'timestamp': idx,
            'spread': s,
            'state': state,
            'cash': cash,
            'position_value': position_value,
            'total_equity': total_equity
        })
    
    return pd.DataFrame(equity_curve), pd.DataFrame(trades)


def compute_strategy_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, float]:
    """Compute performance metrics for optimal switching strategy"""
    
    # Returns
    returns = equity_curve['total_equity'].pct_change().dropna()
    
    # Cumulative return
    total_return = (equity_curve['total_equity'].iloc[-1] / equity_curve['total_equity'].iloc[0]) - 1
    
    # Sharpe ratio (annualized, assuming daily data)
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 and returns.std() > 0 else 0
    
    # Maximum drawdown
    cummax = equity_curve['total_equity'].cummax()
    drawdown = (equity_curve['total_equity'] - cummax) / cummax
    max_dd = drawdown.min()
    
    # Win rate
    if len(trades) > 0 and 'pnl' in trades.columns:
        completed_trades = trades.dropna(subset=['pnl'])
        win_rate = (completed_trades['pnl'] > 0).sum() / len(completed_trades) if len(completed_trades) > 0 else 0
        avg_win = completed_trades[completed_trades['pnl'] > 0]['pnl'].mean() if any(completed_trades['pnl'] > 0) else 0
        avg_loss = completed_trades[completed_trades['pnl'] < 0]['pnl'].mean() if any(completed_trades['pnl'] < 0) else 0
        profit_factor = -avg_win / avg_loss * win_rate / (1 - win_rate) if avg_loss != 0 and win_rate < 1 else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Win Rate': win_rate,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Profit Factor': profit_factor,
        'Num Trades': len(trades)
    }
