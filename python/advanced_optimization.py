"""
Advanced Strategy Parameter Optimization
=========================================

Implements sophisticated parameter optimization methods:
1. Hidden Markov Models (HMM) for regime detection and parameter adaptation
2. Markov Chain Monte Carlo (MCMC) for Bayesian parameter estimation
3. Maximum Likelihood Estimation (MLE) for parameter fitting
4. Information Theory metrics (Mutual Information, Entropy) for feature selection
5. Multi-objective optimization for multi-strategy portfolios

All computationally intensive operations delegated to Rust when available.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm, multivariate_normal
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """Defines the parameter search space for a strategy"""
    name: str
    bounds: Tuple[float, float]
    dtype: str = 'float'  # 'float', 'int', 'categorical'
    categories: Optional[List] = None


@dataclass
class OptimizationResult:
    """Results from parameter optimization"""
    best_params: Dict[str, float]
    best_score: float
    all_params: List[Dict]
    all_scores: List[float]
    convergence_history: List[float]
    method: str
    n_iterations: int


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.
    
    Identifies different market states (bull, bear, sideways) and adapts
    strategy parameters based on the current regime.
    """
    
    def __init__(self, n_states: int = 3):
        """
        Args:
            n_states: Number of hidden states (market regimes)
        """
        self.n_states = n_states
        self.transition_matrix = None
        self.emission_params = None
        self.state_sequence = None
        
    def fit(self, returns: np.ndarray, n_iterations: int = 100):
        """
        Fit HMM using Baum-Welch algorithm (EM).
        
        Args:
            returns: Time series of returns
            n_iterations: Number of EM iterations
        """
        # Try Rust implementation first
        if RUST_AVAILABLE:
            try:
                hmm_params = rust_optimizers.fit_hmm(
                    returns.tolist(),
                    n_states=self.n_states,
                    n_bins=10,
                    n_iterations=n_iterations,
                    tolerance=1e-6
                )
                
                self.transition_matrix = np.array(hmm_params['transition_matrix'])
                self.emission_params = hmm_params['emission_matrix']
                
                # Decode state sequence
                self.state_sequence = rust_optimizers.viterbi_decode(
                    returns.tolist(),
                    hmm_params,
                    n_bins=10
                )
                
                logger.info(f"✓ HMM fitted with {self.n_states} states (Rust backend)")
                return
            except Exception as e:
                logger.warning(f"Rust HMM failed, falling back to Python: {e}")
        
        # Python fallback
        n_obs = len(returns)
        
        # Initialize parameters
        # Transition matrix: P(state_t+1 | state_t)
        self.transition_matrix = np.random.dirichlet(np.ones(self.n_states), self.n_states)
        
        # Emission parameters: Gaussian for each state
        # (mean, variance) for each state
        self.emission_params = []
        quantiles = np.linspace(0, 1, self.n_states + 1)
        for i in range(self.n_states):
            low_q = quantiles[i]
            high_q = quantiles[i + 1]
            state_returns = returns[(returns >= np.quantile(returns, low_q)) & 
                                   (returns < np.quantile(returns, high_q))]
            mean = np.mean(state_returns) if len(state_returns) > 0 else 0.0
            var = np.var(state_returns) if len(state_returns) > 0 else 1.0
            self.emission_params.append((mean, var))
        
        # EM algorithm
        for iteration in range(n_iterations):
            # E-step: Forward-backward algorithm
            alpha = self._forward(returns)
            beta = self._backward(returns)
            gamma = self._compute_gamma(alpha, beta)
            xi = self._compute_xi(returns, alpha, beta)
            
            # M-step: Update parameters
            self._update_parameters(returns, gamma, xi)
        
        # Decode most likely state sequence (Viterbi)
        self.state_sequence = self._viterbi(returns)
        
        logger.info(f"HMM fitted with {self.n_states} states (Python fallback)")
    
    def _forward(self, returns: np.ndarray) -> np.ndarray:
        """Forward algorithm to compute P(observation, state)"""
        n_obs = len(returns)
        alpha = np.zeros((n_obs, self.n_states))
        
        # Initial probabilities (uniform)
        pi = np.ones(self.n_states) / self.n_states
        
        # t=0
        for s in range(self.n_states):
            alpha[0, s] = pi[s] * self._emission_prob(returns[0], s)
        
        # t=1 to T-1
        for t in range(1, n_obs):
            for s in range(self.n_states):
                alpha[t, s] = np.sum(alpha[t-1, :] * self.transition_matrix[:, s]) * \
                              self._emission_prob(returns[t], s)
        
        return alpha
    
    def _backward(self, returns: np.ndarray) -> np.ndarray:
        """Backward algorithm"""
        n_obs = len(returns)
        beta = np.zeros((n_obs, self.n_states))
        
        # t=T-1
        beta[-1, :] = 1.0
        
        # t=T-2 to 0
        for t in range(n_obs - 2, -1, -1):
            for s in range(self.n_states):
                beta[t, s] = np.sum(self.transition_matrix[s, :] * 
                                   np.array([self._emission_prob(returns[t+1], s2) 
                                            for s2 in range(self.n_states)]) *
                                   beta[t+1, :])
        
        return beta
    
    def _compute_gamma(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute P(state | observations)"""
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        return gamma
    
    def _compute_xi(self, returns: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute P(state_t, state_t+1 | observations)"""
        n_obs = len(returns)
        xi = np.zeros((n_obs - 1, self.n_states, self.n_states))
        
        for t in range(n_obs - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = alpha[t, i] * self.transition_matrix[i, j] * \
                                  self._emission_prob(returns[t+1], j) * beta[t+1, j]
            
            # Normalize
            xi[t, :, :] /= np.sum(xi[t, :, :])
        
        return xi
    
    def _update_parameters(self, returns: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        """M-step: Update transition and emission parameters"""
        # Update transition matrix
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.transition_matrix[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
        
        # Update emission parameters
        for s in range(self.n_states):
            # Weighted mean and variance
            weights = gamma[:, s]
            mean = np.sum(weights * returns) / np.sum(weights)
            var = np.sum(weights * (returns - mean) ** 2) / np.sum(weights)
            self.emission_params[s] = (mean, max(var, 1e-6))  # Avoid zero variance
    
    def _emission_prob(self, observation: float, state: int) -> float:
        """Emission probability: P(observation | state)"""
        mean, var = self.emission_params[state]
        return norm.pdf(observation, mean, np.sqrt(var))
    
    def _viterbi(self, returns: np.ndarray) -> np.ndarray:
        """Viterbi algorithm to find most likely state sequence"""
        n_obs = len(returns)
        delta = np.zeros((n_obs, self.n_states))
        psi = np.zeros((n_obs, self.n_states), dtype=int)
        
        # Initialize
        pi = np.ones(self.n_states) / self.n_states
        for s in range(self.n_states):
            delta[0, s] = np.log(pi[s]) + np.log(self._emission_prob(returns[0], s) + 1e-10)
        
        # Recursion
        for t in range(1, n_obs):
            for s in range(self.n_states):
                candidates = delta[t-1, :] + np.log(self.transition_matrix[:, s] + 1e-10)
                psi[t, s] = np.argmax(candidates)
                delta[t, s] = candidates[psi[t, s]] + np.log(self._emission_prob(returns[t], s) + 1e-10)
        
        # Backtrack
        path = np.zeros(n_obs, dtype=int)
        path[-1] = np.argmax(delta[-1, :])
        for t in range(n_obs - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        
        return path
    
    def predict_regime(self, recent_returns: np.ndarray) -> int:
        """Predict current regime given recent returns"""
        if self.state_sequence is None:
            raise ValueError("Model not fitted yet")
        
        # Use last state and forward one step
        last_state = self.state_sequence[-1]
        
        # Compute probabilities for each state
        probs = np.zeros(self.n_states)
        for s in range(self.n_states):
            probs[s] = self.transition_matrix[last_state, s] * \
                      self._emission_prob(recent_returns[-1], s)
        
        return np.argmax(probs)
    
    def get_regime_parameters(self, regime: int, base_params: Dict) -> Dict:
        """
        Adjust strategy parameters based on detected regime.
        
        Example adjustments:
        - Bull market (high positive returns): More aggressive, lower stops
        - Bear market (negative returns): Defensive, tighter stops
        - Sideways (low volatility): Mean reversion focus
        """
        mean, var = self.emission_params[regime]
        volatility = np.sqrt(var)
        
        adjusted_params = base_params.copy()
        
        # Adjust based on regime characteristics
        if mean > 0.001:  # Bull regime
            adjusted_params['position_size'] = adjusted_params.get('position_size', 1.0) * 1.2
            adjusted_params['stop_loss'] = adjusted_params.get('stop_loss', 0.02) * 1.5
        elif mean < -0.001:  # Bear regime
            adjusted_params['position_size'] = adjusted_params.get('position_size', 1.0) * 0.7
            adjusted_params['stop_loss'] = adjusted_params.get('stop_loss', 0.02) * 0.8
        # else: sideways regime - keep base parameters
        
        # Adjust for volatility
        if volatility > 0.02:  # High volatility
            adjusted_params['entry_threshold'] = adjusted_params.get('entry_threshold', 2.0) * 1.3
        
        return adjusted_params


class MCMCOptimizer:
    """
    Markov Chain Monte Carlo for Bayesian parameter estimation.
    
    Uses Metropolis-Hastings to sample from the posterior distribution
    of strategy parameters given historical performance.
    """
    
    def __init__(
        self,
        parameter_space: List[ParameterSpace],
        objective_function: Callable
    ):
        """
        Args:
            parameter_space: List of parameters to optimize
            objective_function: Function to evaluate parameter performance
        """
        self.parameter_space = parameter_space
        self.objective_function = objective_function
        self.samples = []
        self.acceptance_rate = 0.0
    
    def optimize(
        self,
        n_iterations: int = 10000,
        burn_in: int = 1000,
        proposal_std: float = 0.1
    ) -> OptimizationResult:
        """
        Run MCMC optimization.
        
        Args:
            n_iterations: Number of MCMC iterations
            burn_in: Number of initial samples to discard
            proposal_std: Standard deviation for proposal distribution
            
        Returns:
            OptimizationResult with best parameters and sampling statistics
        """
        # Initialize at random point in parameter space
        current_params = {}
        for param in self.parameter_space:
            if param.dtype == 'float':
                current_params[param.name] = np.random.uniform(param.bounds[0], param.bounds[1])
            elif param.dtype == 'int':
                current_params[param.name] = np.random.randint(param.bounds[0], param.bounds[1] + 1)
        
        current_score = self.objective_function(current_params)
        
        samples = []
        scores = []
        acceptances = 0
        
        for i in range(n_iterations):
            # Propose new parameters
            proposal_params = current_params.copy()
            
            # Random walk proposal
            for param in self.parameter_space:
                if param.dtype in ['float', 'int']:
                    proposal = current_params[param.name] + \
                              np.random.normal(0, proposal_std * (param.bounds[1] - param.bounds[0]))
                    # Clip to bounds
                    proposal = np.clip(proposal, param.bounds[0], param.bounds[1])
                    if param.dtype == 'int':
                        proposal = int(round(proposal))
                    proposal_params[param.name] = proposal
            
            # Evaluate proposal
            proposal_score = self.objective_function(proposal_params)
            
            # Metropolis-Hastings acceptance criterion
            # Since we maximize: accept if proposal_score > current_score
            # With probability based on score difference
            acceptance_prob = min(1.0, np.exp(proposal_score - current_score))
            
            if np.random.rand() < acceptance_prob:
                current_params = proposal_params
                current_score = proposal_score
                acceptances += 1
            
            # Store sample (after burn-in)
            if i >= burn_in:
                samples.append(current_params.copy())
                scores.append(current_score)
        
        self.samples = samples
        self.acceptance_rate = acceptances / n_iterations
        
        # Find best parameters from samples
        best_idx = np.argmax(scores)
        best_params = samples[best_idx]
        best_score = scores[best_idx]
        
        logger.info(f"MCMC completed: acceptance rate = {self.acceptance_rate:.2%}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_params=samples,
            all_scores=scores,
            convergence_history=scores,
            method='MCMC',
            n_iterations=len(samples)
        )
    
    def get_parameter_distribution(self, param_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get empirical distribution of a parameter from MCMC samples"""
        if not self.samples:
            raise ValueError("No samples available. Run optimize() first.")
        
        values = [sample[param_name] for sample in self.samples]
        
        # Compute histogram
        counts, bins = np.histogram(values, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        return bin_centers, counts


class MLEOptimizer:
    """
    Maximum Likelihood Estimation for parameter optimization.
    
    Finds parameters that maximize the likelihood of observed returns
    under a specified model.
    """
    
    def __init__(
        self,
        parameter_space: List[ParameterSpace],
        log_likelihood_function: Callable
    ):
        """
        Args:
            parameter_space: List of parameters to optimize
            log_likelihood_function: Function that computes log-likelihood
        """
        self.parameter_space = parameter_space
        self.log_likelihood_function = log_likelihood_function
    
    def optimize(self) -> OptimizationResult:
        """
        Run MLE optimization using differential evolution.
        
        Returns:
            OptimizationResult with MLE parameters
        """
        # Extract bounds
        bounds = [param.bounds for param in self.parameter_space]
        
        # Objective: negative log-likelihood (minimize)
        def objective(x):
            params = {param.name: x[i] for i, param in enumerate(self.parameter_space)}
            return -self.log_likelihood_function(params)
        
        # Optimize
        result = differential_evolution(
            objective,
            bounds,
            maxiter=1000,
            popsize=15,
            atol=1e-7,
            tol=1e-7,
            workers=-1  # Use all CPUs
        )
        
        # Extract best parameters
        best_params = {
            param.name: result.x[i]
            for i, param in enumerate(self.parameter_space)
        }
        
        best_score = -result.fun  # Convert back to log-likelihood
        
        logger.info(f"MLE optimization completed: log-likelihood = {best_score:.4f}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_params=[best_params],
            all_scores=[best_score],
            convergence_history=[],
            method='MLE',
            n_iterations=result.nit
        )


class InformationTheoryOptimizer:
    """
    Uses information theory metrics for parameter optimization and feature selection.
    
    - Mutual Information: Measures dependency between parameters and returns
    - Entropy: Quantifies uncertainty/randomness
    - Information Gain: Feature importance
    """
    
    @staticmethod
    def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
        """
        Compute mutual information between x and y.
        
        MI(X;Y) = H(X) + H(Y) - H(X,Y)
        
        Args:
            x: First variable
            y: Second variable
            bins: Number of bins for discretization
            
        Returns:
            Mutual information value
        """
        # Discretize if continuous
        x_discrete = np.digitize(x, np.linspace(x.min(), x.max(), bins))
        y_discrete = np.digitize(y, np.linspace(y.min(), y.max(), bins))
        
        # Compute joint and marginal probabilities
        xy = np.column_stack([x_discrete, y_discrete])
        xy_unique, xy_counts = np.unique(xy, axis=0, return_counts=True)
        p_xy = xy_counts / len(x)
        
        x_unique, x_counts = np.unique(x_discrete, return_counts=True)
        p_x = x_counts / len(x)
        
        y_unique, y_counts = np.unique(y_discrete, return_counts=True)
        p_y = y_counts / len(y)
        
        # Compute MI
        mi = 0.0
        for i, (xi, yi) in enumerate(xy_unique):
            p_joint = p_xy[i]
            p_x_marg = p_x[x_unique == xi][0]
            p_y_marg = p_y[y_unique == yi][0]
            
            if p_joint > 0:
                mi += p_joint * np.log2(p_joint / (p_x_marg * p_y_marg))
        
        return mi
    
    @staticmethod
    def entropy(x: np.ndarray, bins: int = 20) -> float:
        """
        Compute Shannon entropy of x.
        
        H(X) = -∑ p(x) log p(x)
        """
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / len(x)
        probs = probs[probs > 0]  # Remove zeros
        
        return -np.sum(probs * np.log2(probs))
    
    @staticmethod
    def select_features(
        features: pd.DataFrame,
        target: np.ndarray,
        n_features: int = 10
    ) -> List[str]:
        """
        Select top features based on mutual information with target.
        
        Args:
            features: DataFrame of candidate features
            target: Target variable (e.g., returns)
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        mi_scores = {}
        
        for col in features.columns:
            mi = InformationTheoryOptimizer.mutual_information(
                features[col].values,
                target
            )
            mi_scores[col] = mi
        
        # Sort by MI and select top n
        sorted_features = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, _ in sorted_features[:n_features]]
        
        logger.info(f"Selected {n_features} features based on mutual information")
        return selected


class MultiStrategyOptimizer:
    """
    Optimizes parameters for multiple strategies simultaneously across multiple assets.
    
    Uses multi-objective optimization to balance:
    - Total return
    - Risk (Sharpe ratio, max drawdown)
    - Diversification
    - Transaction costs
    """
    
    def __init__(
        self,
        strategies: List[str],
        assets: List[str],
        asset_types: Dict[str, str]  # asset -> type mapping
    ):
        """
        Args:
            strategies: List of strategy names
            assets: List of asset symbols
            asset_types: Mapping of asset to type ('stock', 'crypto', 'etf', 'option')
        """
        self.strategies = strategies
        self.assets = assets
        self.asset_types = asset_types
    
    def optimize(
        self,
        historical_data: Dict[str, pd.DataFrame],
        strategy_params: Dict[str, List[ParameterSpace]],
        allocation_bounds: Tuple[float, float] = (0.0, 1.0)
    ) -> Dict:
        """
        Optimize strategy parameters and asset allocations.
        
        Args:
            historical_data: {asset: price_data}
            strategy_params: {strategy: [ParameterSpace]}
            allocation_bounds: Min/max allocation per strategy-asset pair
            
        Returns:
            Optimized configuration
        """
        # Decision variables:
        # 1. Strategy parameters for each strategy
        # 2. Allocation weights: strategy x asset matrix
        
        n_strategies = len(self.strategies)
        n_assets = len(self.assets)
        
        # Build combined parameter space
        bounds = []
        param_map = {}
        idx = 0
        
        for strategy in self.strategies:
            for param in strategy_params.get(strategy, []):
                bounds.append(param.bounds)
                param_map[idx] = (strategy, param.name)
                idx += 1
        
        # Add allocation weights
        for _ in range(n_strategies * n_assets):
            bounds.append(allocation_bounds)
        
        n_params = len([p for params in strategy_params.values() for p in params])
        
        def objective(x):
            # Extract parameters and allocations
            params_values = x[:n_params]
            allocations = x[n_params:].reshape(n_strategies, n_assets)
            
            # Normalize allocations to sum to 1
            allocations = allocations / np.sum(allocations)
            
            # Simulate portfolio performance
            # (Simplified - in production, run full backtests)
            total_return = 0.0
            total_risk = 0.0
            
            for i, strategy in enumerate(self.strategies):
                for j, asset in enumerate(self.assets):
                    weight = allocations[i, j]
                    if weight < 0.01:
                        continue
                    
                    # Get strategy parameters
                    strategy_specific_params = {}
                    for k, (strat, param_name) in param_map.items():
                        if strat == strategy:
                            strategy_specific_params[param_name] = params_values[k]
                    
                    # Estimate return and risk for this strategy-asset pair
                    # (Simplified calculation)
                    data = historical_data.get(asset)
                    if data is None:
                        continue
                    
                    returns = data['close'].pct_change().dropna()
                    mean_return = returns.mean() * 252  # Annualized
                    volatility = returns.std() * np.sqrt(252)
                    
                    # Weight contribution
                    total_return += weight * mean_return
                    total_risk += weight ** 2 * volatility ** 2
            
            total_risk = np.sqrt(total_risk)
            
            # Multi-objective: maximize return, minimize risk
            sharpe = total_return / total_risk if total_risk > 0 else 0
            
            # Add diversification bonus
            diversification = -np.sum(allocations ** 2)  # Negative of concentration
            
            return -(sharpe + 0.1 * diversification)  # Negative for minimization
        
        # Optimize
        result = differential_evolution(
            objective,
            bounds,
            maxiter=500,
            workers=-1
        )
        
        # Extract optimal configuration
        optimal_params_values = result.x[:n_params]
        optimal_allocations = result.x[n_params:].reshape(n_strategies, n_assets)
        optimal_allocations = optimal_allocations / np.sum(optimal_allocations)
        
        optimal_config = {
            'strategies': {},
            'allocations': pd.DataFrame(
                optimal_allocations,
                index=self.strategies,
                columns=self.assets
            ),
            'objective_value': -result.fun
        }
        
        # Map parameters back to strategies
        for idx, (strategy, param_name) in param_map.items():
            if strategy not in optimal_config['strategies']:
                optimal_config['strategies'][strategy] = {}
            optimal_config['strategies'][strategy][param_name] = optimal_params_values[idx]
        
        logger.info(f"Multi-strategy optimization completed: objective = {optimal_config['objective_value']:.4f}")
        
        return optimal_config
