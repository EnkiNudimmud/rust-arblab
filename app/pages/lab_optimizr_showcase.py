"""
OptimizR Showcase Laboratory
=============================
Interactive demonstration of all OptimizR optimization algorithms:

1. Hidden Markov Models (HMM) - Market regime detection
2. MCMC Sampling - Bayesian parameter estimation
3. Differential Evolution - Global optimization
4. Grid Search - Exhaustive parameter search
5. Information Theory - Mutual information & entropy

All algorithms feature Rust acceleration with 50-100x speedup.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import time

sys.path.append('/app')

# Import OptimizR
try:
    from optimizr import (
        HMM,
        mcmc_sample,
        differential_evolution,
        grid_search,
        mutual_information,
        shannon_entropy,
    )
    OPTIMIZR_AVAILABLE = True
except ImportError:
    OPTIMIZR_AVAILABLE = False
    st.error("⚠️ OptimizR not installed. Install with: pip install optimizr")

from utils.ui_components import render_sidebar_navigation, apply_custom_css

# Page config
st.set_page_config(
    page_title="OptimizR Showcase",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()
render_sidebar_navigation()

st.title("🚀 OptimizR Showcase Laboratory")

st.markdown("""
**Test and explore all OptimizR optimization algorithms with real market data**

OptimizR provides high-performance implementations of advanced algorithms:
- 🔬 **50-100x faster** than pure Python
- 🦀 **Rust-accelerated** computations
- 🐍 **Easy Python API** with NumPy integration
""")

if not OPTIMIZR_AVAILABLE:
    st.stop()

# Check for data
if 'historical_data' not in st.session_state or st.session_state.historical_data is None:
    st.warning("⚠️ Please load market data first")
    if st.button("📊 Go to Data Loader"):
        st.switch_page("pages/data_loader.py")
    st.stop()

df = st.session_state.historical_data

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    [
        "1. Hidden Markov Models (HMM)",
        "2. MCMC Sampling",
        "3. Differential Evolution",
        "4. Grid Search",
        "5. Information Theory"
    ]
)

# ============================================================================
# 1. HIDDEN MARKOV MODELS
# ============================================================================
if "HMM" in algorithm:
    st.header("🔬 Hidden Markov Models - Market Regime Detection")
    
    st.markdown("""
    **Theory**: HMMs model sequential data with hidden states using:
    - **Baum-Welch Algorithm** (EM) for parameter learning
    - **Viterbi Decoding** for most likely state sequence
    - **Gaussian Emissions** for continuous observations
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        n_states = st.slider("Number of Market Regimes", 2, 5, 3)
        n_iterations = st.slider("Training Iterations", 10, 200, 100)
        tolerance = st.select_slider("Convergence Tolerance", 
                                     options=[1e-3, 1e-4, 1e-5, 1e-6],
                                     value=1e-6,
                                     format_func=lambda x: f"{x:.0e}")
    
    with col2:
        st.subheader("Data Preparation")
        returns = df['close'].pct_change().dropna().values
        st.metric("Data Points", len(returns))
        st.metric("Mean Return", f"{returns.mean():.4%}")
        st.metric("Volatility", f"{returns.std():.4%}")
    
    if st.button("🚀 Fit HMM Model", type="primary"):
        with st.spinner("Fitting HMM with Rust acceleration..."):
            start_time = time.perf_counter()
            
            # Fit HMM
            hmm = HMM(n_states=n_states, random_state=42)
            hmm.fit(returns, n_iterations=n_iterations, tolerance=tolerance)
            
            # Predict states
            predicted_states = hmm.predict(returns)
            
            elapsed = time.perf_counter() - start_time
            
            st.success(f"✅ HMM fitted in {elapsed*1000:.1f}ms")
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Transition Matrix")
            trans_df = pd.DataFrame(
                hmm.transition_matrix_,
                columns=[f"State {i}" for i in range(n_states)],
                index=[f"State {i}" for i in range(n_states)]
            )
            st.dataframe(trans_df.style.format("{:.3f}").background_gradient(cmap='RdYlGn'))
        
        with col2:
            st.subheader("Emission Parameters")
            emission_df = pd.DataFrame({
                'Mean Return': hmm.emission_means_,
                'Std Dev': hmm.emission_stds_,
                'Annual Return': hmm.emission_means_ * 252,
                'Annual Vol': hmm.emission_stds_ * np.sqrt(252)
            }, index=[f"State {i}" for i in range(n_states)])
            st.dataframe(emission_df.style.format("{:.4f}"))
        
        # Visualizations
        st.subheader("📊 Regime Visualization")
        
        # Sort states by mean return
        sorted_indices = np.argsort(hmm.emission_means_)[::-1]
        regime_names = ['Bull', 'Neutral', 'Bear', 'Extreme', 'Volatile'][:n_states]
        state_mapping = {old: regime_names[new] for new, old in enumerate(sorted_indices)}
        
        # Create price chart with regimes
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=df.index[1:],  # Skip first NaN
            y=df['close'].iloc[1:],
            name='Price',
            line=dict(color='blue', width=1),
            hovertemplate='%{y:.2f}<extra></extra>'
        ))
        
        # Color by regime
        colors = {'Bull': 'green', 'Neutral': 'gray', 'Bear': 'red', 
                 'Extreme': 'purple', 'Volatile': 'orange'}
        
        for state_id in range(n_states):
            mask = predicted_states == state_id
            regime = state_mapping[state_id]
            fig.add_trace(go.Scatter(
                x=df.index[1:][mask],
                y=df['close'].iloc[1:].values[mask],
                mode='markers',
                name=regime,
                marker=dict(color=colors.get(regime, 'blue'), size=3),
                hovertemplate=f'{regime}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Price with Detected Market Regimes",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # State distribution
        col1, col2 = st.columns([1, 1])
        
        with col1:
            state_counts = pd.Series([state_mapping[s] for s in predicted_states]).value_counts()
            fig_dist = px.pie(
                values=state_counts.values,
                names=state_counts.index,
                title="Regime Distribution",
                color=state_counts.index,
                color_discrete_map=colors
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Performance by regime
            regime_returns = pd.DataFrame({
                'Return': returns,
                'Regime': [state_mapping[s] for s in predicted_states]
            })
            regime_perf = regime_returns.groupby('Regime')['Return'].agg(['mean', 'std', 'count'])
            regime_perf['Sharpe'] = regime_perf['mean'] / regime_perf['std'] * np.sqrt(252)
            st.dataframe(regime_perf.style.format({
                'mean': '{:.4f}',
                'std': '{:.4f}',
                'count': '{:.0f}',
                'Sharpe': '{:.2f}'
            }))

# ============================================================================
# 2. MCMC SAMPLING
# ============================================================================
elif "MCMC" in algorithm:
    st.header("🔬 MCMC Sampling - Bayesian Parameter Estimation")
    
    st.markdown("""
    **Theory**: Metropolis-Hastings algorithm for sampling from posterior distributions:
    - **Bayesian Inference** with prior beliefs
    - **Posterior Estimation** via MCMC
    - **Uncertainty Quantification**
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        n_samples = st.slider("MCMC Samples", 1000, 50000, 15000)
        burn_in = st.slider("Burn-in Period", 500, 5000, 2000)
        proposal_std_mu = st.slider("Proposal Std (μ)", 0.0001, 0.001, 0.0002, 0.0001)
        proposal_std_sigma = st.slider("Proposal Std (σ)", 0.001, 0.01, 0.002, 0.001)
    
    with col2:
        st.subheader("Target Distribution")
        st.info("Estimating mean (μ) and volatility (σ) of returns")
        returns = df['close'].pct_change().dropna().values
        st.metric("Observations", len(returns))
        st.metric("Sample Mean", f"{returns.mean():.5f}")
        st.metric("Sample Std", f"{returns.std():.5f}")
    
    if st.button("🚀 Run MCMC Sampling", type="primary"):
        
        def log_posterior(params, data):
            """Log posterior for return distribution"""
            mu, sigma = params
            
            if sigma <= 0:
                return -np.inf
            
            # Log-likelihood
            residuals = (data - mu) / sigma
            log_lik = -0.5 * len(data) * np.log(2 * np.pi)
            log_lik -= len(data) * np.log(sigma)
            log_lik -= 0.5 * np.sum(residuals**2)
            
            # Log-prior for mu ~ N(0, 0.1^2)
            log_prior_mu = -0.5 * (mu / 0.1)**2
            
            # Log-prior for sigma ~ LogNormal(log(0.02), 0.5)
            log_prior_sigma = -0.5 * ((np.log(sigma) - np.log(0.02)) / 0.5)**2 - np.log(sigma)
            
            return log_lik + log_prior_mu + log_prior_sigma
        
        with st.spinner("Running MCMC with Rust acceleration..."):
            start_time = time.perf_counter()
            
            samples, acceptance_rate = mcmc_sample(
                log_likelihood_fn=log_posterior,
                data=returns,
                initial_params=[0.001, 0.02],
                param_bounds=[(-0.01, 0.01), (0.001, 0.1)],
                proposal_std=[proposal_std_mu, proposal_std_sigma],
                n_samples=n_samples,
                burn_in=burn_in
            )
            
            elapsed = time.perf_counter() - start_time
            
            st.success(f"✅ MCMC completed in {elapsed*1000:.1f}ms | Acceptance: {acceptance_rate:.2%}")
        
        # Results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Posterior Estimates")
            mu_mean = samples[:, 0].mean()
            mu_std = samples[:, 0].std()
            sigma_mean = samples[:, 1].mean()
            sigma_std = samples[:, 1].std()
            
            results_df = pd.DataFrame({
                'Parameter': ['μ (Mean)', 'σ (Volatility)'],
                'Posterior Mean': [mu_mean, sigma_mean],
                'Posterior Std': [mu_std, sigma_std],
                '95% CI Lower': [
                    np.percentile(samples[:, 0], 2.5),
                    np.percentile(samples[:, 1], 2.5)
                ],
                '95% CI Upper': [
                    np.percentile(samples[:, 0], 97.5),
                    np.percentile(samples[:, 1], 97.5)
                ],
                'Annual Mean': [mu_mean * 252, sigma_mean * np.sqrt(252)],
                'Annual Std': [mu_std * 252, sigma_std * np.sqrt(252)]
            })
            st.dataframe(results_df.style.format({
                'Posterior Mean': '{:.6f}',
                'Posterior Std': '{:.6f}',
                '95% CI Lower': '{:.6f}',
                '95% CI Upper': '{:.6f}',
                'Annual Mean': '{:.4f}',
                'Annual Std': '{:.4f}'
            }))
        
        with col2:
            st.subheader("Diagnostics")
            st.metric("Acceptance Rate", f"{acceptance_rate:.2%}")
            st.metric("Effective Sample Size", f"{len(samples)}")
            st.metric("Burn-in Discarded", f"{burn_in}")
        
        # Visualizations
        st.subheader("📊 MCMC Diagnostics")
        
        # Trace plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Trace: μ', 'Trace: σ', 'Posterior: μ', 'Posterior: σ')
        )
        
        # Traces
        fig.add_trace(
            go.Scatter(y=samples[:, 0]*252, mode='lines', line=dict(width=0.5),
                      name='μ trace', showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=samples[:, 1]*np.sqrt(252), mode='lines', line=dict(width=0.5, color='orange'),
                      name='σ trace', showlegend=False),
            row=1, col=2
        )
        
        # Posteriors
        fig.add_trace(
            go.Histogram(x=samples[:, 0]*252, nbinsx=50, name='μ posterior', showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(x=samples[:, 1]*np.sqrt(252), nbinsx=50, name='σ posterior', 
                        marker_color='orange', showlegend=False),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Sample", row=1, col=1)
        fig.update_xaxes(title_text="Sample", row=1, col=2)
        fig.update_xaxes(title_text="Annual Return", row=2, col=1)
        fig.update_xaxes(title_text="Annual Volatility", row=2, col=2)
        fig.update_yaxes(title_text="μ", row=1, col=1)
        fig.update_yaxes(title_text="σ", row=1, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 3. DIFFERENTIAL EVOLUTION
# ============================================================================
elif "Differential Evolution" in algorithm:
    st.header("🧬 Differential Evolution - Global Optimization")
    
    st.markdown("""
    **Theory**: Population-based stochastic optimization for non-convex problems:
    - **Mutation**: Create trial vectors from population
    - **Crossover**: Mix parent and trial solutions
    - **Selection**: Keep better solutions
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        test_function = st.selectbox(
            "Test Function",
            ["Rosenbrock", "Sphere", "Rastrigin", "Portfolio Sharpe"]
        )
        
        if test_function != "Portfolio Sharpe":
            dimensions = st.slider("Dimensions", 2, 20, 5)
        else:
            dimensions = 2  # Portfolio weights
        
        population_size = st.slider("Population Size", 5, 50, 15)
        max_iterations = st.slider("Max Iterations", 100, 2000, 500)
    
    with col2:
        st.subheader("Problem Description")
        if test_function == "Rosenbrock":
            st.latex(r"f(\mathbf{x}) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1-x_i)^2]")
            st.info("Famous test function with global minimum at (1, 1, ..., 1)")
        elif test_function == "Sphere":
            st.latex(r"f(\mathbf{x}) = \sum_{i=1}^{n} x_i^2")
            st.info("Simple convex function with global minimum at origin")
        elif test_function == "Rastrigin":
            st.latex(r"f(\mathbf{x}) = 10n + \sum_{i=1}^{n} [x_i^2 - 10\cos(2\pi x_i)]")
            st.info("Highly multimodal function, challenging for optimizers")
        else:
            st.latex(r"\max \text{Sharpe} = \frac{\mathbf{w}^T \boldsymbol{\mu}}{\sqrt{\mathbf{w}^T \Sigma \mathbf{w}}}")
            st.info("Optimize portfolio allocation for maximum Sharpe ratio")
    
    if st.button("🚀 Run Optimization", type="primary"):
        
        # Define objective functions
        def rosenbrock(x):
            return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
                      for i in range(len(x)-1))
        
        def sphere(x):
            return sum(xi**2 for xi in x)
        
        def rastrigin(x):
            return 10*len(x) + sum(xi**2 - 10*np.cos(2*np.pi*xi) for xi in x)
        
        def portfolio_objective(x):
            """Negative Sharpe ratio for portfolio"""
            returns = df['close'].pct_change().dropna()
            # Simulate two assets (simplified)
            asset1_ret = returns.values
            asset2_ret = returns.shift(1).dropna().values[:len(asset1_ret)]
            
            w1, w2 = x[0], x[1]
            if abs(w1 + w2 - 1.0) > 0.01:  # Must sum to 1
                return 1e10
            
            port_returns = w1 * asset1_ret + w2 * asset2_ret
            sharpe = port_returns.mean() / port_returns.std() * np.sqrt(252)
            return -sharpe  # Minimize negative Sharpe
        
        # Select function and bounds
        if test_function == "Rosenbrock":
            func = rosenbrock
            bounds = [(-5, 5)] * dimensions
        elif test_function == "Sphere":
            func = sphere
            bounds = [(-10, 10)] * dimensions
        elif test_function == "Rastrigin":
            func = rastrigin
            bounds = [(-5.12, 5.12)] * dimensions
        else:
            func = portfolio_objective
            bounds = [(0, 1), (0, 1)]
        
        with st.spinner("Running Differential Evolution with Rust..."):
            start_time = time.perf_counter()
            
            result = differential_evolution(
                objective_fn=func,
                bounds=bounds,
                population_size=population_size,
                max_iterations=max_iterations,
                tolerance=1e-6
            )
            
            elapsed = time.perf_counter() - start_time
            
            st.success(f"✅ Optimization completed in {elapsed*1000:.1f}ms")
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Optimal Solution")
            if test_function == "Portfolio Sharpe":
                st.metric("Asset 1 Weight", f"{result.x[0]:.2%}")
                st.metric("Asset 2 Weight", f"{result.x[1]:.2%}")
                st.metric("Sharpe Ratio", f"{-result.fun:.3f}")
            else:
                st.metric("Function Value", f"{result.fun:.6e}")
                st.metric("Iterations", result.nit)
                st.metric("Function Evaluations", result.nfev)
        
        with col2:
            st.subheader("Optimal Parameters")
            params_df = pd.DataFrame({
                'Dimension': [f'x{i}' for i in range(len(result.x))],
                'Value': result.x
            })
            st.dataframe(params_df.style.format({'Value': '{:.6f}'}))

# ============================================================================
# 4. GRID SEARCH
# ============================================================================
elif "Grid Search" in algorithm:
    st.header("🔍 Grid Search - Exhaustive Parameter Search")
    
    st.markdown("""
    **Theory**: Systematically evaluate all parameter combinations:
    - **Exhaustive Search** over discretized space
    - **Guaranteed Optimum** within grid resolution
    - **Computational Cost**: O(n_points^n_dimensions)
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        n_points = st.slider("Points per Dimension", 10, 100, 50)
        search_type = st.selectbox(
            "Optimization Problem",
            ["2D Quadratic", "Strategy Parameters"]
        )
    
    with col2:
        if search_type == "2D Quadratic":
            st.subheader("Problem")
            st.latex(r"f(x, y) = -(x^2 + y^2)")
            st.info("Find maximum at (0, 0)")
            total_evals = n_points ** 2
        else:
            st.subheader("Strategy Parameters")
            st.info("Optimize entry/exit thresholds")
            total_evals = n_points ** 2
        
        st.metric("Total Evaluations", f"{total_evals:,}")
    
    if st.button("🚀 Run Grid Search", type="primary"):
        
        if search_type == "2D Quadratic":
            def objective(x):
                return -(x[0]**2 + x[1]**2)
            
            bounds = [(-5, 5), (-5, 5)]
        else:
            def objective(x):
                # Simulate strategy performance
                entry_z, exit_z = x
                returns = df['close'].pct_change().dropna().values
                # Simplified backtest
                positions = np.where(returns < -entry_z * returns.std(), 1, 0)
                positions = np.where(returns > exit_z * returns.std(), 0, positions)
                pnl = (returns * positions).sum()
                return pnl
            
            bounds = [(0.5, 3.0), (0.5, 3.0)]
        
        with st.spinner("Running Grid Search with Rust..."):
            start_time = time.perf_counter()
            
            result = grid_search(
                objective_fn=objective,
                bounds=bounds,
                n_points=n_points
            )
            
            elapsed = time.perf_counter() - start_time
            
            st.success(f"✅ Grid search completed in {elapsed*1000:.1f}ms")
        
        # Results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Optimal Solution")
            st.metric("Best Value", f"{result.fun:.6f}")
            st.metric("Evaluations", f"{result.nfev:,}")
            st.metric("Time per Eval", f"{elapsed/result.nfev*1000:.3f}ms")
        
        with col2:
            st.subheader("Optimal Parameters")
            if search_type == "2D Quadratic":
                st.metric("x", f"{result.x[0]:.4f}")
                st.metric("y", f"{result.x[1]:.4f}")
            else:
                st.metric("Entry Z-Score", f"{result.x[0]:.3f}")
                st.metric("Exit Z-Score", f"{result.x[1]:.3f}")
        
        # Visualize grid
        st.subheader("📊 Search Space Visualization")
        
        # Create grid
        x_vals = np.linspace(bounds[0][0], bounds[0][1], n_points)
        y_vals = np.linspace(bounds[1][0], bounds[1][1], n_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)
        
        for i in range(n_points):
            for j in range(n_points):
                Z[i, j] = objective([X[i, j], Y[i, j]])
        
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
        
        # Add optimal point
        fig.add_trace(go.Scatter3d(
            x=[result.x[0]], y=[result.x[1]], z=[result.fun],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Optimum'
        ))
        
        fig.update_layout(
            title="Objective Function Landscape",
            scene=dict(
                xaxis_title='Parameter 1',
                yaxis_title='Parameter 2',
                zaxis_title='Objective Value'
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 5. INFORMATION THEORY
# ============================================================================
elif "Information Theory" in algorithm:
    st.header("🧠 Information Theory - Dependency Analysis")
    
    st.markdown("""
    **Theory**: Quantify information content and dependencies:
    - **Shannon Entropy**: $H(X) = -\\sum p(x) \\log p(x)$
    - **Mutual Information**: $I(X;Y) = H(X) + H(Y) - H(X,Y)$
    - **Feature Selection** via MI
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        n_bins = st.slider("Number of Bins", 10, 50, 20)
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Price-Volume Dependency", "Returns Autocorrelation", "Cross-Asset MI"]
        )
    
    with col2:
        st.subheader("Data")
        st.metric("Observations", len(df))
        st.metric("Bins", n_bins)
    
    if st.button("🚀 Run Analysis", type="primary"):
        
        with st.spinner("Computing with Rust acceleration..."):
            start_time = time.perf_counter()
            
            if analysis_type == "Price-Volume Dependency":
                x = df['close'].pct_change().dropna().values
                y = df['volume'].pct_change().dropna().values[:len(x)]
                
                mi = mutual_information(x, y, n_bins=n_bins)
                h_x = shannon_entropy(x, n_bins=n_bins)
                h_y = shannon_entropy(y, n_bins=n_bins)
                
            elif analysis_type == "Returns Autocorrelation":
                returns = df['close'].pct_change().dropna().values
                x = returns[:-1]
                y = returns[1:]
                
                mi = mutual_information(x, y, n_bins=n_bins)
                h_x = shannon_entropy(x, n_bins=n_bins)
                h_y = shannon_entropy(y, n_bins=n_bins)
                
            else:  # Cross-Asset
                # Simulate second asset (or use another column if available)
                x = df['close'].pct_change().dropna().values
                y = df['high'].pct_change().dropna().values[:len(x)]
                
                mi = mutual_information(x, y, n_bins=n_bins)
                h_x = shannon_entropy(x, n_bins=n_bins)
                h_y = shannon_entropy(y, n_bins=n_bins)
            
            elapsed = time.perf_counter() - start_time
            
            st.success(f"✅ Analysis completed in {elapsed*1000:.1f}ms")
        
        # Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mutual Information", f"{mi:.4f} bits")
        
        with col2:
            st.metric("Entropy X", f"{h_x:.4f} bits")
        
        with col3:
            st.metric("Entropy Y", f"{h_y:.4f} bits")
        
        # Normalized MI
        norm_mi = mi / min(h_x, h_y) if min(h_x, h_y) > 0 else 0
        st.metric("Normalized MI (correlation-like)", f"{norm_mi:.4f}")
        
        # Interpretation
        st.subheader("📊 Interpretation")
        
        if norm_mi < 0.1:
            st.info("🔵 **Weak Dependency**: Variables are nearly independent")
        elif norm_mi < 0.3:
            st.info("🟡 **Moderate Dependency**: Some relationship exists")
        else:
            st.info("🔴 **Strong Dependency**: Variables are highly correlated")
        
        # Visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Joint Distribution', 'Individual Entropies')
        )
        
        # Joint scatter
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='markers', 
                      marker=dict(size=3, opacity=0.5),
                      name='Data', showlegend=False),
            row=1, col=1
        )
        
        # Entropies bar chart
        fig.add_trace(
            go.Bar(x=['H(X)', 'H(Y)', 'I(X;Y)'], 
                  y=[h_x, h_y, mi],
                  marker_color=['blue', 'orange', 'green'],
                  showlegend=False),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="X", row=1, col=1)
        fig.update_yaxes(title_text="Y", row=1, col=1)
        fig.update_xaxes(title_text="Metric", row=1, col=2)
        fig.update_yaxes(title_text="Bits", row=1, col=2)
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
### 🚀 About OptimizR

OptimizR provides **production-ready** optimization algorithms with:
- **50-100x speedup** vs pure Python
- **Rust-powered** performance
- **Easy Python API** with NumPy
- **Comprehensive testing** & documentation

[GitHub](https://github.com/ThotDjehuty/optimiz-r) | [Documentation](https://github.com/ThotDjehuty/optimiz-r/blob/main/README.md)
""")
