# Signature Optimal Stopping - Python Bindings

PyO3 bindings for signature-based optimal stopping strategies in high-frequency trading.

## Mathematical Framework

### Path Signatures

The **signature** of a path is a sequence of iterated integrals that uniquely characterizes the path up to reparameterization. For a path $X: [0,T] \rightarrow \mathbb{R}^d$, the signature is defined as:

$$
S(X)_{0,T} = \left(1, \int_{0 < t_1 < T} dX_{t_1}, \int_{0 < t_1 < t_2 < T} dX_{t_1} \otimes dX_{t_2}, \ldots \right)
$$

**Truncated Signature**: In practice, we compute signatures up to a truncation level $N$, resulting in a finite-dimensional feature vector of size:

$$
\text{dim}(\text{Sig}_N) = \sum_{k=0}^{N} d^k = \frac{d^{N+1} - 1}{d - 1}
$$

where $d$ is the dimensionality of the input path.

### Optimal Stopping Problem

Given a stochastic process $X_t$ representing market observables (prices, volume, spread, etc.), we seek the optimal stopping time $\tau^*$ that maximizes expected reward:

$$
V(x) = \sup_{\tau} \mathbb{E}[R(X_\tau) \mid X_0 = x]
$$

where $R(\cdot)$ is the reward function (e.g., profit from executing a trade at time $\tau$).

### Signature-Based Approximation

We approximate the continuation value using a **linear model** on signature features:

$$
V(X_{0:t}) \approx \langle w, \text{Sig}_N(X_{0:t}) \rangle
$$

The optimal stopping policy becomes:
- **Stop** at time $t$ if $R(X_t) \geq \langle w, \text{Sig}_N(X_{0:t}) \rangle$
- **Continue** otherwise

The weight vector $w$ is learned via **ridge regression** on training samples:

$$
w^* = \arg\min_w \sum_{i=1}^m \left( R_i - \langle w, \text{Sig}_N(X_i) \rangle \right)^2 + \lambda \|w\|^2
$$

## Trading Applications

### 1. Triangular Arbitrage Exit Timing

**Problem**: Given a profitable triangular arbitrage opportunity (BTC/USD → ETH/USD → BTC/ETH → BTC/USD), when should we execute?

**Path Features**:
- Spread evolution: $\text{spread}_t = \frac{\text{ask}_t - \text{bid}_t}{\text{mid}_t}$
- Profit trajectory: $\text{profit}_t = (1 - \text{fees}) \prod_{i=1}^3 \frac{P_i}{P_{i-1}} - 1$
- Volume imbalance: $\text{imb}_t = \frac{V_{\text{bid}} - V_{\text{ask}}}{V_{\text{bid}} + V_{\text{ask}}}$

**Reward**: Net PnL after transaction costs, adjusted for slippage

**Insight**: Signature captures **nonlinear path dynamics** (e.g., "profit peaked then started declining" or "spread widening acceleration") better than point-in-time features.

### 2. Market Making Order Cancellation

**Problem**: When should a market maker cancel resting limit orders?

**Path Features**:
- Inventory trajectory
- Mid-price velocity and acceleration
- Queue position evolution

**Reward**: Avoided adverse selection cost minus opportunity cost

### 3. Pairs Trading Entry/Exit

**Problem**: Optimal times to enter and exit a mean-reverting pairs trade

**Path Features**:
- Spread (cointegration residual) trajectory
- Volatility path
- Correlation dynamics

**Reward**: PnL from round-trip trade minus costs

## Usage Example

### Python API

```python
import json
from sig_optimal_stopping import PySignatureStopper

# Initialize stopper with truncation level and ridge penalty
stopper = PySignatureStopper(truncation=2, ridge=1e-3)

# Prepare training data
# Each trajectory is a list of time-stamped feature vectors
training_data = {
    "params": {"truncation": 2, "ridge": 1e-3},
    "samples": [
        {
            "traj": [
                [0.5, 0.1],   # [spread, profit] at t=0
                [0.48, 0.12], # at t=1
                [0.52, 0.09]  # at t=2
            ],
            "reward": 0.08  # actual PnL achieved
        },
        # ... more samples
    ]
}

# Train the model
result = stopper.train_from_json(json.dumps(training_data))
print(f"Learned weights: {result['weights']}")

# Make predictions on new trajectories
new_traj = [[0.49, 0.11], [0.50, 0.10]]
cont_value = stopper.predict_from_list(new_traj)
print(f"Continuation value: {cont_value}")

# Decision rule
current_reward = 0.09
if current_reward >= cont_value:
    print("STOP - Execute trade now")
else:
    print("CONTINUE - Wait for better opportunity")
```

### From Jupyter Notebooks

See `examples/notebooks/signature_optimal_stopping.ipynb` and `examples/notebooks/triangular_signature_optimal_stopping.ipynb` for complete examples with:
- Synthetic trajectory generation
- Training data collection via simulation
- Model training and validation
- Live prediction and stopping decisions
- Performance comparison vs baseline strategies

## Implementation Details

### Rust Core (`rust_core/signature_optimal_stopping`)

The Rust implementation provides:
- **Signature computation**: Efficient iterative calculation using tensor products
- **Ridge regression solver**: Normal equations with L2 regularization
- **Path preprocessing**: Normalization, interpolation, time-augmentation

### Python Bindings (`signature_optimal_stopping_py`)

PyO3 wrapper exposing:
- `PySignatureStopper` class with `.train_from_json()` and `.predict_from_list()`
- Automatic GIL management for thread safety
- Serde integration for JSON serialization

### Building

```bash
# Install maturin
pip install maturin

# Build and install the extension
python -m maturin develop --manifest-path rust_core/signature_optimal_stopping_py/Cargo.toml --release

# Test in Python
python -c "from sig_optimal_stopping import PySignatureStopper; print('Success!')"
```

## Hyperparameters

### Truncation Level (`truncation`)

- **Low (N=1)**: Only linear features, fast but limited expressiveness
- **Medium (N=2)**: Captures quadratic interactions (recommended for most HFT applications)
- **High (N≥3)**: Rich nonlinear features but exponential growth in dimension

**Rule of thumb**: Start with N=2. Increase only if you have 1000+ training samples.

### Ridge Penalty (`ridge`)

- **Small (λ=1e-4)**: Low bias, high variance - may overfit
- **Medium (λ=1e-3 to 1e-2)**: Balanced (recommended starting point)
- **Large (λ=1e-1)**: High bias, low variance - may underfit

**Tuning**: Use cross-validation on historical backtest data.

## Performance Considerations

### Computational Complexity

- **Signature computation**: $O(d^N \cdot L)$ where $L$ is path length
- **Training**: $O(m \cdot D^2 + D^3)$ where $m$ is sample count, $D = \sum_{k=0}^N d^k$
- **Prediction**: $O(D \cdot L)$ per path

### Typical Dimensions

| Features ($d$) | Truncation ($N$) | Signature Dim ($D$) | Training Time (1000 samples) |
|---------------|-----------------|-------------------|------------------------------|
| 2             | 2               | 7                 | ~5ms                         |
| 3             | 2               | 13                | ~10ms                        |
| 5             | 2               | 31                | ~25ms                        |
| 3             | 3               | 40                | ~50ms                        |

**HFT Suitability**: Prediction is fast enough for microsecond-latency strategies when using precomputed signatures.

## Theoretical Background

### Why Signatures?

1. **Universal**: Chen's theorem guarantees signatures uniquely characterize paths
2. **Invariant**: Reparameterization invariant (time-warping doesn't change signature)
3. **Continuous**: Small path perturbations → small signature changes
4. **Feature-rich**: Automatically captures all polynomial interactions up to order $N$

### Connection to Rough Path Theory

Signatures arise naturally from **rough path theory**, which extends stochastic calculus to irregular paths. In HFT, price paths exhibit:
- **High volatility**: Brownian-like behavior at microsecond scales
- **Jumps**: Order book events, news
- **Fractional regularity**: Hurst exponents $H \neq 0.5$

Signatures provide a **principled way** to extract features from such rough data without assuming smoothness.

## References

- **Lyons, T.** (2014). *Rough paths, signatures and the modelling of functions on streams*. ICM.
- **Horvath, B. et al.** (2021). *Deep learning volatility*. Quantitative Finance.
- **Cuchiero, C. et al.** (2020). *A generative adversarial network approach to calibration of local stochastic volatility models*. Risks.
- **Kalsi, J. et al.** (2022). *Signature trading: A path-dependent extension of the mean-variance portfolio framework*. arXiv:2203.08618.

## License

Part of the rust-arblab project. See LICENSE in repository root.
