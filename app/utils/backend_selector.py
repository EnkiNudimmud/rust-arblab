"""
Backend Selector Component for Streamlit

Allows users to switch between Legacy and gRPC backends
and compare their performance
"""

import streamlit as st
import time
import numpy as np
from typing import Dict, Any, Callable
import pandas as pd

from app.utils.backend_config import BackendType, BackendConfig, get_backend_config, set_backend_config
from app.utils.backend_interface import get_backend, clear_backend_cache


def render_backend_selector():
    """Render backend selection UI in sidebar"""
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Backend Selection")
    
    config = get_backend_config()
    
    # Check backend availability
    grpc_backend = get_backend(BackendType.GRPC)
    legacy_backend = get_backend(BackendType.LEGACY)
    
    grpc_available = grpc_backend.is_available()
    legacy_available = legacy_backend.is_available()
    
    # Backend selection
    backend_options = []
    if grpc_available:
        backend_options.append(("🚀 gRPC (Ultra-Fast)", BackendType.GRPC))
    if legacy_available:
        backend_options.append(("🐌 Legacy PyO3", BackendType.LEGACY))
    
    if not backend_options:
        st.sidebar.error("⚠️ No backends available!")
        return
    
    # Current selection
    current_idx = 0
    for i, (_, btype) in enumerate(backend_options):
        if btype == config.backend_type:
            current_idx = i
            break
    
    # Selection widget
    selected_label, selected_type = backend_options[
        st.sidebar.selectbox(
            "Active Backend",
            range(len(backend_options)),
            index=current_idx,
            format_func=lambda i: backend_options[i][0]
        )
    ]
    
    # Update config if changed
    if selected_type != config.backend_type:
        config.backend_type = selected_type
        set_backend_config(config)
        clear_backend_cache()
        st.rerun()
    
    # Display status
    backend = get_backend()
    if selected_type == BackendType.GRPC:
        st.sidebar.success(f"✅ Connected to gRPC server\n`{config.grpc_host}:{config.grpc_port}`")
    else:
        st.sidebar.info("✅ Using Legacy PyO3 bindings")
    
    # Performance comparison button
    if grpc_available and legacy_available:
        if st.sidebar.button("📊 Compare Backends", use_container_width=True):
            st.session_state.show_backend_comparison = True


def render_backend_comparison():
    """Render backend performance comparison"""
    
    if not st.session_state.get('show_backend_comparison', False):
        return
    
    st.markdown("---")
    st.markdown("### 📊 Backend Performance Comparison")
    
    with st.spinner("Running performance benchmarks..."):
        results = run_backend_benchmark()
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🐌 Legacy PyO3")
        st.metric("Average Latency", f"{results['legacy']['avg_latency']:.3f} ms")
        st.metric("Throughput", f"{results['legacy']['throughput']:.0f} ops/s")
    
    with col2:
        st.markdown("#### 🚀 gRPC")
        st.metric("Average Latency", f"{results['grpc']['avg_latency']:.3f} ms")
        st.metric("Throughput", f"{results['grpc']['throughput']:.0f} ops/s")
    
    # Speedup metrics
    speedup_latency = results['legacy']['avg_latency'] / results['grpc']['avg_latency']
    speedup_throughput = results['grpc']['throughput'] / results['legacy']['throughput']
    
    st.success(f"""
    **⚡ Performance Gains:**
    - **{speedup_latency:.1f}x** faster latency
    - **{speedup_throughput:.1f}x** higher throughput
    """)
    
    # Detailed results table
    st.markdown("#### Detailed Results")
    
    df = pd.DataFrame({
        'Operation': results['operations'],
        'Legacy (ms)': results['legacy']['latencies'],
        'gRPC (ms)': results['grpc']['latencies'],
        'Speedup': [l / g for l, g in zip(results['legacy']['latencies'], results['grpc']['latencies'])]
    })
    
    st.dataframe(df, use_container_width=True)
    
    # Bar chart comparison
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[
        go.Bar(name='Legacy', x=results['operations'], y=results['legacy']['latencies'], marker_color='#FF6B6B'),
        go.Bar(name='gRPC', x=results['operations'], y=results['grpc']['latencies'], marker_color='#4ECDC4')
    ])
    
    fig.update_layout(
        title='Latency Comparison (Lower is Better)',
        xaxis_title='Operation',
        yaxis_title='Latency (ms)',
        barmode='group',
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Close button
    if st.button("Close Comparison"):
        st.session_state.show_backend_comparison = False
        st.rerun()


def run_backend_benchmark() -> Dict[str, Any]:
    """Run performance benchmark on both backends"""
    
    # Test data
    np.random.seed(42)
    prices_simple = (np.random.randn(100).cumsum() + 100).tolist()
    
    prices_multi = [
        {'symbol': f'ASSET_{i}', 'prices': (np.random.randn(100).cumsum() + 100).tolist()}
        for i in range(5)
    ]
    
    observations = np.random.randn(100).tolist()
    
    # Test operations
    operations = [
        ('Mean Reversion', lambda backend: backend.calculate_mean_reversion(prices_simple, lookback=20, threshold=1.5)),
        ('Portfolio Optimization', lambda backend: backend.optimize_portfolio(prices_multi, method='max_sharpe')),
        ('HMM Regime Detection', lambda backend: backend.run_hmm(observations, n_states=2, max_iterations=20)),
        ('Sparse Portfolio', lambda backend: backend.calculate_sparse_portfolio(prices_multi, lambda_param=0.5)),
    ]
    
    results = {
        'operations': [op[0] for op in operations],
        'legacy': {'latencies': [], 'avg_latency': 0, 'throughput': 0},
        'grpc': {'latencies': [], 'avg_latency': 0, 'throughput': 0}
    }
    
    # Benchmark each backend
    for backend_type in [BackendType.LEGACY, BackendType.GRPC]:
        backend = get_backend(backend_type)
        
        if not backend.is_available():
            # Fill with zeros if not available
            results[backend_type.value]['latencies'] = [0.0] * len(operations)
            continue
        
        total_time = 0
        n_runs = 50  # Reduced for UI responsiveness
        
        for op_name, op_func in operations:
            latencies = []
            
            # Warm up
            for _ in range(5):
                try:
                    op_func(backend)
                except:
                    pass
            
            # Actual benchmark
            for _ in range(n_runs):
                start = time.time()
                try:
                    op_func(backend)
                except Exception as e:
                    pass  # Handle errors gracefully
                latencies.append((time.time() - start) * 1000)
            
            avg_latency = np.mean(latencies)
            results[backend_type.value]['latencies'].append(avg_latency)
            total_time += np.sum(latencies) / 1000
        
        # Calculate aggregate metrics
        results[backend_type.value]['avg_latency'] = np.mean(results[backend_type.value]['latencies'])
        results[backend_type.value]['throughput'] = (n_runs * len(operations)) / total_time if total_time > 0 else 0
    
    return results


def benchmark_operation(operation_name: str, func: Callable, backend: Any, n_runs: int = 100) -> float:
    """Benchmark a single operation"""
    latencies = []
    
    for _ in range(n_runs):
        start = time.time()
        try:
            func(backend)
        except Exception as e:
            # Handle errors but continue
            pass
        latencies.append(time.time() - start)
    
    return float(np.mean(latencies) * 1000)  # Convert to ms
