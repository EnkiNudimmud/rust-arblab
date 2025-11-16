import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

# make repo importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.meanrev import fetch_price_series, compute_log_returns, pca_portfolios, backtest_portfolio, RUST_AVAILABLE

st.set_page_config(page_title="Mean-Reverting Vector Portfolios", layout="wide")
st.title("Mean-Reverting Vector Portfolios — Analysis & Visualization")

# Show acceleration status
if RUST_AVAILABLE:
    st.success("⚡ Rust acceleration enabled — using high-performance compiled functions")
else:
    st.warning("Using pure Python fallback — build rust_connector for 10-100x speedup")

with st.sidebar:
    st.header("Data & Parameters")
    connector_name = st.selectbox("Connector", ["finnhub", "mock"], index=0)
    symbols_text = st.text_area("Symbols (comma separated)", value="AAPL,MSFT,GOOGL,AMZN,FB")
    start = st.date_input("Start date", value=(datetime.utcnow() - timedelta(days=365)).date())
    end = st.date_input("End date", value=datetime.utcnow().date())
    freq = st.selectbox("Frequency", ["1D", "1H"], index=0)
    n_components = st.slider("PCA components", 1, 10, 3)
    entry_z = st.number_input("Entry z-score", value=1.5)
    exit_z = st.number_input("Exit z-score", value=0.5)
    run = st.button("Run Analysis")

if run:
    symbols = [s.strip() for s in symbols_text.split(',') if s.strip()]
    st.info(f"Fetching data for {len(symbols)} symbols via {connector_name}...")
    # simple connector resolution
    connector = None
    try:
        from python.rust_bridge import get_connector
        connector = get_connector(connector_name)
    except Exception:
        connector = None

    price_df = pd.DataFrame()
    for s in symbols:
        ser = fetch_price_series(connector, s, start.isoformat(), end.isoformat(), freq=freq)
        price_df[s] = ser

    st.success("Data fetched")
    st.subheader("Price matrix (tail)")
    st.dataframe(price_df.tail(10))

    st.subheader("PCA discovery of portfolios")
    returns = compute_log_returns(price_df)
    comps, pca_info = pca_portfolios(returns, n_components=n_components)

    for i, comp in enumerate(comps):
        exp_var = pca_info.get("explained_variance_ratio_", [0.0] * len(comps))
        st.markdown(f"**PC {i+1} (explained var {exp_var[i]:.3f})**")
        weights = pd.Series(comp, index=returns.columns)
        st.bar_chart(weights)

    st.subheader("Backtest a selected PCA portfolio")
    sel = st.number_input("Select PC index", min_value=1, max_value=len(comps), value=1)
    w = comps[sel-1]
    res = backtest_portfolio(w, price_df, entry_z=entry_z, exit_z=exit_z)

    st.write(f"Sharpe: {res['sharpe']:.3f}")
    st.subheader("Equity curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['equity'].index, y=res['equity'].values, name='Equity'))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Z-score (last 200)")
    st.line_chart(res['z'].tail(200))

    st.markdown("---")
    st.info("This is an initial, Python-based implementation. For performance at scale, we can port the heavy computations to Rust and expose them via the Rust connector.")
