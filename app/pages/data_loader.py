"""
Data Loading Module
===================

Load historical market data from multiple sources:
- Finnhub (primary)
- Yahoo Finance (fallback)
- CSV upload (custom data)
- Mock/Synthetic data (testing)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from python.data_fetcher import fetch_intraday_data, get_close_prices, get_universe_symbols
from python.rust_bridge import list_connectors, get_connector

def render():
    """Render the data loading page"""
    st.title("📊 Historical Data Loading")
    st.markdown("Load and preview market data for backtesting strategies")
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Data Source Configuration")
        
        # Data source selection
        data_source = st.selectbox(
            "Data Source",
            ["Finnhub (API)", "Yahoo Finance", "Upload CSV", "Mock/Synthetic"],
            help="Choose where to fetch market data from"
        )
        
        # Symbol selection
        if data_source == "Upload CSV":
            st.info("📁 Upload a CSV file with columns: timestamp, symbol, open, high, low, close, volume")
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    st.session_state.historical_data = df
                    st.success(f"✅ Loaded {len(df)} rows from CSV")
                    symbols = df['symbol'].unique().tolist()
                    st.session_state.symbols = symbols
                except Exception as e:
                    st.error(f"Failed to load CSV: {e}")
        else:
            # Symbol input
            symbols_input = st.text_area(
                "Symbols (one per line or comma-separated)",
                value="\n".join(st.session_state.symbols),
                height=100,
                help="Enter stock/crypto symbols to fetch data for"
            )
            
            # Parse symbols
            symbols = []
            for line in symbols_input.split('\n'):
                symbols.extend([s.strip().upper() for s in line.split(',') if s.strip()])
            symbols = list(set(symbols))  # Remove duplicates
            
            # Date range
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365),
                    max_value=datetime.now()
                )
            with col_date2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    max_value=datetime.now()
                )
            
            # Interval selection
            interval = st.selectbox(
                "Data Interval",
                ["1m", "5m", "15m", "30m", "1h", "1d"],
                index=4,  # Default to 1h
                help="Time interval for OHLCV data"
            )
            
            # Fetch button
            if st.button("🔄 Fetch Data", type="primary", use_container_width=True):
                if not symbols:
                    st.error("Please enter at least one symbol")
                else:
                    fetch_data(
                        symbols=symbols,
                        start=start_date.isoformat(),
                        end=end_date.isoformat(),
                        interval=interval,
                        source=data_source.lower().split()[0]  # Extract first word
                    )
    
    with col2:
        st.markdown("### Quick Info")
        
        if st.session_state.historical_data is not None:
            df = st.session_state.historical_data
            
            # Show stats
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Symbols", len(df['symbol'].unique()))
            
            if 'timestamp' in df.columns:
                date_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
                st.metric("Date Range", date_range)
            
            # Data quality
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.2f}%")
            
            # Clear data button
            if st.button("🗑️ Clear Data", use_container_width=True):
                st.session_state.historical_data = None
                st.rerun()
        else:
            st.info("No data loaded yet")
            st.markdown("""
            **Data sources:**
            - **Finnhub**: Real-time & historical market data via API
            - **Yahoo Finance**: Free historical data (limited intraday)
            - **CSV Upload**: Custom data files
            - **Mock**: Synthetic data for testing
            """)
    
    st.markdown("---")
    
    # Data preview and visualization
    if st.session_state.historical_data is not None:
        display_data_preview()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbols: List[str], start: str, end: str, interval: str, source: str) -> pd.DataFrame:
    """Fetch data with caching"""
    with st.spinner(f"Fetching data for {len(symbols)} symbols from {source}..."):
        try:
            df = fetch_intraday_data(
                symbols=symbols,
                start=start,
                end=end,
                interval=interval,
                source=source
            )
            
            # Reset index to make it easier to work with
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            
            st.session_state.historical_data = df
            st.session_state.symbols = symbols
            st.session_state.data_source = source
            st.session_state.date_range = (start, end)
            
            st.success(f"✅ Successfully loaded {len(df):,} records for {len(symbols)} symbols")
            return df
            
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return pd.DataFrame()

def display_data_preview():
    """Display data preview and visualization"""
    df = st.session_state.historical_data
    
    st.markdown("### 📋 Data Preview & Visualization")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📑 Data Table", "📈 Statistics", "💾 Export"])
    
    with tab1:
        # Symbol selector for charting
        symbols = df['symbol'].unique().tolist()
        selected_symbol = st.selectbox("Select Symbol for Chart", symbols)
        
        # Filter data for selected symbol
        symbol_df = df[df['symbol'] == selected_symbol].copy()
        if 'timestamp' in symbol_df.columns:
            symbol_df = symbol_df.sort_values('timestamp')
        
        # Create OHLC candlestick chart
        if all(col in symbol_df.columns for col in ['open', 'high', 'low', 'close']):
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{selected_symbol} Price', 'Volume')
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=symbol_df['timestamp'],
                    open=symbol_df['open'],
                    high=symbol_df['high'],
                    low=symbol_df['low'],
                    close=symbol_df['close'],
                    name='OHLC'
                ),
                row=1, col=1
            )
            
            # Volume bars
            if 'volume' in symbol_df.columns:
                colors = ['red' if close < open else 'green' 
                         for close, open in zip(symbol_df['close'], symbol_df['open'])]
                
                fig.add_trace(
                    go.Bar(
                        x=symbol_df['timestamp'],
                        y=symbol_df['volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.5
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=600,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Missing required columns for candlestick chart")
    
    with tab2:
        st.markdown("#### Raw Data Table")
        
        # Filtering options
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_symbol = st.multiselect("Filter by Symbol", df['symbol'].unique())
        with col2:
            n_rows = st.number_input("Number of rows", min_value=10, max_value=10000, value=100)
        with col3:
            sort_order = st.selectbox("Sort by", ["timestamp", "symbol", "close"])
        
        # Apply filters
        display_df = df.copy()
        if filter_symbol:
            display_df = display_df[display_df['symbol'].isin(filter_symbol)]
        
        display_df = display_df.sort_values(sort_order, ascending=False).head(n_rows)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
    
    with tab3:
        st.markdown("#### Statistical Summary")
        
        # Per-symbol statistics
        for symbol in df['symbol'].unique()[:5]:  # Show first 5 symbols
            symbol_df = df[df['symbol'] == symbol]
            
            with st.expander(f"📊 {symbol}", expanded=False):
                cols = st.columns(4)
                
                if 'close' in symbol_df.columns:
                    with cols[0]:
                        st.metric("Mean Price", f"${symbol_df['close'].mean():.2f}")
                    with cols[1]:
                        st.metric("Std Dev", f"${symbol_df['close'].std():.2f}")
                    with cols[2]:
                        st.metric("Min", f"${symbol_df['close'].min():.2f}")
                    with cols[3]:
                        st.metric("Max", f"${symbol_df['close'].max():.2f}")
                
                # Returns analysis
                if 'close' in symbol_df.columns and len(symbol_df) > 1:
                    returns = symbol_df['close'].pct_change().dropna()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Return", f"{returns.mean()*100:.3f}%")
                    with col2:
                        st.metric("Return Volatility", f"{returns.std()*100:.3f}%")
                    with col3:
                        sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0
                        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
    
    with tab4:
        st.markdown("#### Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export to CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export to Parquet (more efficient)
            try:
                parquet_buffer = df.to_parquet(index=False)
                st.download_button(
                    label="📥 Download as Parquet",
                    data=parquet_buffer,
                    file_name=f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Parquet export not available: {e}")
        
        st.info("💡 Tip: Parquet format is more efficient for large datasets")

# Execute the render function when page is loaded
if __name__ == "__main__":
    render()
