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

# Predefined sectors, indexes, and ETF constituents
SECTORS = {
    "Technology": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
        "ORCL", "ADBE", "CRM", "ACN", "CSCO", "INTC", "AMD", "IBM",
        "QCOM", "TXN", "INTU", "NOW", "AMAT", "MU", "LRCX", "KLAC",
        "SNPS", "CDNS", "MCHP", "ADI", "NXPI", "MRVL"
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW",
        "AXP", "USB", "PNC", "TFC", "CME", "BK", "COF", "AFL",
        "MET", "PRU", "AIG", "ALL", "TRV", "PGR", "CB", "MMC"
    ],
    "Healthcare": [
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT",
        "DHR", "BMY", "AMGN", "CVS", "MDT", "GILD", "CI", "ISRG",
        "REGN", "VRTX", "ZTS", "SYK", "HUM", "BSX", "ELV", "MCK"
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "NKE", "MCD", "LOW", "SBUX", "TJX",
        "BKNG", "CMG", "TGT", "MAR", "ABNB", "GM", "F", "ROST",
        "YUM", "DHI", "ORLY", "AZO", "LEN", "BBY", "DG", "EBAY"
    ],
    "Consumer Staples": [
        "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ",
        "CL", "KMB", "GIS", "KHC", "STZ", "SYY", "HSY", "K",
        "CAG", "CPB", "TSN", "HRL", "MKC", "CHD", "CLX", "SJM"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
        "OXY", "HES", "BKR", "HAL", "KMI", "WMB", "DVN", "FANG",
        "MRO", "APA", "OKE", "TRGP", "EQT", "CTRA", "LNG", "EPD"
    ],
    "Industrials": [
        "BA", "HON", "UNP", "RTX", "UPS", "CAT", "GE", "DE",
        "LMT", "MMM", "GD", "NOC", "ETN", "CSX", "NSC", "FDX",
        "EMR", "ITW", "CARR", "PCAR", "JCI", "WM", "RSG", "OTIS"
    ],
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "DOW", "DD",
        "NUE", "PPG", "VMC", "MLM", "CTVA", "ALB", "BALL", "AVY",
        "CF", "MOS", "PKG", "IP", "EMN", "FMC", "CE", "IFF"
    ],
    "Real Estate": [
        "PLD", "AMT", "EQIX", "CCI", "PSA", "WELL", "DLR", "O",
        "VICI", "AVB", "EQR", "SPG", "SBAC", "VTR", "EXR", "INVH",
        "MAA", "ESS", "ARE", "KIM", "DOC", "UDR", "CPT", "BXP"
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL",
        "WEC", "ED", "ES", "PEG", "AWK", "DTE", "PPL", "AEE",
        "ATO", "CMS", "FE", "ETR", "EIX", "CNP", "NI", "LNT"
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS",
        "CHTR", "EA", "TTWO", "WBD", "OMC", "IPG", "NWSA", "FOXA",
        "DISH", "PARA", "LUMN", "CABO", "MTCH", "IAC", "ZI", "PINS"
    ]
}

INDEXES = {
    "S&P 500 Top 30": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
        "UNH", "JNJ", "XOM", "JPM", "V", "PG", "MA", "LLY",
        "HD", "CVX", "ABBV", "MRK", "AVGO", "COST", "PEP", "KO",
        "WMT", "TMO", "BAC", "MCD", "CSCO", "ACN"
    ],
    "Dow Jones 30": [
        "AAPL", "MSFT", "UNH", "GS", "HD", "MCD", "AMGN", "V",
        "BA", "CAT", "HON", "IBM", "JPM", "AMZN", "CRM", "JNJ",
        "CVX", "WMT", "PG", "TRV", "NKE", "AXP", "MMM", "DIS",
        "CSCO", "MRK", "KO", "INTC", "DOW", "VZ"
    ],
    "NASDAQ 100 Top 30": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
        "COST", "NFLX", "ADBE", "PEP", "AMD", "CSCO", "TMUS", "CMCSA",
        "INTC", "TXN", "QCOM", "INTU", "AMGN", "AMAT", "HON", "SBUX",
        "BKNG", "ISRG", "ADP", "GILD", "ADI", "VRTX"
    ],
    "Russell 2000 Sample": [
        "SIRI", "PLUG", "AMC", "SAVA", "JBLU", "AAL", "FCEL", "DKNG",
        "RIG", "TELL", "LCID", "WKHS", "MULN", "GEVO", "SOLO", "WIMI",
        "MARA", "RIOT", "BTBT", "CAN", "EBON", "GREE", "SOS", "IDEX"
    ]
}

ETFS = {
    "SPY (S&P 500)": ["SPY"],
    "QQQ (NASDAQ 100)": ["QQQ"],
    "DIA (Dow Jones)": ["DIA"],
    "IWM (Russell 2000)": ["IWM"],
    "Sector ETFs": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLB", "XLRE", "XLU", "XLC"],
    "Tech Giants (FANG+)": ["AAPL", "AMZN", "GOOGL", "META", "NFLX", "NVDA", "TSLA", "MSFT", "BABA", "BIDU"],
    "Crypto ETFs": ["BITO", "GBTC", "ETHE", "BITQ", "BLOK"],
    "Gold & Precious Metals": ["GLD", "SLV", "GDX", "GDXJ", "IAU", "PSLV"],
    "Bond ETFs": ["AGG", "TLT", "IEF", "BND", "LQD", "HYG", "MUB"],
    "Commodity ETFs": ["DBC", "USO", "UNG", "CORN", "WEAT", "DBA"],
    "International ETFs": ["EFA", "EEM", "VEA", "VWO", "IEFA", "IEMG"],
    "Volatility ETFs": ["VXX", "UVXY", "SVXY", "VIXY"],
    "Leveraged ETFs": ["TQQQ", "SQQQ", "UPRO", "SPXU", "TNA", "TZA"],
    "ARK Innovation ETFs": ["ARKK", "ARKQ", "ARKW", "ARKG", "ARKF", "ARKX"]
}

CRYPTO_UNIVERSES = {
    "Top 10 Crypto": [
        "BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:BNBUSDT", "BINANCE:XRPUSDT",
        "BINANCE:ADAUSDT", "BINANCE:DOGEUSDT", "BINANCE:SOLUSDT", "BINANCE:MATICUSDT",
        "BINANCE:DOTUSDT", "BINANCE:AVAXUSDT"
    ],
    "DeFi Tokens": [
        "BINANCE:UNIUSDT", "BINANCE:AAVEUSDT", "BINANCE:LINKUSDT", "BINANCE:MKRUSDT",
        "BINANCE:CRVUSDT", "BINANCE:COMPUSDT", "BINANCE:SNXUSDT", "BINANCE:SUSHIUSDT"
    ],
    "Layer 1 Blockchains": [
        "BINANCE:ETHUSDT", "BINANCE:SOLUSDT", "BINANCE:AVAXUSDT", "BINANCE:DOTUSDT",
        "BINANCE:ADAUSDT", "BINANCE:ATOMUSDT", "BINANCE:NEARUSDT", "BINANCE:ALGOUSDT"
    ],
    "Meme Coins": [
        "BINANCE:DOGEUSDT", "BINANCE:SHIBUSDT", "BINANCE:PEPEUSDT", "BINANCE:FLOKIUSDT",
        "BINANCE:BONKUSDT", "BINANCE:WIFUSDT"
    ]
}

def get_preset_symbols(category: str, name: str) -> List[str]:
    """Get symbols for a preset category"""
    if category == "Sector":
        return SECTORS.get(name, [])
    elif category == "Index":
        return INDEXES.get(name, [])
    elif category == "ETF":
        return ETFS.get(name, [])
    elif category == "Crypto":
        return CRYPTO_UNIVERSES.get(name, [])
    return []

def render():
    """Render the data loading page"""
    # Initialize session state
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'symbols' not in st.session_state:
        st.session_state.symbols = ["AAPL", "MSFT", "GOOGL"]  # Default symbols
    
    st.title("📊 Historical Data Loading")
    st.markdown("Load and preview market data for backtesting strategies")
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Data Source Configuration")
        
        # Data source selection
        data_source = st.selectbox(
            "🔌 Data Source",
            [
                "CCXT - Crypto Exchanges (FREE! ⭐)", 
                "Yahoo Finance", 
                "Finnhub (API)", 
                "Upload CSV", 
                "Mock/Synthetic"
            ],
            help="💡 CCXT is recommended for crypto - it's FREE with no API key required!"
        )
        
        # Exchange selection for CCXT
        exchange_id = 'binance'  # default
        if data_source.startswith("CCXT"):
            exchange_id = st.selectbox(
                "📊 Exchange",
                ["binance", "kraken", "coinbase", "bybit", "okx"],
                help=(
                    "• Binance: Most liquid, best for most pairs\\n"
                    "• Kraken: Reliable, regulated\\n"
                    "• Coinbase: US-based, highly regulated\\n"
                    "• Bybit: Good for perpetuals\\n"
                    "• OKX: Wide variety of altcoins"
                )
            )
            st.info(
                f"✅ Using {exchange_id.title()} - FREE public data, no API key needed!\\n"
                f"Supports second-level historical data for crypto pairs."
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
            # Preset selector
            st.markdown("#### 📋 Quick Select")
            
            preset_col1, preset_col2 = st.columns(2)
            
            with preset_col1:
                preset_category = st.selectbox(
                    "Category",
                    ["None", "Sector", "Index", "ETF", "Crypto"],
                    help="Select a predefined category"
                )
            
            with preset_col2:
                if preset_category == "Sector":
                    preset_options = list(SECTORS.keys())
                elif preset_category == "Index":
                    preset_options = list(INDEXES.keys())
                elif preset_category == "ETF":
                    preset_options = list(ETFS.keys())
                elif preset_category == "Crypto":
                    preset_options = list(CRYPTO_UNIVERSES.keys())
                else:
                    preset_options = []
                
                if preset_options:
                    preset_name = st.selectbox(
                        "Select Preset",
                        preset_options,
                        help="Choose from predefined lists"
                    )
                    
                    if st.button("📥 Load Preset", use_container_width=True):
                        preset_symbols = get_preset_symbols(preset_category, preset_name)
                        st.session_state.symbols = preset_symbols
                        st.success(f"✅ Loaded {len(preset_symbols)} symbols from {preset_name}")
                        st.rerun()
            
            st.markdown("---")
            st.markdown("#### ✏️ Manual Entry")
            
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
                    # Map UI source names to internal source names
                    source_map = {
                        'ccxt': 'ccxt',
                        'yahoo': 'yfinance',
                        'finnhub': 'finnhub',
                        'mock': 'synthetic',
                        'upload': 'synthetic'
                    }
                    source_key = data_source.lower().split()[0]
                    internal_source = source_map.get(source_key, 'auto')
                    
                    fetch_data(
                        symbols=symbols,
                        start=start_date.isoformat(),
                        end=end_date.isoformat(),
                        interval=interval,
                        source=internal_source,
                        exchange_id=exchange_id if data_source.startswith("CCXT") else None
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
            - **⭐ CCXT (Recommended)**: FREE access to 100+ crypto exchanges
              - Binance, Kraken, Coinbase, Bybit, OKX and more
              - No API key required for public data
              - Second-level historical data
              - Best for crypto trading strategies
            - **Yahoo Finance**: Free historical data (stocks & major crypto)
            - **Finnhub**: Real-time & historical via API (requires key)
            - **CSV Upload**: Custom data files
            - **Mock**: Synthetic data for testing
            """)
    
    st.markdown("---")
    
    # Data preview and visualization
    if st.session_state.historical_data is not None:
        display_data_preview()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbols: List[str], start: str, end: str, interval: str, source: str, exchange_id: Optional[str] = None) -> pd.DataFrame:
    """Fetch data with caching"""
    display_source = f"{source} ({exchange_id})" if exchange_id else source
    with st.spinner(f"Fetching data for {len(symbols)} symbols from {display_source}..."):
        try:
            # For CCXT, pass exchange_id through params
            if source == 'ccxt' and exchange_id:
                from python.data_fetcher import _fetch_ccxt
                df = _fetch_ccxt(symbols, start, end, interval, exchange_id)
            else:
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
