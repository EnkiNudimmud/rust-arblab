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
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.data_fetcher import fetch_intraday_data, get_close_prices, get_universe_symbols
from python.rust_bridge import list_connectors, get_connector
from utils.ui_components import render_sidebar_navigation, apply_custom_css
from utils.data_persistence import (
    save_dataset, load_dataset, load_all_datasets, delete_dataset,
    list_datasets, get_storage_stats, merge_datasets
)

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
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "SOL/USDT", "MATIC/USDT",
        "DOT/USDT", "AVAX/USDT"
    ],
    "DeFi Tokens": [
        "UNI/USDT", "AAVE/USDT", "LINK/USDT", "MKR/USDT",
        "CRV/USDT", "COMP/USDT", "SNX/USDT", "SUSHI/USDT"
    ],
    "Layer 1 Blockchains": [
        "ETH/USDT", "SOL/USDT", "AVAX/USDT", "DOT/USDT",
        "ADA/USDT", "ATOM/USDT", "NEAR/USDT", "ALGO/USDT"
    ],
    "Meme Coins": [
        "DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "FLOKI/USDT",
        "BONK/USDT", "WIF/USDT"
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

# Set page config
st.set_page_config(page_title="Data Loader", page_icon="💾", layout="wide")

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Data Loader")
apply_custom_css()

def render():
    """Render the data loading page"""
    # Initialize session state
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'dark'
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'symbols' not in st.session_state:
        st.session_state.symbols = ["AAPL", "MSFT", "GOOGL"]  # Default symbols
    
    st.title("📊 Historical Data Loading")
    st.markdown("Load and preview market data for backtesting strategies")
    
    # Quick guide
    with st.expander("💡 Quick Start Guide", expanded=False):
        col_guide1, col_guide2 = st.columns(2)
        
        with col_guide1:
            st.markdown("""
            **📈 For Stocks (US Equities):**
            - ✅ Use **Yahoo Finance**
            - Examples: AAPL, GOOGL, MSFT
            - Use **Sector** or **Index** presets
            """)
        
        with col_guide2:
            st.markdown("""
            **₿ For Cryptocurrencies:**
            - ✅ Use **CCXT - Crypto Exchanges**
            - Examples: BTC/USDT, ETH/USDT, SOL/USDT
            - Use **Crypto** presets
            """)
    
    st.markdown("---")
    
    # Persisted Datasets Section
    with st.expander("💾 Persisted Datasets", expanded=False):
        st.markdown("### Manage Saved Datasets")
        
        # Get storage stats
        stats = get_storage_stats()
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("📦 Datasets", stats['total_datasets'])
        with col_stat2:
            st.metric("📊 Total Rows", f"{stats['total_rows']:,}")
        with col_stat3:
            st.metric("🏷️ Symbols", stats['total_symbols'])
        with col_stat4:
            st.metric("💽 Storage", f"{stats['total_size_mb']} MB")
        
        st.markdown("---")
        
        # List persisted datasets
        datasets = list_datasets()
        
        if datasets:
            st.markdown("#### Available Datasets")
            
            for dataset in datasets:
                with st.container():
                    col_ds1, col_ds2, col_ds3 = st.columns([3, 1, 1])
                    
                    with col_ds1:
                        st.markdown(f"**{dataset['name']}**")
                        st.caption(
                            f"Source: {dataset['source']} | "
                            f"Symbols: {len(dataset['symbols'])} | "
                            f"Rows: {dataset['row_count']:,} | "
                            f"Updated: {dataset['last_updated'][:10]}"
                        )
                    
                    with col_ds2:
                        if st.button("📂 Load", key=f"load_{dataset['name']}", use_container_width=True):
                            df = load_dataset(dataset['name'])
                            if df is not None:
                                st.session_state.historical_data = df
                                st.session_state.symbols = dataset['symbols']
                                st.success(f"✅ Loaded {dataset['name']}")
                                st.rerun()
                    
                    with col_ds3:
                        if st.button("🗑️ Delete", key=f"delete_{dataset['name']}", use_container_width=True):
                            if delete_dataset(dataset['name']):
                                st.success(f"✅ Deleted {dataset['name']}")
                                st.rerun()
                    
                    st.divider()
        else:
            st.info("📭 No saved datasets yet. Fetch data below to save it automatically!")
    
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
            
            # Show recommended categories based on data source
            if data_source.startswith("CCXT"):
                st.info("💡 **Recommended:** Use 'Crypto' category for crypto exchanges")
                recommended_categories = ["None", "Crypto", "Sector", "Index", "ETF"]
            else:
                st.info("💡 **Recommended:** Use 'Sector' or 'Index' for stocks")
                recommended_categories = ["None", "Sector", "Index", "ETF", "Crypto"]
            
            preset_col1, preset_col2 = st.columns(2)
            
            with preset_col1:
                preset_category = st.selectbox(
                    "Category",
                    recommended_categories,
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
                    
                    preset_col_btn1, preset_col_btn2 = st.columns(2)
                    
                    with preset_col_btn1:
                        if st.button("➕ Append", use_container_width=True, help="Add to existing symbols"):
                            preset_symbols = get_preset_symbols(preset_category, preset_name)
                            # Append and remove duplicates
                            current_symbols = st.session_state.symbols if st.session_state.symbols else []
                            combined = list(set(current_symbols + preset_symbols))
                            st.session_state.symbols = combined
                            new_count = len(combined) - len(current_symbols)
                            st.success(f"✅ Added {new_count} new symbols (Total: {len(combined)})")
                            st.rerun()
                    
                    with preset_col_btn2:
                        if st.button("🔄 Replace", use_container_width=True, help="Replace all symbols"):
                            preset_symbols = get_preset_symbols(preset_category, preset_name)
                            st.session_state.symbols = preset_symbols
                            st.success(f"✅ Replaced with {len(preset_symbols)} symbols from {preset_name}")
                            st.rerun()
            
            st.markdown("---")
            st.markdown("#### ✏️ Manual Entry")
            
            # Dynamic help text based on data source
            if data_source.startswith("CCXT"):
                symbol_help = (
                    "Enter crypto pairs (e.g., BTC/USDT, ETH/USDT)\\n"
                    "Format: BASE/QUOTE (e.g., BTC/USDT, ETH/BTC)"
                )
                placeholder_text = "BTC/USDT\nETH/USDT\nSOL/USDT"
            else:
                symbol_help = "Enter stock symbols (e.g., AAPL, GOOGL, MSFT)"
                placeholder_text = "AAPL\nMSFT\nGOOGL"
            
            # Symbol management buttons
            symbol_btn_col1, symbol_btn_col2 = st.columns([3, 1])
            
            with symbol_btn_col2:
                if st.button("🗑️ Clear All", use_container_width=True, help="Clear all symbols"):
                    st.session_state.symbols = []
                    st.rerun()
            
            # Symbol input
            with symbol_btn_col1:
                symbols_input = st.text_area(
                    "Symbols (one per line or comma-separated)",
                    value="\n".join(st.session_state.symbols) if st.session_state.symbols else "",
                    placeholder=placeholder_text,
                    height=100,
                    help=symbol_help,
                    label_visibility="visible"
                )
            
            # Parse and clean symbols
            symbols = []
            for line in symbols_input.split('\n'):
                for s in line.split(','):
                    s = s.strip().upper()
                    # Remove extra spaces within symbol (e.g., "DO GE" -> "DOGE")
                    s = s.replace(' ', '')
                    if s:
                        symbols.append(s)
            symbols = list(set(symbols))  # Remove duplicates
            
            # Show cleaned symbols if any were modified
            original_symbols = []
            for line in symbols_input.split('\n'):
                original_symbols.extend([s.strip().upper() for s in line.split(',') if s.strip()])
            if any(' ' in s for s in original_symbols):
                st.info(f"ℹ️  Cleaned symbols: removed spaces from {len([s for s in original_symbols if ' ' in s])} symbol(s)")
            
            # Interval selection (moved before date range for smart defaults)
            interval = st.selectbox(
                "Data Interval",
                ["1m", "5m", "15m", "30m", "1h", "1d"],
                index=4,  # Default to 1h
                help="Time interval for OHLCV data"
            )
            
            # Smart default date range based on interval and data source
            if data_source == "Yahoo Finance":
                if interval == "1m":
                    default_days = 5  # Yahoo Finance limit: 7 days
                    st.caption("⚠️ Yahoo Finance: 1m data limited to last 7 days")
                elif interval in ["5m", "15m", "30m"]:
                    default_days = 30  # Yahoo Finance limit: 60 days
                    st.caption("ℹ️ Yahoo Finance: Intraday data limited to last 60 days")
                elif interval == "1h":
                    default_days = 90
                else:  # 1d
                    default_days = 365
            else:
                # CCXT and other sources have more flexible limits
                if interval in ["1m", "5m"]:
                    default_days = 30
                elif interval in ["15m", "30m", "1h"]:
                    default_days = 90
                else:
                    default_days = 365
            
            # Date range
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=default_days),
                    max_value=datetime.now()
                )
            with col_date2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    max_value=datetime.now()
                )
            
            # Validation warning for CCXT with stock symbols
            if data_source.startswith("CCXT"):
                # Check if symbols look like stock tickers (common patterns)
                stock_like_symbols = [s for s in symbols if len(s) <= 5 and s.isalpha() and '/' not in s]
                crypto_like_symbols = [s for s in symbols if '/' in s or s in ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'ADA', 'DOT', 'MATIC', 'AVAX', 'LINK']]
                
                if stock_like_symbols and not crypto_like_symbols:
                    st.warning(
                        f"⚠️ **Warning:** You selected CCXT (crypto exchange) but your symbols look like stocks: {', '.join(stock_like_symbols[:5])}{'...' if len(stock_like_symbols) > 5 else ''}\\n\\n"
                        f"**Crypto exchanges don't have stock data!**\\n\\n"
                        f"**Options:**\\n"
                        f"1. Use **Yahoo Finance** source for stocks (AAPL, GOOGL, etc.)\\n"
                        f"2. Or use crypto pairs like: BTC/USDT, ETH/USDT, SOL/USDT\\n"
                        f"3. Or switch to 'Crypto' category in Quick Select"
                    )
            
            # Validation warning for Yahoo Finance with crypto pairs
            if data_source == "Yahoo Finance":
                crypto_pairs = [s for s in symbols if '/' in s]
                if crypto_pairs:
                    st.warning(
                        f"⚠️ **Warning:** You selected Yahoo Finance but have crypto pair format: {', '.join(crypto_pairs[:3])}\\n\\n"
                        f"**Yahoo Finance uses different format for crypto:**\\n"
                        f"• Use 'BTC-USD' instead of 'BTC/USDT'\\n"
                        f"• Or switch to 'CCXT - Crypto Exchanges' for crypto pairs"
                    )
                
                # Check date range for intraday data
                days_diff = (end_date - start_date).days
                
                if interval == "1m" and days_diff > 7:
                    st.error(
                        f"❌ **Invalid Date Range for 1m interval**\\n\\n"
                        f"Yahoo Finance **1-minute data** is limited to the **last 7 days only**.\\n"
                        f"Your range: {days_diff} days\\n\\n"
                        f"**Solutions:**\\n"
                        f"1. Reduce date range to last 7 days\\n"
                        f"2. Use **5m** or higher interval for longer history\\n"
                        f"3. Use **CCXT** for crypto (supports longer 1m history)"
                    )
                elif interval in ["5m", "15m", "30m"] and days_diff > 60:
                    st.error(
                        f"❌ **Invalid Date Range for {interval} interval**\\n\\n"
                        f"Yahoo Finance **{interval} data** is limited to the **last 60 days only**.\\n"
                        f"Your range: {days_diff} days\\n\\n"
                        f"**Solutions:**\\n"
                        f"1. Reduce date range to last 60 days\\n"
                        f"2. Use **1h** or **1d** interval for longer history\\n"
                        f"3. Use **CCXT** for crypto (supports longer history)"
                    )
                elif interval in ["1m", "5m", "15m", "30m"]:
                    # Show helpful info for valid ranges
                    max_days = 7 if interval == "1m" else 60
                    if days_diff > max_days * 0.8:  # Warn if approaching limit
                        st.info(
                            f"ℹ️  **Note:** You're requesting {days_diff} days of {interval} data.\\n"
                            f"Yahoo Finance limit is {max_days} days for this interval.\\n"
                            f"Consider using **1h** or **1d** for longer historical analysis."
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
def fetch_data(symbols: List[str], start: str, end: str, interval: str, source: str, exchange_id: Optional[str] = None, save_mode: str = "append") -> pd.DataFrame:
    """Fetch data with caching and persistence"""
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
            
            # Auto-save to persistent storage
            dataset_name = f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if exchange_id:
                dataset_name = f"{exchange_id}_{dataset_name}"
            
            append_mode = (save_mode == "append")
            if save_dataset(df, dataset_name, symbols, display_source, (start, end), append=append_mode):
                save_msg = "appended to existing dataset" if append_mode else "saved"
                st.info(f"💾 Data {save_msg} as '{dataset_name}'")
            
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
        st.markdown("#### 💾 Save & Export")
        
        # Save to persistent storage
        st.markdown("**Save to Persistent Storage**")
        
        col_save1, col_save2 = st.columns([2, 1])
        
        with col_save1:
            dataset_name = st.text_input(
                "Dataset Name",
                value=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Name for the saved dataset (will be stored in data/persisted/)"
            )
        
        with col_save2:
            save_mode = st.selectbox(
                "Save Mode",
                ["Create New", "Append to Existing"],
                help="Create new dataset or append to existing one with same name"
            )
        
        col_savebtn1, col_savebtn2 = st.columns(2)
        
        with col_savebtn1:
            if st.button("💾 Save Dataset", use_container_width=True, type="primary"):
                append_mode = (save_mode == "Append to Existing")
                symbols = df['symbol'].unique().tolist() if 'symbol' in df.columns else st.session_state.get('symbols', [])
                source = st.session_state.get('data_source', 'Unknown')
                date_range = st.session_state.get('date_range')
                
                if save_dataset(df, dataset_name, symbols, source, date_range, append=append_mode):
                    st.success(f"✅ Dataset saved as '{dataset_name}'")
                    st.balloons()
                else:
                    st.error("❌ Failed to save dataset")
        
        with col_savebtn2:
            if st.button("🗑️ Clear Session Data", use_container_width=True):
                st.session_state.historical_data = None
                st.session_state.symbols = []
                st.success("✅ Session data cleared")
                st.rerun()
        
        st.markdown("---")
        
        # Export downloads
        st.markdown("**Download Files**")
        
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
        
        st.info("💡 Tip: Saved datasets persist across sessions and are accessible via Docker volumes")

# Execute the render function when page is loaded
if __name__ == "__main__":
    render()
