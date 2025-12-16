"""
Data Loading Module
===================

Load historical market data from multiple sources:
- CCXT (crypto exchanges - FREE!)
- Yahoo Finance (stocks & ETFs - FREE!)
- Finnhub (stocks/forex - API key configured)
- Alpha Vantage (stocks/forex/crypto - API key configured, 25 calls/day)
- Massive (institutional data - API key configured, 100 calls/day)
- CSV upload (custom data)
- Mock/Synthetic data (testing)

All API keys are configured in api_keys.propertiesFeatures:
- Stackable data loading (append new queries to existing data)
- Persistent storage to /data folder for long-living sessions
- Rate-limit aware fetching for intraday data
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.grpc_wrapper import TradingGrpcClient, GrpcConfig
from utils.data_persistence import (
    save_dataset, load_dataset, list_datasets, delete_dataset, load_all_datasets,
    get_storage_stats, merge_datasets
)
from utils.ui_components import render_sidebar_navigation, apply_custom_css

# Helper functions for missing imports
def get_total_storage_size():
    """Calculate total storage size used by datasets"""
    try:
        import os
        from pathlib import Path
        
        data_dir = Path(__file__).parent.parent.parent / "data" / "persisted"
        total_size = 0
        if data_dir.exists():
            for file_path in data_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return total_size
    except Exception:
        return 0

def format_size(size_bytes):
    """Format size in bytes to human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def stack_data(existing_df, new_df, mode):
    """Stack data based on mode (append or update)"""
    if existing_df is None or existing_df.empty:
        return new_df
    
    if new_df is None or new_df.empty:
        return existing_df
    
    # Combine dataframes
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates based on symbol and timestamp if available
    if 'symbol' in combined_df.columns and 'timestamp' in combined_df.columns:
        if mode == "update":
            combined_df = combined_df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')
        else:  # append mode
            combined_df = combined_df.drop_duplicates(subset=['symbol', 'timestamp'], keep='first')
    
    return combined_df

def generate_dataset_name(symbols, interval, source):
    """Generate a dataset name based on parameters"""
    symbol_str = "_".join(symbols[:3])  # Use first 3 symbols
    if len(symbols) > 3:
        symbol_str += "_etc"
    return f"{source}_{interval}_{symbol_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

def get_smart_source(symbol: str, intervals: List[str]) -> str:
    """Determine the best data source for a symbol based on type and intervals"""
    is_crypto = '/' in symbol

    if is_crypto:
        return 'ccxt'  # Crypto exchanges for crypto pairs
    else:
        # For stocks, use Alpaca if 1s interval is selected (only Alpaca supports 1s for stocks)
        if '1s' in intervals:
            return 'alpaca'
        else:
            return 'yfinance'  # Yahoo Finance for other intervals (free and reliable)

def sync_symbols_input():
    """Sync the symbols input text area with session state symbols"""
    if 'symbols' in st.session_state:
        # Convert symbols list to text format (one per line)
        st.session_state.symbols_input = "\n".join(st.session_state.symbols) if st.session_state.symbols else ""

def parse_symbols_from_input(symbols_input: str) -> List[str]:
    """Parse symbols from input text (supports both comma-separated and line-separated)"""
    symbols = []
    for line in symbols_input.split('\n'):
        for s in line.split(','):
            s = s.strip().upper()
            # Remove extra spaces within symbol (e.g., "DO GE" -> "DOGE")
            s = s.replace(' ', '')
            if s:
                symbols.append(s)
    return list(set(symbols))  # Remove duplicates

# Set page config
st.set_page_config(page_title="Data Loader", page_icon="💾", layout="wide")

# Render sidebar navigation and apply CSS
render_sidebar_navigation(current_page="Data Loader")
apply_custom_css()

def render():
    """Render the data loading page"""
    # Initialize session state
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = 'light'
    if 'data_load_mode' not in st.session_state:
        st.session_state.data_load_mode = "replace"  # 'replace', 'append', 'update'
    
    # Initialize data state for clean start
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'symbols' not in st.session_state:
        st.session_state.symbols = []
    if 'symbol_input_value' not in st.session_state:
        st.session_state.symbol_input_value = ""
    
    # Do NOT auto-load data on Data Loader page - let user start fresh
    # Users can explicitly load data from saved datasets tab if needed
    
    st.title("📊 Historical Data Loading")
    st.markdown("Load and preview market data for backtesting strategies")
    
    # Main tabs for different sections
    main_tab_auto, main_tab1, main_tab2, main_tab3 = st.tabs(["✨ Auto Smart Fetch", "📥 Manual Fetch", "💾 Saved Datasets", "🔗 Merge/Append"])
    
    with main_tab_auto:
        render_auto_fetch_tab()

    with main_tab1:
        render_fetch_tab()
    
    with main_tab2:
        render_saved_datasets_tab()
    
    with main_tab3:
        render_merge_append_tab()



def render_auto_fetch_tab():
    """Render the Auto Smart Fetch tab"""
    st.markdown("### ✨ Auto Smart Fetch")
    st.markdown("""
    Automatically selects the best data source (Alpaca, Yahoo, CCXT) based on:
    - **Symbol Type** (Crypto vs Stock)
    - **Interval** (1s/1m/1h/1d)
    - **Free Tier Limits** (Automatic fallback from SIP to IEX, rate limit handling)
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Symbol Input
        symbols_input = st.text_area(
            "Symbols (Mix of Stocks and Crypto allowed!)", 
            placeholder="AAPL, BTC/USDT, MSFT, ETH/USDT",
            help="Enter any mix of symbols. The Smart Fetcher will route them automatically.",
            height=150
        )
        
        # Parse symbols
        symbols = [s.strip().upper() for s in symbols_input.replace('\n', ',').split(',') if s.strip()]
        if symbols:
            st.caption(f"Found {len(symbols)} symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
    
    with col2:
        # Date and Interval
        st.markdown("**Settings**")
        intervals = st.multiselect(
            "Intervals to Fetch",
            ["1s", "1m", "5m", "15m", "1h", "1d"],
            default=["1m", "1h"],
            help="Select multiple intervals. Results will be combined into one dataset with an 'interval' column."
        )
        
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
        end_date = st.date_input("End Date", value=datetime.now())
        
        st.markdown("---")
        fetch_btn = st.button("🚀 Start Smart Fetch", type="primary", use_container_width=True, disabled=not symbols)
        
    if fetch_btn and symbols:
        # Smart Fetch Implementation
        with st.spinner("🔄 Starting Smart Fetch..."):
            # Group symbols by optimal data source
            source_groups = {}
            for symbol in symbols:
                source = get_smart_source(symbol, intervals)
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append(symbol)

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_groups = len(source_groups)
            current_progress = 0

            all_data_frames = []

            for source, syms in source_groups.items():
                current_progress += 1
                progress_bar.progress(current_progress / total_groups,
                                    text=f"Fetching {len(syms)} symbols from {source.upper()}...")

                # Determine exchange for CCXT
                exchange_id = 'binance' if source == 'ccxt' else None

                # Fetch data for this source group
                df = fetch_data(
                    symbols=syms,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    interval=intervals,
                    source=source,
                    exchange_id=exchange_id,
                    save_mode="append"
                )

                if df is not None and not df.empty:
                    all_data_frames.append(df)
                    status_text.text(f"✅ Fetched {len(df):,} records from {source.upper()}")
                else:
                    status_text.text(f"⚠️ No data from {source.upper()}")

            # Combine all fetched data
            progress_bar.progress(1.0, text="Combining data...")
            if all_data_frames:
                combined_df = pd.concat(all_data_frames, ignore_index=True)

                # Update session state
                st.session_state.historical_data = combined_df
                st.session_state.symbols = symbols
                st.session_state.data_source = "smart_fetch"
                st.session_state.date_range = (start_date.isoformat(), end_date.isoformat())

                # Clear progress
                progress_bar.empty()
                status_text.empty()

                st.success(f"✅ Smart Fetch completed! Loaded {len(combined_df):,} records from {len(symbols)} symbols")
                st.balloons()
            else:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ No data was fetched from any source. Please check your symbols and try again.")

            st.rerun()


def render_saved_datasets_tab():
    """Render the saved datasets management tab."""
    st.markdown("### 💾 Saved Datasets")
    st.markdown("Manage your persisted datasets for long-living sessions")
    
    # Get list of saved datasets
    datasets = list_datasets()
    
    if not datasets:
        st.info("No saved datasets yet. Fetch some data and save it to see it here.")
        return
    
    # Show storage info
    total_size = get_total_storage_size()
    st.metric("Total Storage Used", format_size(total_size))
    
    # Display datasets in a table-like format
    for ds in datasets:
        with st.expander(f"📁 {ds.get('name', 'Unknown')}", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Handle both 'rows' and 'row_count' for compatibility
                rows = ds.get('rows') or ds.get('row_count', 'N/A')
                rows_str = f"{rows:,}" if isinstance(rows, (int, float)) else rows
                st.markdown(f"**Rows:** {rows_str}")
                st.markdown(f"**Symbols:** {', '.join(ds.get('symbols', [])[:5])}{'...' if len(ds.get('symbols', [])) > 5 else ''}")
                
                date_range = ds.get('date_range', {})
                if date_range:
                    if isinstance(date_range, dict):
                        start = str(date_range.get('start', 'N/A'))[:10]
                        end = str(date_range.get('end', 'N/A'))[:10]
                    elif isinstance(date_range, (list, tuple)) and len(date_range) >= 2:
                        start = str(date_range[0])[:10]
                        end = str(date_range[1])[:10]
                    else:
                        start = end = 'N/A'
                    st.markdown(f"**Date Range:** {start} to {end}")
            
            with col2:
                st.markdown(f"**Source:** {ds.get('source', 'N/A')}")
                st.markdown(f"**Interval:** {ds.get('interval', 'N/A')}")
                st.markdown(f"**Created:** {ds.get('created_at', 'N/A')[:10]}")
            
            with col3:
                # Load button
                if st.button("📤 Load", key=f"saved_load_{ds['name']}", use_container_width=True):
                    try:
                        result: tuple[pd.DataFrame, dict] = load_dataset(ds['name'])  # type: ignore
                        df, meta = result
                        
                        # Option to stack or replace
                        if st.session_state.historical_data is not None and st.session_state.data_load_mode == "append":
                            st.session_state.historical_data = stack_data(
                                st.session_state.historical_data, df, "append"
                            )
                            st.success(f"✅ Appended {len(df):,} rows from '{ds['name']}'")
                        else:
                            st.session_state.historical_data = df
                            st.success(f"✅ Loaded {len(df):,} rows from '{ds['name']}'")
                        
                        # Update symbols
                        if meta.get('symbols'):
                            st.session_state.symbols = meta['symbols']
                            sync_symbols_input()  # Sync the input field
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load dataset: {e}")
                
                # Delete button
                if st.button("🗑️ Delete", key=f"saved_delete_{ds['name']}", use_container_width=True):
                    if delete_dataset(ds['name']):
                        st.success(f"Deleted '{ds['name']}'")
                        st.rerun()
                    else:
                        st.error("Failed to delete dataset")


def render_merge_append_tab():
    """Render the merge/append datasets tab."""
    st.markdown("### 🔗 Merge & Append Datasets")
    st.markdown("Combine multiple datasets or add new data to existing datasets")
    
    # Get list of saved datasets
    datasets = list_datasets()
    
    if not datasets:
        st.info("No saved datasets available. Fetch and save data first.")
        return
    
    dataset_names = [ds['name'] for ds in datasets]
    
    # Two modes: Merge multiple datasets, or Append to existing dataset
    mode = st.radio(
        "Operation Mode",
        ["Merge Multiple Datasets", "Append New Data to Existing Dataset"],
        help="Choose how you want to combine data"
    )
    
    if mode == "Merge Multiple Datasets":
        st.markdown("#### 🔀 Merge Multiple Datasets")
        st.markdown("Combine two or more datasets into a new dataset. Duplicates will be removed based on symbol and timestamp.")
        
        # Select datasets to merge
        selected_datasets = st.multiselect(
            "Select datasets to merge",
            dataset_names,
            help="Choose 2 or more datasets to combine"
        )
        
        if len(selected_datasets) < 2:
            st.warning("⚠️ Please select at least 2 datasets to merge")
            return
        
        # Show preview of what will be merged
        st.markdown("**Preview:**")
        preview_data = []
        for ds_name in selected_datasets:
            ds_info = next((ds for ds in datasets if ds['name'] == ds_name), None)
            if ds_info:
                preview_data.append({
                    'Dataset': ds_name,
                    'Rows': f"{ds_info.get('row_count', 'N/A'):,}",
                    'Symbols': len(ds_info.get('symbols', [])),
                    'Source': ds_info.get('source', 'N/A')
                })
        
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
        
        # New dataset name
        new_dataset_name = st.text_input(
            "New dataset name",
            value=f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Name for the merged dataset"
        )
        
        # Merge button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔀 Merge Datasets", type="primary", use_container_width=True):
                with st.spinner("Merging datasets..."):
                    if merge_datasets(selected_datasets, new_dataset_name):
                        st.success(f"✅ Successfully merged {len(selected_datasets)} datasets into '{new_dataset_name}'")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Failed to merge datasets")
    
    else:  # Append mode
        st.markdown("#### ➕ Append New Data to Existing Dataset")
        st.markdown("Add newly fetched data to an existing dataset. Duplicates will be removed automatically.")
        
        # Select target dataset
        target_dataset = st.selectbox(
            "Target dataset (to append to)",
            dataset_names,
            help="Select the dataset you want to add data to"
        )
        
        # Show current dataset info
        target_info = next((ds for ds in datasets if ds['name'] == target_dataset), None)
        if target_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Rows", f"{target_info.get('row_count', 'N/A'):,}")
            with col2:
                st.metric("Symbols", len(target_info.get('symbols', [])))
            with col3:
                st.metric("Source", target_info.get('source', 'N/A'))
            
            st.markdown(f"**Symbols in dataset:** {', '.join(target_info.get('symbols', [])[:10])}{'...' if len(target_info.get('symbols', [])) > 10 else ''}")
        
        st.markdown("---")
        
        # Option 1: Append from current session data
        st.markdown("**Option 1: Append from currently loaded data**")
        
        if st.session_state.historical_data is not None:
            current_data = st.session_state.historical_data
            st.info(f"📊 Current session has {len(current_data):,} rows of data loaded")
            
            if st.button("➕ Append Session Data", use_container_width=True):
                with st.spinner("Appending data..."):
                    # Load target dataset
                    result = load_dataset(target_dataset)
                    if result:
                        target_df, target_meta = result
                        
                        # Combine with current data
                        combined_df = pd.concat([target_df, current_data], ignore_index=True)
                        
                        # Remove duplicates
                        if 'symbol' in combined_df.columns and 'timestamp' in combined_df.columns:
                            combined_df = combined_df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')
                        
                        # Get all symbols
                        if 'symbol' in combined_df.columns:
                            all_symbols = combined_df['symbol'].unique().tolist()
                        else:
                            all_symbols = target_meta.get('symbols', [])
                        
                        # Save back to target dataset
                        if save_dataset(
                            combined_df,
                            target_dataset,
                            all_symbols,
                            target_meta.get('source', 'Mixed'),
                            target_meta.get('date_range'),
                            append=False  # Replace with merged data
                        ):
                            old_row_count = target_info.get('row_count', 0) if target_info else 0
                            new_rows = len(combined_df) - old_row_count
                            st.success(f"✅ Added {new_rows:,} new rows to '{target_dataset}' (total: {len(combined_df):,} rows)")
                            st.rerun()
                        else:
                            st.error("❌ Failed to append data")
        else:
            st.warning("⚠️ No data loaded in current session. Fetch data first, then append it here.")
        
        st.markdown("---")
        
        # Option 2: Append from another saved dataset
        st.markdown("**Option 2: Append from another saved dataset**")
        
        source_datasets = [ds for ds in dataset_names if ds != target_dataset]
        if source_datasets:
            source_dataset = st.selectbox(
                "Source dataset (to append from)",
                source_datasets,
                help="Select the dataset to append to the target"
            )
            
            if st.button("➕ Append from Dataset", use_container_width=True):
                with st.spinner("Appending datasets..."):
                    # Merge the two datasets
                    if merge_datasets([target_dataset, source_dataset], target_dataset):
                        st.success(f"✅ Successfully appended '{source_dataset}' to '{target_dataset}'")
                        st.run()
                    else:
                        st.error("❌ Failed to append datasets")
        else:
            st.info("No other datasets available to append from")


def render_fetch_tab():
    """Render the data fetching tab."""
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
                        if st.button("📂 Load", key=f"fetch_load_{dataset['name']}", use_container_width=True):
                            result = load_dataset(dataset['name'])
                            if result is not None:
                                df, meta = result
                                st.session_state.historical_data = df
                                st.session_state.symbols = meta.get('symbols', dataset['symbols'])
                                sync_symbols_input()  # Sync the input field
                                st.success(f"✅ Loaded {dataset['name']}")
                                st.rerun()
                    
                    with col_ds3:
                        if st.button("🗑️ Delete", key=f"fetch_delete_{dataset['name']}", use_container_width=True):
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
        data_sources = [
            "CCXT - Crypto Exchanges (FREE! ⭐)",
            "Yahoo Finance (FREE! Stocks & ETFs)",
            "Finnhub (Stock Market Data)",
            "Alpha Vantage (FREE 25 calls/day)",
            "Massive (Institutional Data - 100 calls/day)",
            "Alpaca (FREE! US Stocks, 1s bars)",
            "Upload CSV",
            "Mock/Synthetic"
        ]
        data_source = st.selectbox(
            "🔌 Data Source",
            data_sources,
            help="💡 All connectors have API keys configured in api_keys.properties"
        )

        # Exchange selection for CCXT
        exchange_id = 'binance'  # default
        if data_source.startswith("CCXT"):
            exchange_id = st.selectbox(
                "📊 Exchange",
                ["binance", "kraken", "coinbase", "bybit", "okx"],
                help=(
                    "• Binance: Most liquid, best for most pairs\n"
                    "• Kraken: Reliable, regulated\n"
                    "• Coinbase: US-based, highly regulated\n"
                    "• Bybit: Good for perpetuals\n"
                    "• OKX: Wide variety of altcoins"
                )
            )

        # Alpaca limitations and info
        if data_source.startswith("Alpaca"):
            st.info("""
            **Alpaca Data Source (FREE! US Stocks, 1s bars)**
            - 1-second bars for US stocks (NYSE, NASDAQ, AMEX)
            - Up to 5 years history
            - Market hours only
            - Rate limit: 200 requests/minute
            - Requires free Alpaca account and API keys (already configured)
            - Only supports US stocks (no ETFs, no crypto)
            """)
            st.info(
                f"✅ Using {exchange_id.title()} - FREE public data, no API key needed!\n"
                f"Supports second-level historical data for crypto pairs."
            )
        
        # Alpha Vantage rate limiting info
        if data_source.startswith("Alpha Vantage"):
            # Initialize rate limit tracking
            if 'av_calls_today' not in st.session_state:
                st.session_state.av_calls_today = 0
                st.session_state.av_last_reset = datetime.now().date()
            
            # Reset daily counter
            if st.session_state.av_last_reset < datetime.now().date():
                st.session_state.av_calls_today = 0
                st.session_state.av_last_reset = datetime.now().date()
            
            calls_remaining = 25 - st.session_state.av_calls_today
            
            if calls_remaining > 15:
                st.info(
                    f"📊 **Alpha Vantage Status:** ✅ {calls_remaining}/25 calls remaining today\n"
                    f"⏱️ Rate: 5 calls/min | Resets: midnight UTC"
                )
            elif calls_remaining > 0:
                st.warning(
                    f"⚠️ **Alpha Vantage Status:** {calls_remaining}/25 calls remaining\n"
                    f"Use wisely - limit resets at midnight UTC"
                )
            else:
                st.error("🚫 Alpha Vantage daily limit reached! Try Yahoo Finance or CCXT")
        
        # Massive rate limiting info
        if data_source.startswith("Massive"):
            # Initialize rate limit tracking
            if 'massive_calls_today' not in st.session_state:
                st.session_state.massive_calls_today = 0
                st.session_state.massive_last_reset = datetime.now().date()
            
            # Reset daily counter
            if st.session_state.massive_last_reset < datetime.now().date():
                st.session_state.massive_calls_today = 0
                st.session_state.massive_last_reset = datetime.now().date()
            
            calls_remaining = 100 - st.session_state.massive_calls_today
            
            if calls_remaining > 50:
                st.info(
                    f"🏛️ **Massive.com Status:** ✅ {calls_remaining}/100 calls remaining\n"
                    f"📥 10 GB/month bulk downloads available"
                )
            elif calls_remaining > 0:
                st.warning(f"⚠️ **Massive.com Status:** {calls_remaining}/100 calls remaining")
            else:
                st.error("🚫 Massive.com daily limit reached!")
        
        # Finnhub info
        if data_source.startswith("Finnhub"):
            st.info(
                "📊 **Finnhub API** configured from api_keys.properties\n"
                "✅ Real-time & historical stock data available"
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
                    if 'symbol' in df.columns:
                        symbols = df['symbol'].unique().tolist()
                        st.session_state.symbols = symbols
                        sync_symbols_input()  # Sync the input field
                    else:
                        st.warning("CSV does not contain a 'symbol' column. Some features may be limited.")
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
                            # Get current symbols from the manual entry text area
                            current_input = st.session_state.get('symbols_input', '')
                            current_symbols = parse_symbols_from_input(current_input)
                            # Append and remove duplicates
                            combined = list(set(current_symbols + preset_symbols))
                            st.session_state.symbols = combined
                            sync_symbols_input()  # Sync the input field immediately
                            new_count = len(combined) - len(current_symbols)
                            st.success(f"✅ Added {new_count} new symbols (Total: {len(combined)})")
                            st.rerun()
                    
                    with preset_col_btn2:
                        if st.button("🔄 Replace", use_container_width=True, help="Replace all symbols"):
                            preset_symbols = get_preset_symbols(preset_category, preset_name)
                            st.session_state.symbols = preset_symbols
                            sync_symbols_input()  # Sync the input field immediately
                            st.success(f"✅ Replaced with {len(preset_symbols)} symbols from {preset_name}")
                            st.rerun()
            
            st.markdown("---")
            st.markdown("#### ✏️ Manual Entry")
            
            # Dynamic help text based on data source
            if data_source.startswith("CCXT"):
                symbol_help = (
                    "Enter crypto pairs (e.g., BTC/USDT, ETH/USDT)\n"
                    "Format: BASE/QUOTE (e.g., BTC/USDT, ETH/BTC)"
                )
                placeholder_text = "BTC/USDT\nETH/USDT\nSOL/USDT"
            else:
                symbol_help = "Enter stock symbols (e.g., AAPL, GOOGL, MSFT)"
                placeholder_text = "AAPL\nMSFT\nGOOGL"
            
            # Symbol management buttons
            symbol_btn_col1, symbol_btn_col2, symbol_btn_col3 = st.columns([2, 1, 1])

            with symbol_btn_col2:
                if st.button("🗑️ Clear Symbols", use_container_width=True, help="Clear all symbols"):
                    st.session_state.symbols = []
                    st.session_state.symbols_input = ""
                    st.success("✅ Symbols cleared!")
                    st.rerun()

            with symbol_btn_col3:
                if st.button("🗑️ Clear Data", use_container_width=True, help="Clear loaded data"):
                    st.session_state.historical_data = None
                    st.success("✅ Data cleared!")
                    st.rerun()

            # Symbol input
            with symbol_btn_col1:
                # Ensure input value is synced with session state
                if 'symbols_input' not in st.session_state:
                    st.session_state.symbols_input = ""
                if 'symbols' in st.session_state and st.session_state.symbols:
                    sync_symbols_input()

                symbols_input = st.text_area(
                    "Symbols (one per line or comma-separated)",
                    placeholder=placeholder_text,
                    height=100,
                    help=symbol_help,
                    label_visibility="visible",
                    key="symbols_input"
                )

                # Update session state when input changes
                st.session_state.symbol_input_value = symbols_input
            
            # Parse and clean symbols
            symbols = parse_symbols_from_input(symbols_input)

            # Update session state symbols
            st.session_state.symbols = symbols

            # Show cleaned symbols if any were modified
            original_symbols = []
            for line in symbols_input.split('\n'):
                original_symbols.extend([s.strip().upper() for s in line.split(',') if s.strip()])
            if any(' ' in s for s in original_symbols):
                st.info(f"ℹ️  Cleaned symbols: removed spaces from {len([s for s in original_symbols if ' ' in s])} symbol(s)")
            
            # Interval selection (moved before date range for smart defaults)
            if data_source.startswith("Alpaca"):
                # Alpaca supports 1-second bars
                interval = st.selectbox(
                    "Data Interval",
                    ["1s", "1m", "5m", "15m", "30m", "1h", "1d"],
                    index=0,  # Default to 1s for Alpaca
                    help="⭐ Alpaca supports 1-second bars for US stocks!"
                )
            else:
                interval = st.selectbox(
                    "Data Interval",
                    ["1m", "5m", "15m", "30m", "1h", "1d"],
                    index=4,  # Default to 1h
                    help="Time interval for OHLCV data"
                )
            
            # Smart default date range based on interval and data source
            if data_source.startswith("Alpaca"):
                if interval == "1s":
                    default_days = 1  # 1-second data: recommend 1 day
                    st.caption("⭐ Alpaca: 1s bars available! Recommend 1-3 days for optimal performance")
                elif interval == "1m":
                    default_days = 7
                elif interval in ["5m", "15m"]:
                    default_days = 30
                else:
                    default_days = 90
            elif data_source == "Yahoo Finance":
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
                        f"⚠️ **Warning:** You selected CCXT (crypto exchange) but your symbols look like stocks: {', '.join(stock_like_symbols[:5])}{'...' if len(stock_like_symbols) > 5 else ''}\n\n"
                        f"**Crypto exchanges don't have stock data!**\n\n"
                        f"**Options:**\n"
                        f"1. Use **Yahoo Finance** source for stocks (AAPL, GOOGL, etc.)\n"
                        f"2. Or use crypto pairs like: BTC/USDT, ETH/USDT, SOL/USDT\n"
                        f"3. Or switch to 'Crypto' category in Quick Select"
                    )
            
            # Validation warning for Yahoo Finance with crypto pairs
            if data_source == "Yahoo Finance":
                crypto_pairs = [s for s in symbols if '/' in s]
                if crypto_pairs:
                    st.warning(
                        f"⚠️ **Warning:** You selected Yahoo Finance but have crypto pair format: {', '.join(crypto_pairs[:3])}\n\n"
                        f"**Yahoo Finance uses different format for crypto:**\n"
                        f"• Use 'BTC-USD' instead of 'BTC/USDT'\n"
                        f"• Or switch to 'CCXT - Crypto Exchanges' for crypto pairs"
                    )
                
                # Check date range for intraday data
                days_diff = (end_date - start_date).days
                
                if interval == "1m" and days_diff > 7:
                    st.error(
                        f"❌ **Invalid Date Range for 1m interval**\n\n"
                        f"Yahoo Finance **1-minute data** is limited to the **last 7 days only**.\n"
                        f"Your range: {days_diff} days\n\n"
                        f"**Solutions:**\n"
                        f"1. Reduce date range to last 7 days\n"
                        f"2. Use **5m** or higher interval for longer history\n"
                        f"3. Use **CCXT** for crypto (supports longer 1m history)"
                    )
                elif interval in ["5m", "15m", "30m"] and days_diff > 60:
                    st.error(
                        f"❌ **Invalid Date Range for {interval} interval**\n\n"
                        f"Yahoo Finance **{interval} data** is limited to the **last 60 days only**.\n"
                        f"Your range: {days_diff} days\n\n"
                        f"**Solutions:**\n"
                        f"1. Reduce date range to last 60 days\n"
                        f"2. Use **1h** or **1d** interval for longer history\n"
                        f"3. Use **CCXT** for crypto (supports longer history)"
                    )
                elif interval in ["1m", "5m", "15m", "30m"]:
                    # Show helpful info for valid ranges
                    max_days = 7 if interval == "1m" else 60
                    if days_diff > max_days * 0.8:  # Warn if approaching limit
                        st.info(
                            f"ℹ️  **Note:** You're requesting {days_diff} days of {interval} data.\n"
                            f"Yahoo Finance limit is {max_days} days for this interval.\n"
                            f"Consider using **1h** or **1d** for longer historical analysis."
                        )
            
            # Rate-limit tips for intraday data
            if interval == "1m" and data_source.startswith("CCXT"):
                with st.expander("💡 Tips for Collecting 1m Intraday Data", expanded=False):
                    st.markdown("""
                    **📊 Free Tier Rate Limit Strategy:**
                    
                    Most exchanges allow ~1000 candles per request. For 1-minute data:
                    - 1 day = ~1440 candles (24h × 60min)
                    - 7 days = ~10,080 candles (fetched in batches)
                    
                    **🚀 Recommended Approach for Arbitrage Research:**
                    1. **Start small:** Fetch 1-3 days first
                    2. **Use Append mode:** Add more data incrementally
                    3. **Save frequently:** Use the "Save to Disk" button after each fetch
                    4. **Stack datasets:** Use "Load" from Saved Datasets to restore + Append new data
                    
                    **⏱️ Expected Fetch Times (Binance):**
                    - 1 day, 1 symbol: ~5-10 seconds
                    - 1 day, 10 symbols: ~30-60 seconds
                    - 7 days, 1 symbol: ~20-40 seconds
                    - 7 days, 10 symbols: ~3-5 minutes
                    
                    **💾 Persistence:**
                    Save your data to disk after fetching. It will be available after restart.
                    """)
            
            # Load mode selection
            st.markdown("---")
            st.markdown("#### 📦 Data Loading Mode")
            load_mode_col1, load_mode_col2 = st.columns([2, 1])
            
            with load_mode_col1:
                load_mode = st.radio(
                    "When fetching new data:",
                    ["Replace", "Append", "Update"],
                    index=0,
                    horizontal=True,
                    help=(
                        "**Replace:** Clear existing data and load fresh\n"
                        "**Append:** Add new data, keep existing for overlaps\n"
                        "**Update:** Add new data, prefer new for overlaps"
                    )
                )
                st.session_state.data_load_mode = load_mode.lower()
            
            with load_mode_col2:
                if st.session_state.historical_data is not None:
                    current_rows = len(st.session_state.historical_data)
                    st.caption(f"📊 Current: {current_rows:,} rows")
            
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
                        'alpha': 'alpha_vantage',
                        'massive': 'massive',
                        'alpaca': 'alpaca',
                        'mock': 'synthetic',
                        'upload': 'synthetic'
                    }
                    source_key = data_source.lower().split()[0]
                    internal_source = source_map.get(source_key, 'yfinance')  # Default to yfinance
                    
                    fetch_data(
                        symbols=symbols,
                        start=start_date.isoformat(),
                        end=end_date.isoformat(),
                        interval=interval,
                        source=internal_source,
                        exchange_id=exchange_id if data_source.startswith("CCXT") else None,
                        save_mode=st.session_state.data_load_mode
                    )
    
    with col2:
        st.markdown("### Quick Info")
        
        if st.session_state.historical_data is not None:
            df = st.session_state.historical_data
            
            # Reset index if MultiIndex for display
            if isinstance(df.index, pd.MultiIndex):
                df_display = df.reset_index()
            else:
                df_display = df
            
            # Show stats
            st.metric("Total Records", f"{len(df_display):,}")
            if 'symbol' in df_display.columns:
                st.metric("Symbols", len(df_display['symbol'].unique()))
            
            if 'timestamp' in df_display.columns:
                date_range = f"{df_display['timestamp'].min().date()} to {df_display['timestamp'].max().date()}"
                st.metric("Date Range", date_range)
            
            # Data quality
            missing_pct = (df_display.isnull().sum().sum() / (len(df_display) * len(df_display.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.2f}%")
            
            st.markdown("---")
            st.markdown("### 💾 Save Data")
            
            # Save dataset button
            save_name = st.text_input(
                "Dataset Name",
                value=generate_dataset_name(
                    st.session_state.symbols if st.session_state.symbols else ["unknown"],
                    st.session_state.get('interval', '1h'),
                    st.session_state.get('data_source', 'manual')
                ),
                help="Name for the saved dataset"
            )
            
            if st.button("💾 Save to Disk", use_container_width=True, type="secondary"):
                try:
                    metadata = {
                        "source": st.session_state.get('data_source', 'unknown'),
                        "interval": st.session_state.get('interval', 'unknown'),
                        "symbols": st.session_state.symbols,
                    }
                    date_range_tuple = None
                    if st.session_state.get('date_range'):
                        date_range_tuple = (
                            st.session_state.date_range[0],
                            st.session_state.date_range[1]
                        )
                    
                    save_dataset(
                        df, save_name,
                        symbols=st.session_state.symbols or [],
                        source=metadata.get('source', 'unknown'),
                        date_range=date_range_tuple,
                        append=False
                    )
                    st.success(f"✅ Saved as '{save_name}'")
                except Exception as e:
                    st.error(f"Failed to save: {e}")
            
            # Clear data button
            if st.button("🗑️ Clear Data", use_container_width=True):
                st.session_state.historical_data = None
                st.session_state.symbols = []
                sync_symbols_input()
                st.success("✅ Data and symbols cleared!")
                st.rerun()
        else:
            st.info("No data loaded yet")
            st.markdown("""
            **Data sources:**
            - **⭐ CCXT (Recommended for Crypto)**: FREE access to 100+ crypto exchanges
              - Binance, Kraken, Coinbase, Bybit, OKX and more
              - No API key required for public data
              - Second-level historical data (1m, 5m, 15m, 1h, 1d)
              - Best for crypto trading strategies
            - **⭐ Yahoo Finance (Recommended for Stocks)**: FREE historical data
              - All US stocks, ETFs, indices
              - Major cryptocurrencies
              - No API key required
              - Reliable and unlimited (within reasonable use)
            - **⭐ Alpaca (Recommended for US Stocks HFT)**: FREE 1-second bars
              - US stocks with 1-second resolution
              - FREE tier: perfect for HFT development
              - API key configured in api_keys.properties
              - WebSocket support for live trading
            - **Finnhub**: Stock market data via API
              - Real-time & historical stock data
              - API key configured in api_keys.properties
              - Good for US stocks and forex
            - **Alpha Vantage**: FREE 25 API calls/day
              - Stocks, forex, crypto data
              - API key configured in api_keys.properties
              - ⚠️ Free tier: 5 calls/minute, 25 calls/day
            - **Massive**: Institutional-grade market data
              - FREE 100 calls/day + 10GB/month bulk files
              - API key configured in api_keys.properties
              - Stocks, options, futures, crypto
            - **CSV Upload**: Custom data files
            - **Mock**: Synthetic data for testing
            """)
    
    st.markdown("---")
    
    # Data preview and visualization
    if st.session_state.historical_data is not None:
        display_data_preview()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(symbols: List[str], start: str, end: str, interval: Union[str, List[str]], source: str, exchange_id: Optional[str] = None, save_mode: str = "append") -> pd.DataFrame:
    """Fetch data with caching and persistence using real data sources"""
    display_source = f"{source} ({exchange_id})" if exchange_id else source
    with st.spinner(f"Fetching data for {len(symbols)} symbols from {display_source}..."):
        try:
            # Handle multiple intervals
            intervals_to_fetch = [interval] if isinstance(interval, str) else interval

            all_data_frames = []
            for intv in intervals_to_fetch:
                # Try to use real data fetching first
                try:
                    # Use gRPC client for data fetching
                    client = TradingGrpcClient(GrpcConfig())
                    st.info("🔄 Using gRPC service for data fetching...")

                    # For now, we'll implement basic data fetching based on source
                    # In a real implementation, you'd add specific endpoints to the Rust gRPC server

                    new_df = fetch_real_data(symbols, start, end, intv, source, exchange_id)

                except Exception as e:
                    st.warning(f"gRPC service not available, falling back to local data fetching: {e}")
                    new_df = fetch_real_data(symbols, start, end, intv, source, exchange_id)

                if new_df is not None and not new_df.empty:
                    # Add interval column
                    new_df['interval'] = intv
                    all_data_frames.append(new_df)

            # Combine all interval data
            if all_data_frames:
                new_df = pd.concat(all_data_frames, ignore_index=True)
            else:
                new_df = pd.DataFrame()

            if new_df is None or new_df.empty:
                st.warning("No data fetched. This might be due to API limitations or connectivity issues.")
                return pd.DataFrame()

            # Handle data loading mode: replace, append, or update
            if save_mode == "replace" or st.session_state.historical_data is None:
                st.session_state.historical_data = new_df
            else:
                # Use stack_data to append or update existing data
                st.session_state.historical_data = stack_data(
                    st.session_state.historical_data, new_df, save_mode
                )

            st.session_state.symbols = symbols
            st.session_state.data_source = source
            st.session_state.date_range = (start, end)

            # Auto-save to persistent storage
            dataset_name = f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if exchange_id:
                dataset_name = f"{exchange_id}_{dataset_name}"

            # Auto-save dataset
            save_dataset(
                new_df, dataset_name,
                symbols=symbols,
                source=display_source,
                date_range=(start, end),
                append=(save_mode == "append")
            )
            st.info(f"💾 Data saved as '{dataset_name}'")

            # Display appropriate success message based on mode
            total_rows = len(st.session_state.historical_data)
            if save_mode == "replace":
                st.success(f"✅ Successfully loaded {len(new_df):,} records for {len(symbols)} symbols")
            elif save_mode == "append":
                st.success(f"✅ Successfully appended {len(new_df):,} new records for {len(symbols)} symbols (Total: {total_rows:,} rows)")
            else:  # update
                st.success(f"✅ Successfully updated data with {len(new_df):,} records for {len(symbols)} symbols (Total: {total_rows:,} rows)")
            return st.session_state.historical_data

        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return pd.DataFrame()

def fetch_real_data(symbols: List[str], start: str, end: str, interval: str, source: str, exchange_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Fetch real data from various sources"""
    try:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        
        if source == 'ccxt':
            return fetch_ccxt_data(symbols, start_date, end_date, interval, exchange_id or 'binance')
        elif source == 'yfinance':
            return fetch_yfinance_data(symbols, start_date, end_date, interval)
        elif source == 'finnhub':
            return fetch_finnhub_data(symbols, start_date, end_date, interval)
        elif source == 'alpha_vantage':
            return fetch_alpha_vantage_data(symbols, start_date, end_date, interval)
        elif source == 'alpaca':
            return fetch_alpaca_data(symbols, start_date, end_date, interval)
        else:
            # Fallback to mock data for testing
            return generate_mock_data(symbols, start_date, end_date, interval)
            
    except Exception as e:
        st.error(f"Error fetching {source} data: {e}")
        return generate_mock_data(symbols, pd.to_datetime(start), pd.to_datetime(end), interval)

def fetch_ccxt_data(symbols: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp, interval: str, exchange_id: str) -> pd.DataFrame:
    """Fetch data from CCXT crypto exchanges"""
    try:
        import ccxt
        
        exchange = getattr(ccxt, exchange_id)({
            'sandbox': False,  # change to True if you have a testnet
            'enableRateLimit': True,
        })
        
        # Convert interval to CCXT format
        timeframe_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '1d': '1d'
        }
        timeframe = timeframe_map.get(interval, '1h')
        
        data_rows = []
        
        for symbol in symbols:
            try:
                # Ensure symbol format is correct for CCXT
                if '/' not in symbol:
                    symbol = f"{symbol}/USDT"  # Default to USDT pairs
                
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=int(start_date.timestamp() * 1000), limit=1000)
                
                for candle in ohlcv:
                    timestamp, open_price, high, low, close, volume = candle
                    if timestamp <= end_date.timestamp() * 1000:
                        data_rows.append({
                            'timestamp': pd.to_datetime(timestamp, unit='ms'),
                            'symbol': symbol,
                            'open': float(open_price),
                            'high': float(high),
                            'low': float(low),
                            'close': float(close),
                            'volume': float(volume)
                        })
                        
            except Exception as e:
                st.warning(f"Failed to fetch {symbol} from {exchange_id}: {e}")
                continue
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            return df.sort_values(['timestamp', 'symbol'])
        else:
            st.warning(f"No data returned from {exchange_id}")
            return pd.DataFrame()
            
    except ImportError:
        st.error("CCXT library not installed. Install with: pip install ccxt")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"CCXT fetch error: {e}")
        return pd.DataFrame()

def fetch_yfinance_data(symbols: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp, interval: str) -> pd.DataFrame:
    """Fetch data from Yahoo Finance"""
    try:
        import yfinance as yf
        
        # Convert interval to Yahoo Finance format
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '1d': '1d'
        }
        yf_interval = interval_map.get(interval, '1h')
        
        data_rows = []
        
        for symbol in symbols:
            try:
                # Convert crypto pairs to Yahoo Finance format
                if '/' in symbol:
                    symbol = symbol.replace('/', '-')
                    # For crypto, use USDT or USD suffix
                    if not symbol.endswith('-USD'):
                        symbol = symbol.replace('-USDT', '-USD')
                
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, interval=yf_interval)
                
                if not hist.empty:
                    for timestamp, row in hist.iterrows():
                        data_rows.append({
                            'timestamp': pd.to_datetime(timestamp),
                            'symbol': symbol,
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close']),
                            'volume': float(row['Volume'])
                        })
                        
            except Exception as e:
                st.warning(f"Failed to fetch {symbol} from Yahoo Finance: {e}")
                continue
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            return df.sort_values(['timestamp', 'symbol'])
        else:
            st.warning("No data returned from Yahoo Finance")
            return pd.DataFrame()
            
    except ImportError:
        st.error("yfinance library not installed. Install with: pip install yfinance")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Yahoo Finance fetch error: {e}")
        return pd.DataFrame()

def fetch_finnhub_data(symbols: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp, interval: str) -> pd.DataFrame:
    """Fetch data from Finnhub API"""
    try:
        import finnhub
        import os
        
        api_key = os.getenv('FINNHUB_API_KEY') or 'demo'
        client = finnhub.Client(api_key=api_key)
        
        # Convert interval to Finnhub format
        interval_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '1d': 'D'
        }
        resolution = interval_map.get(interval, 3600)
        
        data_rows = []
        
        for symbol in symbols:
            try:
                # Remove exchange suffix if present (e.g., AAPL -> AAPL)
                clean_symbol = symbol.split('.')[0] if '.' in symbol else symbol
                
                # Fetch data
                candles = client.stock_candles(
                    clean_symbol, 
                    resolution, 
                    int(start_date.timestamp()), 
                    int(end_date.timestamp())
                )
                
                if candles['s'] == 'ok':
                    for i in range(len(candles['t'])):
                        data_rows.append({
                            'timestamp': pd.to_datetime(candles['t'][i], unit='s'),
                            'symbol': clean_symbol,
                            'open': float(candles['o'][i]),
                            'high': float(candles['h'][i]),
                            'low': float(candles['l'][i]),
                            'close': float(candles['c'][i]),
                            'volume': float(candles['v'][i])
                        })
                        
            except Exception as e:
                st.warning(f"Failed to fetch {symbol} from Finnhub: {e}")
                continue
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            return df.sort_values(['timestamp', 'symbol'])
        else:
            st.warning("No data returned from Finnhub")
            return pd.DataFrame()
            
    except ImportError:
        st.error("finnhub library not installed. Install with: pip install finnhub")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Finnhub fetch error: {e}")
        return pd.DataFrame()

def fetch_alpha_vantage_data(symbols: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp, interval: str) -> pd.DataFrame:
    """Fetch data from Alpha Vantage API"""
    try:
        import requests
        import os
        
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY') or 'demo'
        
        # Convert interval to Alpha Vantage format
        interval_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '60min', '1d': 'daily'
        }
        av_interval = interval_map.get(interval, '60min')
        
        data_rows = []
        
        for symbol in symbols:
            try:
                if av_interval == 'daily':
                    # Daily data
                    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
                else:
                    # Intraday data
                    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={av_interval}&apikey={api_key}"
                
                response = requests.get(url)
                data = response.json()
                
                # Parse the response
                if av_interval == 'daily':
                    time_series_key = 'Time Series (Daily)'
                else:
                    time_series_key = f'Time Series ({av_interval})'
                
                if time_series_key in data:
                    for date_str, values in data[time_series_key].items():
                        timestamp = pd.to_datetime(date_str)
                        if start_date <= timestamp <= end_date:
                            data_rows.append({
                                'timestamp': timestamp,
                                'symbol': symbol,
                                'open': float(values['1. open']),
                                'high': float(values['2. high']),
                                'low': float(values['3. low']),
                                'close': float(values['4. close']),
                                'volume': float(values['5. volume'])
                            })
                            
            except Exception as e:
                st.warning(f"Failed to fetch {symbol} from Alpha Vantage: {e}")
                continue
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            return df.sort_values(['timestamp', 'symbol'])
        else:
            st.warning("No data returned from Alpha Vantage")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Alpha Vantage fetch error: {e}")
        return pd.DataFrame()

def fetch_alpaca_data(symbols: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp, interval: str) -> pd.DataFrame:
    """Fetch data from Alpaca API"""
    try:
        import alpaca_trade_api as tradeapi
        import os
        
        api_key = os.getenv('ALPACA_API_KEY_ID') or 'demo'
        api_secret = os.getenv('ALPACA_API_SECRET_KEY') or 'demo'
        base_url = 'https://data.alpaca.markets'
        
        client = tradeapi.REST(api_key, api_secret, base_url=base_url)
        
        # Convert interval to Alpaca format
        interval_map = {
            '1s': '1Sec', '1m': '1Min', '5m': '5Min', '15m': '15Min',
            '30m': '30Min', '1h': '1Hour', '1d': '1Day'
        }
        alpaca_interval = interval_map.get(interval, '1Min')
        
        data_rows = []
        
        for symbol in symbols:
            try:
                # Fetch bars
                bars = client.get_bars(
                    symbol, 
                    alpaca_interval, 
                    start=start_date.isoformat(), 
                    end=end_date.isoformat()
                )
                
                for bar in bars:
                    data_rows.append({
                        'timestamp': pd.to_datetime(bar.t),
                        'symbol': symbol,
                        'open': float(bar.o),
                        'high': float(bar.h),
                        'low': float(bar.l),
                        'close': float(bar.c),
                        'volume': float(bar.v)
                    })
                    
            except Exception as e:
                st.warning(f"Failed to fetch {symbol} from Alpaca: {e}")
                continue
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            return df.sort_values(['timestamp', 'symbol'])
        else:
            st.warning("No data returned from Alpaca")
            return pd.DataFrame()
            
    except ImportError:
        st.error("alpaca-trade-api library not installed. Install with: pip install alpaca-trade-api")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Alpaca fetch error: {e}")
        return pd.DataFrame()

def generate_mock_data(symbols: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp, interval: str) -> pd.DataFrame:
    """Generate mock data for testing when real sources fail"""
    import numpy as np
    
    # Generate mock OHLCV data
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    data_rows = []

    for symbol in symbols:
        # Generate realistic price data
        np.random.seed(hash(symbol) % 2**32)
        n_points = len(date_range)

        # Start with a base price
        base_price = 100 + np.random.randn() * 20

        # Generate random walk prices
        price_changes = np.random.randn(n_points) * 0.02  # 2% volatility
        prices = base_price * np.exp(np.cumsum(price_changes))

        for i, timestamp in enumerate(date_range):
            price = prices[i]
            # Generate OHLC around the price
            volatility = abs(np.random.randn()) * 0.01
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = price * (1 + np.random.randn() * 0.005)
            close = price
            volume = np.random.randint(1000, 100000)

            data_rows.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

    new_df = pd.DataFrame(data_rows)
    
    # Convert to MultiIndex format
    if not new_df.empty:
        new_df = new_df.set_index(['timestamp', 'symbol'])
        new_df = new_df.unstack(level='symbol')
        new_df.columns = new_df.columns.swaplevel(0, 1)
        new_df = new_df.sort_index(axis=1)
    
    st.info("🔄 Generated mock data for testing purposes")
    return new_df

def display_data_preview():
    """Display data preview and visualization"""
    df = st.session_state.historical_data
    
    # Validate dataframe
    if df is None or df.empty:
        st.warning("No data available to preview.")
        return
    
    st.markdown("### 📋 Data Preview & Visualization")
    
    # Convert column-based MultiIndex to flat format for display/charting
    df_display = df.copy()
    
    # Check if we have a MultiIndex columns structure (Symbol, OHLCV)
    if isinstance(df_display.columns, pd.MultiIndex):
        try:
            # Check levels to determine structure
            # Expecting (Symbol, OHLCV) or (OHLCV, Symbol)
            if len(df_display.columns.levels) >= 2:
                # Assume level 0 is Symbol if it has more unique values than level 1, or based on conventions
                # But our fetch_data ensures (Symbol, Field) structure
                
                # Stack based on the symbol level (usually level 0 after our fetch_data transformation)
                # We want to end up with a flat DataFrame: timestamp, symbol, open, high, low, close...
                
                # Stack level 0 (Symbol) to move it to index
                df_display = df_display.stack(level=0)
                
                # Now index is (timestamp, symbol)
                # Reset index to make them columns
                df_display = df_display.reset_index()
                
                # Rename columns if generic names are assigned
                if 'level_0' in df_display.columns:
                    df_display = df_display.rename(columns={'level_0': 'timestamp'})
                if 'level_1' in df_display.columns:
                    df_display = df_display.rename(columns={'level_1': 'symbol'})
                
                # Ensure we have a symbol column
                if 'symbol' not in df_display.columns:
                    # Try to find a column that looks like a symbol
                    object_cols = df_display.select_dtypes(include=['object']).columns
                    for col in object_cols:
                        if df_display[col].nunique() < 500: # Heuristic for symbol column
                            df_display = df_display.rename(columns={col: 'symbol'})
                            break
                            
        except Exception as e:
            st.warning(f"Error flattening MultiIndex data: {e}. Showing raw data.")
            df_display = df.copy() # Fallback
            
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📑 Data Table", "📈 Statistics", "💾 Export"])
    
    with tab1:
        # Symbol selector for charting
        if 'symbol' in df_display.columns:
            try:
                symbols = sorted([s for s in df_display['symbol'].dropna().unique().tolist() if s is not None])
                if not symbols:
                    st.warning("No symbols found in data.")
                    selected_symbol = None
                else:
                    col_sel1, col_sel2 = st.columns([1, 3])
                    with col_sel1:
                        selected_symbol = st.selectbox("Select Symbol for Chart", symbols)
                    
                    # Filter data for selected symbol
                    symbol_df = df_display[df_display['symbol'] == selected_symbol].copy()
            except Exception as e:
                st.error(f"Error accessing symbol column: {e}")
                symbol_df = df_display.copy()
                selected_symbol = None
        else:
            st.info("Data does not contain a 'symbol' column. Showing all data.")
            symbol_df = df_display.copy()
            selected_symbol = None
        
        if not symbol_df.empty:
            # Ensure timestamp is available (either as column or index)
            if 'timestamp' not in symbol_df.columns:
                if isinstance(symbol_df.index, pd.DatetimeIndex):
                    symbol_df = symbol_df.reset_index()
                    symbol_df = symbol_df.rename(columns={'index': 'timestamp'})
                elif 'index' in symbol_df.columns:
                    # Check if 'index' column is actually datetime
                    try:
                        symbol_df['timestamp'] = pd.to_datetime(symbol_df['index'])
                    except:
                        pass
            
            # If we still don't have a timestamp column, look for datetime columns
            if 'timestamp' not in symbol_df.columns:
                datetime_cols = symbol_df.select_dtypes(include=['datetime64']).columns
                if not datetime_cols.empty:
                    symbol_df = symbol_df.rename(columns={datetime_cols[0]: 'timestamp'})
                else:
                    # Last resort: use index
                    symbol_df = symbol_df.reset_index()
                    symbol_df = symbol_df.rename(columns={'index': 'timestamp'})

            if 'timestamp' in symbol_df.columns:
                symbol_df = symbol_df.sort_values('timestamp')
            
            # Standardize column names (lowercase)
            symbol_df.columns = [c.lower() for c in symbol_df.columns]
            
            # Create OHLC candlestick chart
            if all(col in symbol_df.columns for col in ['open', 'high', 'low', 'close']):
                x_values = symbol_df['timestamp']
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f'{selected_symbol} Price' if selected_symbol else 'Price', 'Volume')
                )
                
                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=x_values,
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
                            x=x_values,
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
                    showlegend=False,
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Missing required columns for candlestick chart. Found: {list(symbol_df.columns)}")
                st.dataframe(symbol_df.head(), use_container_width=True)
    
    with tab2:
        st.markdown("#### Raw Data Table")
        
        # Filtering options
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'symbol' in df_display.columns:
                try:
                    filter_symbol = st.multiselect("Filter by Symbol", df_display['symbol'].unique())
                except Exception:
                    filter_symbol = []
            else:
                filter_symbol = []
        with col2:
            n_rows = st.number_input("Number of rows", min_value=10, max_value=10000, value=100)
        with col3:
            # Dynamic sort options based on available columns
            sort_options = [col for col in ['timestamp', 'symbol', 'close', 'date'] if col in df_display.columns]
            if not sort_options:
                sort_options = df_display.columns.tolist()[:5]  # First 5 columns as fallback
            sort_order = st.selectbox("Sort by", sort_options)
        
        # Apply filters
        display_df = df_display.copy()
        if filter_symbol and 'symbol' in df_display.columns:
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
        if 'symbol' in df_display.columns:
            try:
                for symbol in df_display['symbol'].unique()[:5]:  # Show first 5 symbols
                    symbol_df = df_display[df_display['symbol'] == symbol]
                    
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
            except Exception as e:
                st.error(f"Error displaying symbol statistics: {e}")
        else:
            # Overall statistics for non-symbol data
            st.dataframe(df_display.describe(), use_container_width=True)
    
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
                try:
                    symbols = df['symbol'].unique().tolist() if 'symbol' in df.columns else st.session_state.get('symbols', [])
                except Exception:
                    symbols = st.session_state.get('symbols', [])
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
                sync_symbols_input()
                st.success("✅ Session data and symbols cleared!")
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
