"""
Data Loading Module
===================

Load historical market data from multiple sources:
- Finnhub (primary)
- Yahoo Finance (fallback)
- CSV upload (custom data)
- Mock/Synthetic data (testing)

Features:
- Stackable data loading (append new queries to existing data)
- Persistent storage to /data folder for long-living sessions
- Rate-limit aware fetching for intraday data
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
from python.data_persistence import (
    save_dataset, load_dataset, list_datasets, delete_dataset,
    stack_data, generate_dataset_name, get_total_storage_size, format_size
)
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
        st.session_state.theme_mode = 'light'
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'symbols' not in st.session_state:
        st.session_state.symbols = ["AAPL", "MSFT", "GOOGL"]  # Default symbols
    if 'data_load_mode' not in st.session_state:
        st.session_state.data_load_mode = "replace"  # 'replace', 'append', 'update'
    
    st.title("📊 Historical Data Loading")
    st.markdown("Load and preview market data for backtesting strategies")
    
    # Main tabs for different sections
    main_tab1, main_tab2, main_tab3 = st.tabs(["📥 Fetch Data", "💾 Saved Datasets", "🔗 Merge/Append"])
    
    with main_tab1:
        render_fetch_tab()
    
    with main_tab2:
        render_saved_datasets_tab()
    
    with main_tab3:
        render_merge_append_tab()


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
                        st.rerun()
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
        data_source = st.selectbox(
            "🔌 Data Source",
            [
                "CCXT - Crypto Exchanges (FREE! ⭐)", 
                "Yahoo Finance", 
                "Finnhub (API)", 
                "Alpha Vantage (API - FREE 25 calls/day)",
                "Massive (Institutional-grade - FREE 100 calls/day)",
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
        
        # Massive rate limit warnings and method selection
        # Initialize in session state for global access
        if 'massive_fetch_method' not in st.session_state:
            st.session_state.massive_fetch_method = "auto"
        if 'massive_data_type' not in st.session_state:
            st.session_state.massive_data_type = "ohlcv"
        
        if data_source.startswith("Massive"):
            # Important notice about Massive.com availability
            st.warning(
                "⚠️ **Massive.com API Notice**\\n\\n"
                "Massive.com is currently a placeholder/example service for demonstration purposes. "
                "The REST API and WebSocket endpoints are not yet publicly available.\\n\\n"
                "**🔄 Working Alternatives:**\\n"
                "• **CCXT** - Free real-time crypto data (Binance, Kraken, etc.)\\n"
                "• **Yahoo Finance** - Free stock data\\n"
                "• **Finnhub** - Free stock market data with API key\\n\\n"
                "**Flat File Downloads** may work if you have valid S3 credentials from a compatible provider."
            )
            
            # Initialize rate limit tracking in session state
            if 'massive_calls_today' not in st.session_state:
                st.session_state.massive_calls_today = 0
                st.session_state.massive_last_reset = datetime.now().date()
            
            # Reset daily counter if new day
            if st.session_state.massive_last_reset < datetime.now().date():
                st.session_state.massive_calls_today = 0
                st.session_state.massive_last_reset = datetime.now().date()
            
            # Method selection
            st.markdown("#### 📊 Fetch Method")
            st.session_state.massive_fetch_method = st.radio(
                "How to fetch data",
                ["auto", "rest", "flat_file"],
                index=["auto", "rest", "flat_file"].index(st.session_state.massive_fetch_method) if st.session_state.massive_fetch_method in ["auto", "rest", "flat_file"] else 0,
                format_func=lambda x: {
                    "auto": "🤖 Auto (Smart Selection)",
                    "rest": "📡 REST API (Fast, small queries)",
                    "flat_file": "📦 Flat Files / S3 (Bulk downloads)"
                }[x],
                help=(
                    "**Auto**: Automatically chooses best method based on query size\\n"
                    "  - REST API for ≤5 symbols AND ≤7 days\\n"
                    "  - Flat Files for larger queries\\n\\n"
                    "**REST API**: Fast for small queries, counts against 100 calls/day\\n\\n"
                    "**Flat Files (S3)**: Best for bulk data (>10 symbols or >30 days), "
                    "uses S3 credentials (separate from API key), 10 GB/month quota"
                )
            )
            
            # Data type selection for flat files
            if st.session_state.massive_fetch_method == "flat_file":
                st.session_state.massive_data_type = st.radio(
                    "Data Type",
                    ["ohlcv", "trades"],
                    format_func=lambda x: {
                        "ohlcv": "📊 OHLCV Bars (Aggregated candles)",
                        "trades": "⚡ Tick-Level Trades (Raw trade data)"
                    }[x],
                    help=(
                        "**OHLCV Bars**: Aggregated data (open, high, low, close, volume) for backtesting\\n"
                        "**Tick-Level Trades**: Raw trade-by-trade data from SIP feed (US stocks only, very large files)"
                    ),
                    horizontal=True
                )
            
            # Display rate limit status
            calls_remaining = 100 - st.session_state.massive_calls_today
            
            if st.session_state.massive_fetch_method in ["auto", "rest"]:
                if calls_remaining > 50:
                    st.info(
                        f"🏛️ **Massive.com Free Tier Status:**\\n"
                        f"✅ {calls_remaining}/100 REST API calls remaining today\\n"
                        f"⏱️ Rate limit: 10 calls/minute (6 sec between calls)\\n"
                        f"🔌 WebSocket: 10 concurrent connections, 100 messages/min\\n"
                        f"📥 Bulk Files: 10 GB/month downloads available\\n"
                        f"💡 Institutional-grade data with generous free tier!"
                    )
                elif calls_remaining > 20:
                    st.warning(
                        f"⚠️ **Massive.com Free Tier Status:**\\n"
                        f"🔶 {calls_remaining}/100 REST API calls remaining today\\n"
                        f"⏱️ Rate limit: 10 calls/minute (6 sec between calls)\\n"
                        f"💡 Consider using WebSocket streaming or flat file downloads\\n"
                        f"📥 10 GB/month bulk downloads available"
                    )
                elif calls_remaining > 0:
                    st.error(
                        f"🚨 **Massive.com Free Tier Status:**\\n"
                        f"🔴 Only {calls_remaining}/100 REST API calls left today!\\n"
                        f"⏱️ Rate limit: 10 calls/minute (6 sec between calls)\\n"
                        f"💡 Switch to flat file downloads\\n"
                        f"📥 10 GB/month still available for flat files"
                    )
                else:
                    st.error(
                        f"🚫 **Massive.com Daily REST Limit Reached!**\\n"
                        f"❌ 0/100 REST API calls remaining\\n"
                        f"⏰ Limit resets at midnight UTC\\n"
                        f"💡 Alternative: Use flat files (10 GB/month)\\n"
                        f"📥 Get S3 credentials from Massive dashboard"
                    )
                    # Auto-switch to flat files if REST limit reached
                    if st.session_state.massive_fetch_method == "auto":
                        st.info("🤖 Auto-switching to flat files since REST limit reached")
                        st.session_state.massive_fetch_method = "flat_file"
            
            if st.session_state.massive_fetch_method == "flat_file":
                st.success(
                    f"📦 **Using Flat Files (S3 Downloads)**\\n"
                    f"✅ Doesn't count against REST API quota\\n"
                    f"📥 10 GB/month download limit\\n"
                    f"💡 Requires S3 credentials from 'Accessing Flat Files (S3)' tab\\n"
                    f"⚙️ Add to api_keys.properties: MASSIVE_S3_ACCESS_KEY_ID, MASSIVE_S3_SECRET_ACCESS_KEY"
                )
                
                # Show what's available for bulk download
                with st.expander("📋 View Available Flat File Data", expanded=False):
                    if st.session_state.massive_data_type == "ohlcv":
                        st.markdown("""
                        **Available OHLCV Bar Data on Massive S3:**
                        
                        **Equities (US Stocks):**
                        - 🕐 Minute-level data: 2000-present
                        - 📈 Hourly data: 2000-present  
                        - 📊 Daily data: 1970-present
                        - 📅 Weekly data: 1970-present
                        
                        **Coverage:**
                        - All NYSE, NASDAQ, AMEX listed stocks
                        - ~8,000+ active tickers
                        - Corporate actions adjusted
                        - Institutional-grade quality
                        
                        **File Organization:**
                        ```
                        equities/{interval}/{SYMBOL}/{YYYY-MM}.parquet
                        ```
                        
                        **Typical File Sizes:**
                        - 1 month minute data (1 symbol): ~50-200 MB
                        - 1 month daily data (1 symbol): ~1-5 MB
                        - 1 year daily data (1 symbol): ~10-50 MB
                        
                        **Performance (Rust Backend):**
                        - Small files (<1GB): Polars engine (50-100x faster than Python)
                        - Large files (>1GB): DataFusion streaming
                        - Automatic engine selection
                        
                        **Recommended Usage:**
                        - ≤10 symbols, ≤30 days: Use REST API
                        - >10 symbols OR >30 days: Use Flat Files
                        - Backtesting (100s of symbols): Use Flat Files
                        """)
                    else:  # trades
                        st.markdown("""
                        **Available Tick-Level Trade Data on Massive S3:**
                        
                        **US Stocks SIP (Securities Information Processor):**
                        - ⚡ Tick-by-tick trade data
                        - 🕐 Coverage: 2020-present
                        - 📍 Source: Consolidated SIP feed (all exchanges)
                        - 📊 Fields: timestamp, symbol, price, size, exchange, conditions
                        
                        **Coverage:**
                        - All NYSE, NASDAQ, AMEX listed stocks
                        - Every single trade execution
                        - Sub-millisecond timestamps
                        - Exchange identifiers and trade conditions
                        
                        **File Organization:**
                        ```
                        us_stocks_sip/trades_v1/{SYMBOL}/{YYYY-MM-DD}.parquet
                        ```
                        
                        **⚠️ Large File Sizes:**
                        - 1 day tick data (liquid stock): 100-500 MB
                        - 1 day tick data (very liquid): 500 MB - 2 GB
                        - 1 month tick data: 3-15 GB per symbol
                        
                        **Performance:**
                        - Rust backend automatically uses DataFusion for streaming
                        - Processes millions of ticks efficiently
                        - Filters applied before loading into memory
                        
                        **⚠️ Recommended Usage:**
                        - HFT strategy development and backtesting
                        - Market microstructure research
                        - Order flow analysis
                        - **Start with 1-2 days for testing due to large file sizes**
                        - Monitor your 10 GB/month quota carefully
                        """)
        
        # Alpha Vantage rate limit warnings
        elif data_source.startswith("Alpha Vantage"):
            # Initialize rate limit tracking in session state
            if 'av_calls_today' not in st.session_state:
                st.session_state.av_calls_today = 0
                st.session_state.av_last_reset = datetime.now().date()
            
            # Reset daily counter if new day
            if st.session_state.av_last_reset < datetime.now().date():
                st.session_state.av_calls_today = 0
                st.session_state.av_last_reset = datetime.now().date()
            
            # Display rate limit status
            calls_remaining = 25 - st.session_state.av_calls_today
            
            if calls_remaining > 15:
                st.info(
                    f"📊 **Alpha Vantage Free Tier Status:**\\n"
                    f"✅ {calls_remaining}/25 API calls remaining today\\n"
                    f"⏱️ Rate limit: 5 calls/minute (12 sec between calls)\\n"
                    f"💡 Upgrade at alphavantage.co for more calls"
                )
            elif calls_remaining > 5:
                st.warning(
                    f"⚠️ **Alpha Vantage Free Tier Status:**\\n"
                    f"🔶 {calls_remaining}/25 API calls remaining today\\n"
                    f"⏱️ Rate limit: 5 calls/minute (12 sec between calls)\\n"
                    f"💡 Use wisely - limit resets at midnight UTC"
                )
            elif calls_remaining > 0:
                st.error(
                    f"🚨 **Alpha Vantage Free Tier Status:**\\n"
                    f"🔴 Only {calls_remaining}/25 API calls left today!\\n"
                    f"⏱️ Rate limit: 5 calls/minute (12 sec between calls)\\n"
                    f"⚠️ Daily limit resets at midnight UTC"
                )
            else:
                st.error(
                    f"🚫 **Alpha Vantage Daily Limit Reached!**\\n"
                    f"❌ 0/25 API calls remaining\\n"
                    f"⏰ Limit resets at midnight UTC\\n"
                    f"💡 Use Yahoo Finance or CCXT as alternative"
                )
            
            # Show recommendation for symbol limits
            st.info(
                "💡 **Free Tier Tips:**\\n"
                "• Fetch 1-5 symbols at a time to stay within limits\\n"
                "• Use 'compact' output (last 100 points)\\n"
                "• Save datasets for reuse to avoid re-fetching\\n"
                "• Consider Yahoo Finance for unlimited stock data"
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
            
            # Show download preview for Massive flat files
            if data_source.startswith("Massive") and st.session_state.massive_fetch_method == "flat_file":
                st.markdown("---")
                st.markdown("#### 📋 Download Preview")
                
                days_diff = (end_date - start_date).days
                num_symbols = len(symbols)
                
                # Estimate download size
                if st.session_state.massive_data_type == "trades":
                    # Tick-level trade data is MUCH larger
                    est_mb_per_symbol = days_diff * 300  # ~300 MB per symbol per day for tick data
                    data_density = "⚡ Tick-Level Trades"
                elif interval in ['1m', '5m', '15m', '30m']:
                    est_mb_per_symbol = days_diff * 10  # ~10 MB per symbol per day
                    data_density = "Minute-level"
                elif interval in ['1h', '4h']:
                    est_mb_per_symbol = days_diff * 1  # ~1 MB per symbol per day
                    data_density = "Hourly"
                else:
                    est_mb_per_symbol = days_diff * 0.5  # ~0.5 MB per symbol per day
                    data_density = "Daily/Weekly"
                
                total_est_mb = num_symbols * est_mb_per_symbol
                
                # Show preview
                preview_col1, preview_col2, preview_col3 = st.columns(3)
                with preview_col1:
                    st.metric("Symbols", f"{num_symbols}")
                with preview_col2:
                    st.metric("Days", f"{days_diff}")
                with preview_col3:
                    st.metric("Est. Download", f"~{total_est_mb:.1f} MB")
                
                st.info(
                    f"📊 **{data_density} data** for **{num_symbols} symbol(s)** "
                    f"over **{days_diff} days**\\n\\n"
                    f"🦀 Rust backend: {('Polars (fast)' if total_est_mb < 1000 else 'DataFusion (streaming)')}\\n"
                    f"⏱️ Estimated time: {max(2, total_est_mb / 50):.0f}-{max(5, total_est_mb / 20):.0f} seconds\\n"
                    f"💾 Quota used: {total_est_mb:.1f} MB of 10 GB monthly limit"
                )
                
                # Warning if approaching quota
                if total_est_mb > 5000:  # > 5 GB
                    st.warning(
                        f"⚠️ This download will use **{total_est_mb/1024:.1f} GB** "
                        f"of your monthly quota!\\n"
                        f"Consider reducing the date range or number of symbols."
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
                        'alpha': 'alpha_vantage',
                        'massive': 'massive',
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
            - **Alpha Vantage**: FREE 25 API calls/day (stocks, forex, crypto)
              - ⚠️ Free tier: 5 calls/minute, 25 calls/day
              - Real-time quotes (15-20 min delay)
              - Intraday & daily historical data
              - Requires free API key from alphavantage.co
            - **🏛️ Massive**: Institutional-grade market data (FREE 100 calls/day)
              - ⚡ REST API: 100 requests/day, 10/minute
              - 🔌 WebSocket: 10 concurrent connections, 100 messages/min
              - 📥 Bulk Files: 10 GB/month historical data downloads
              - Stocks, options, futures, forex, crypto
              - Real-time and historical data
              - Requires free API key from massive.com
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
                new_df = _fetch_ccxt(symbols, start, end, interval, exchange_id)
            # For Massive, use transparent fetch_data with method selection
            elif source == 'massive':
                from python.massive_helper import fetch_data
                new_df = fetch_data(
                    symbols=symbols,
                    start=start,
                    end=end,
                    interval=interval,
                    method=st.session_state.get('massive_fetch_method', 'auto'),  # "auto", "rest", or "flat_file"
                    data_type=st.session_state.get('massive_data_type', 'ohlcv') if st.session_state.get('massive_fetch_method', 'auto') == "flat_file" else "ohlcv"  # "ohlcv" or "trades"
                )
            else:
                new_df = fetch_intraday_data(
                    symbols=symbols,
                    start=start,
                    end=end,
                    interval=interval,
                    source=source
                )
            
            # Increment API call counter for rate-limited sources
            if source == 'massive' and st.session_state.get('massive_fetch_method', 'auto') in ["auto", "rest"]:
                # Only count REST API calls, not flat file downloads
                if 'massive_calls_today' in st.session_state:
                    st.session_state.massive_calls_today += len(symbols)
            elif source == 'alpha_vantage':
                if 'av_calls_today' in st.session_state:
                    st.session_state.av_calls_today += len(symbols)
            
            # Reset index to make it easier to work with
            if isinstance(new_df.index, pd.MultiIndex):
                new_df = new_df.reset_index()
            
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

def display_data_preview():
    """Display data preview and visualization"""
    df = st.session_state.historical_data
    
    # Validate dataframe
    if df is None or df.empty:
        st.warning("No data available to preview.")
        return
    
    st.markdown("### 📋 Data Preview & Visualization")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📑 Data Table", "📈 Statistics", "💾 Export"])
    
    with tab1:
        # Symbol selector for charting
        if 'symbol' in df.columns:
            try:
                symbols = df['symbol'].unique().tolist()
                selected_symbol = st.selectbox("Select Symbol for Chart", symbols)
                # Filter data for selected symbol
                symbol_df = df[df['symbol'] == selected_symbol].copy()
            except Exception as e:
                st.error(f"Error accessing symbol column: {e}")
                symbol_df = df.copy()
                selected_symbol = None
        else:
            st.info("Data does not contain a 'symbol' column. Showing all data.")
            symbol_df = df.copy()
            selected_symbol = None
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
            if 'symbol' in df.columns:
                try:
                    filter_symbol = st.multiselect("Filter by Symbol", df['symbol'].unique())
                except Exception:
                    filter_symbol = []
            else:
                filter_symbol = []
        with col2:
            n_rows = st.number_input("Number of rows", min_value=10, max_value=10000, value=100)
        with col3:
            # Dynamic sort options based on available columns
            sort_options = [col for col in ['timestamp', 'symbol', 'close', 'date'] if col in df.columns]
            if not sort_options:
                sort_options = df.columns.tolist()[:5]  # First 5 columns as fallback
            sort_order = st.selectbox("Sort by", sort_options)
        
        # Apply filters
        display_df = df.copy()
        if filter_symbol and 'symbol' in df.columns:
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
        if 'symbol' in df.columns:
            try:
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
            except Exception as e:
                st.error(f"Error displaying symbol statistics: {e}")
    
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
