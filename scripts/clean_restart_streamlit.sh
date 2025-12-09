#!/bin/bash
# Clean restart script for Streamlit app

echo "🧹 Cleaning up any existing Streamlit processes..."
pkill -9 -f "streamlit run" 2>/dev/null
sleep 2

echo "🗑️ Clearing Streamlit cache..."
rm -rf ~/.streamlit/cache 2>/dev/null

echo "🚀 Starting fresh Streamlit app..."
cd "$(dirname "$0")/.."
streamlit run app/HFT_Arbitrage_Lab.py --server.port 8501 --server.headless true
