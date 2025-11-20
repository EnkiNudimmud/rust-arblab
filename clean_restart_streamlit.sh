#!/bin/bash
# Clean restart script for Streamlit app

echo "🧹 Cleaning up any existing Streamlit processes..."
pkill -9 -f "streamlit run" 2>/dev/null
sleep 2

echo "🗑️ Clearing Streamlit cache..."
rm -rf ~/.streamlit/cache 2>/dev/null

echo "🚀 Starting fresh Streamlit app..."
cd /Users/melvinalvarez/Documents/Workspace/rust-hft-arbitrage-lab
streamlit run app/main_app.py --server.port 8501 --server.headless true
