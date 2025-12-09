#!/bin/bash
# Quick script to run Streamlit app locally without authentication

cd "$(dirname "$0")/.."

echo "🚀 Starting Streamlit in standalone mode (no authentication)..."
export ENABLE_AUTH=false
streamlit run app/HFT_Arbitrage_Lab.py --server.port 8501
