#!/bin/bash
# Quick script to run Streamlit app with authentication

cd "$(dirname "$0")/.."

if [ ! -f ".streamlit/secrets.toml" ]; then
    echo "❌ Error: .streamlit/secrets.toml not found"
    echo "Create it from .streamlit/secrets.toml.example and add your password"
    exit 1
fi

echo "🔒 Starting Streamlit with authentication..."
export ENABLE_AUTH=true
streamlit run app/HFT_Arbitrage_Lab.py --server.port 8501
