#!/usr/bin/env bash
# Start Jupyter Notebook server

set -e

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Print banner with URL
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         🚀 Starting Jupyter Notebook Server 🚀                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "📍 Jupyter will be available at:"
echo ""
echo "   🌐 http://localhost:8888"
echo "   🌐 http://127.0.0.1:8888"
echo ""
echo "⚡ No password or token required"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.password='' --allow-root
