#!/bin/bash
# Run the Multi-Strategy HFT Trading Platform

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Multi-Strategy HFT Trading Platform                   ║${NC}"
echo -e "${BLUE}║   Rust HFT Arbitrage Lab                                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo -e "${GREEN}✓${NC} Python version: ${PYTHON_VERSION}"

# Verify minimum version (3.7+)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 7 ]); then
    echo -e "${YELLOW}Error: Python 3.7 or higher is required${NC}"
    echo -e "${YELLOW}You are using Python ${PYTHON_VERSION}${NC}"
    exit 1
fi

# Warn if not using recommended version (3.8+)
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; then
    echo -e "${YELLOW}⚠${NC}  Python 3.8+ recommended for best performance"
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo -e "${YELLOW}Warning: Streamlit not found. Installing requirements...${NC}"
    pip install -r app/requirements.txt
fi

# Check if api_keys.properties exists
if [ ! -f "api_keys.properties" ]; then
    echo -e "${YELLOW}Warning: api_keys.properties not found${NC}"
    echo -e "${YELLOW}Some features may not work without API keys${NC}"
    echo -e "${YELLOW}Copy api_keys.properties.example to api_keys.properties and add your keys${NC}"
    echo ""
fi

# Check if Rust acceleration is available
if python3 -c "import rust_connector" &> /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Rust acceleration enabled (10-100x speedup)"
else
    echo -e "${YELLOW}⚠${NC} Rust acceleration not available (using pure Python)"
    echo -e "  ${YELLOW}Build rust_connector for better performance${NC}"
fi

echo ""
echo -e "${GREEN}Starting application...${NC}"
echo ""
echo -e "Access the app at: ${BLUE}http://localhost:8501${NC}"
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop"
echo ""

# Run Streamlit
cd "$(dirname "$0")"
streamlit run app/main_app.py
