#!/usr/bin/env bash

# Environment Setup Helper for rust-arblab
# This script detects your system and guides you through the setup process

set -e

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_header() {
    echo -e "${BLUE}===================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*)
            OS="macOS"
            PACKAGE_MANAGER="brew"
            ;;
        Linux*)
            OS="Linux"
            if command -v apt-get &> /dev/null; then
                PACKAGE_MANAGER="apt"
            elif command -v dnf &> /dev/null; then
                PACKAGE_MANAGER="dnf"
            elif command -v yum &> /dev/null; then
                PACKAGE_MANAGER="yum"
            else
                PACKAGE_MANAGER="unknown"
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            OS="Windows"
            PACKAGE_MANAGER="manual"
            ;;
        *)
            OS="Unknown"
            PACKAGE_MANAGER="unknown"
            ;;
    esac
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check Python version
check_python() {
    print_info "Checking Python installation..."
    
    if command_exists python3.11; then
        PYTHON_CMD="python3.11"
        PYTHON_VERSION=$(python3.11 --version | awk '{print $2}')
        print_success "Python 3.11 found: $PYTHON_VERSION"
        return 0
    elif command_exists python3; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
            PYTHON_CMD="python3"
            print_success "Python found: $PYTHON_VERSION"
            return 0
        else
            print_warning "Python $PYTHON_VERSION found, but 3.11+ required"
            return 1
        fi
    else
        print_error "Python 3.11+ not found"
        return 1
    fi
}

# Check Rust installation
check_rust() {
    print_info "Checking Rust installation..."
    
    if command_exists rustc && command_exists cargo; then
        RUST_VERSION=$(rustc --version | awk '{print $2}')
        print_success "Rust found: $RUST_VERSION"
        return 0
    else
        print_error "Rust not found"
        return 1
    fi
}

# Check Docker installation
check_docker() {
    print_info "Checking Docker installation..."
    
    if command_exists docker; then
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
        print_success "Docker found: $DOCKER_VERSION"
        
        if docker compose version &> /dev/null; then
            print_success "Docker Compose (v2) available"
            return 0
        elif command_exists docker-compose; then
            print_success "Docker Compose (v1) available"
            return 0
        else
            print_warning "Docker Compose not found"
            return 1
        fi
    else
        print_warning "Docker not found (optional)"
        return 1
    fi
}

# Check virtual environment
check_venv() {
    print_info "Checking virtual environment..."
    
    if [ -n "$VIRTUAL_ENV" ]; then
        print_success "Virtual environment active: $VIRTUAL_ENV"
        return 0
    elif [ -n "$CONDA_DEFAULT_ENV" ]; then
        print_success "Conda environment active: $CONDA_DEFAULT_ENV"
        return 0
    else
        print_warning "No virtual environment active"
        return 1
    fi
}

# Check if rust_connector is installed
check_rust_connector() {
    print_info "Checking rust_connector installation..."
    
    if $PYTHON_CMD -c "import rust_connector" 2>/dev/null; then
        print_success "rust_connector installed"
        return 0
    else
        print_warning "rust_connector not installed"
        return 1
    fi
}

# Installation suggestions
suggest_installation() {
    print_header "Installation Suggestions"
    
    case "$OS" in
        macOS)
            if ! command_exists brew; then
                echo ""
                echo "Install Homebrew first:"
                echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                echo ""
            fi
            
            if ! check_python; then
                echo ""
                echo "Install Python 3.11:"
                echo "  brew install python@3.11"
                echo ""
            fi
            
            if ! check_rust; then
                echo ""
                echo "Install Rust:"
                echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
                echo "  source \$HOME/.cargo/env"
                echo ""
            fi
            ;;
        
        Linux)
            if [ "$PACKAGE_MANAGER" = "apt" ]; then
                if ! check_python || ! check_rust; then
                    echo ""
                    echo "Install dependencies:"
                    echo "  sudo apt-get update"
                    echo "  sudo apt-get install -y build-essential curl git pkg-config libssl-dev python3.11 python3.11-dev python3-pip"
                    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
                    echo "  source \$HOME/.cargo/env"
                    echo ""
                fi
            elif [ "$PACKAGE_MANAGER" = "dnf" ]; then
                if ! check_python || ! check_rust; then
                    echo ""
                    echo "Install dependencies:"
                    echo "  sudo dnf install -y gcc gcc-c++ make curl git pkg-config openssl-devel python3.11 python3.11-devel"
                    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
                    echo "  source \$HOME/.cargo/env"
                    echo ""
                fi
            fi
            ;;
        
        Windows)
            echo ""
            echo "For Windows, we recommend using Docker:"
            echo "  1. Install Docker Desktop from https://www.docker.com/products/docker-desktop"
            echo "  2. Run: docker compose up --build"
            echo ""
            echo "Or install manually:"
            echo "  1. Install Python 3.11 from https://www.python.org/downloads/"
            echo "  2. Install Rust from https://rustup.rs/"
            echo "  3. Install Visual Studio Build Tools"
            echo ""
            ;;
    esac
}

# Setup Python environment
setup_python_env() {
    print_header "Setting up Python Environment"
    
    # Check if we're already in a venv
    if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_DEFAULT_ENV" ]; then
        print_success "Already in a virtual environment"
        read -p "Continue with current environment? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Please activate your desired environment and run this script again"
            exit 0
        fi
    else
        # Ask user which environment to create
        echo ""
        echo "Choose Python environment:"
        echo "  1) venv (lightweight, built-in)"
        echo "  2) conda (recommended for macOS)"
        echo "  3) Skip (I'll set it up manually)"
        echo ""
        read -p "Enter choice [1-3]: " choice
        
        case $choice in
            1)
                print_info "Creating venv environment..."
                $PYTHON_CMD -m venv .venv
                print_success "venv created in .venv/"
                print_info "Activate with: source .venv/bin/activate"
                # Activate for this script
                source .venv/bin/activate
                ;;
            2)
                if command_exists conda; then
                    print_info "Creating conda environment..."
                    conda create -n rhftlab python=3.11 -y
                    print_success "Conda environment 'rhftlab' created"
                    print_info "Activate with: conda activate rhftlab"
                    print_warning "Please activate the conda environment and run this script again"
                    exit 0
                else
                    print_error "Conda not found. Install Miniconda or Anaconda first."
                    exit 1
                fi
                ;;
            3)
                print_info "Skipping environment creation"
                ;;
            *)
                print_error "Invalid choice"
                exit 1
                ;;
        esac
    fi
}

# Install Python dependencies
install_python_deps() {
    print_header "Installing Python Dependencies"
    
    print_info "Upgrading pip, setuptools, wheel, maturin..."
    $PYTHON_CMD -m pip install --upgrade pip setuptools wheel maturin
    
    # Detect Python version and choose appropriate requirements file
    PY_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")
    
    if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 13 ]; then
        # Python 3.13+ - use Python 3.13 compatible requirements
        if [ -f "requirements-py313.txt" ]; then
            print_info "Detected Python 3.$PY_MINOR - using requirements-py313.txt..."
            $PYTHON_CMD -m pip install -r requirements-py313.txt
        else
            print_warning "requirements-py313.txt not found, falling back to app/requirements.txt"
            if [ -f "app/requirements.txt" ]; then
                $PYTHON_CMD -m pip install -r app/requirements.txt
            fi
        fi
    elif [ -f "docker/requirements.txt" ]; then
        print_info "Installing from docker/requirements.txt..."
        $PYTHON_CMD -m pip install -r docker/requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "docker/requirements.txt not found"
        exit 1
    fi
}

# Build Rust connector
build_rust_connector() {
    print_header "Building Rust Connector"
    
    if [ -f "rust_connector/Cargo.toml" ]; then
        print_info "Building rust_connector with maturin..."
        maturin develop --manifest-path rust_connector/Cargo.toml --release
        
        if check_rust_connector; then
            print_success "Rust connector built and installed successfully"
        else
            print_error "Rust connector build failed"
            exit 1
        fi
    else
        print_error "rust_connector/Cargo.toml not found"
        exit 1
    fi
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    # Check Python
    if check_python; then
        print_success "Python OK"
    else
        print_error "Python verification failed"
    fi
    
    # Check Rust connector
    if check_rust_connector; then
        print_success "rust_connector OK"
    else
        print_error "rust_connector verification failed"
    fi
    
    # Test connectors
    print_info "Testing connectors..."
    CONNECTORS=$($PYTHON_CMD -c "from python.rust_bridge import list_connectors; print(', '.join(list_connectors()))" 2>/dev/null)
    if [ -n "$CONNECTORS" ]; then
        print_success "Connectors available: $CONNECTORS"
    else
        print_warning "Could not list connectors"
    fi
}

# Main setup flow
main() {
    clear
    print_header "rust-arblab Environment Setup"
    echo ""
    echo "This script will help you set up your development environment."
    echo ""
    
    # Detect system
    detect_os
    print_info "Detected OS: $OS"
    print_info "Package Manager: $PACKAGE_MANAGER"
    echo ""
    
    # Check prerequisites
    print_header "Checking Prerequisites"
    
    HAS_PYTHON=false
    HAS_RUST=false
    HAS_DOCKER=false
    
    check_python && HAS_PYTHON=true
    check_rust && HAS_RUST=true
    check_docker && HAS_DOCKER=true
    
    echo ""
    
    # If missing prerequisites, suggest installation
    if [ "$HAS_PYTHON" = false ] || [ "$HAS_RUST" = false ]; then
        suggest_installation
        
        echo ""
        read -p "Have you installed the required dependencies? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Please install dependencies and run this script again"
            exit 0
        fi
        
        # Re-check
        check_python && HAS_PYTHON=true
        check_rust && HAS_RUST=true
    fi
    
    # Offer Docker option
    if [ "$HAS_DOCKER" = true ]; then
        echo ""
        echo "Docker is available. You can choose:"
        echo "  1) Docker setup (recommended, most consistent)"
        echo "  2) Local setup (faster iteration, requires proper Python/Rust)"
        echo ""
        read -p "Enter choice [1-2]: " docker_choice
        
        if [ "$docker_choice" = "1" ]; then
            print_header "Docker Setup"
            print_info "Building Docker image..."
            docker compose build
            print_success "Docker image built successfully"
            print_info "Start with: docker compose up"
            print_info "Access Streamlit at: http://localhost:8501"
            exit 0
        fi
    fi
    
    # Local setup
    if [ "$HAS_PYTHON" = false ] || [ "$HAS_RUST" = false ]; then
        print_error "Python 3.11+ and Rust are required for local setup"
        exit 1
    fi
    
    # Setup Python environment
    setup_python_env
    
    # Install dependencies
    install_python_deps
    
    # Build Rust connector
    build_rust_connector
    
    # Verify
    verify_installation
    
    # Success message
    print_header "Setup Complete!"
    echo ""
    print_success "Environment is ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Configure API keys (optional):"
    echo "     cp api_keys.properties.example api_keys.properties"
    echo "     # Edit api_keys.properties with your keys"
    echo ""
    echo "  2. Run Streamlit app:"
    echo "     streamlit run app/streamlit_app.py"
    echo ""
    echo "  3. Or run Jupyter notebooks:"
    echo "     jupyter notebook examples/notebooks/"
    echo ""
    echo "  4. Test WebSocket streaming:"
    echo "     python test_websocket.py"
    echo ""
    print_info "For more details, see ENVIRONMENT_SETUP.md"
}

# Run main
main
