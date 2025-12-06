# Multi-stage optimized Dockerfile for HFT Arbitrage Lab

# ============================================
# Stage 1: Rust Builder (gRPC + Connectors)
# ============================================
FROM rust:1.82-slim AS rust-builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config libssl-dev clang cmake \
    protobuf-compiler libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy gRPC server files
COPY hft-grpc-server/ ./hft-grpc-server/
COPY proto/ ./proto/

# Build gRPC server
RUN cd hft-grpc-server && cargo build --release

# ============================================
# Stage 2: Python Builder (Maturin + PyO3)
# ============================================
FROM python:3.11-slim AS python-builder

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl pkg-config libssl-dev \
    python3-dev patchelf \
    && rm -rf /var/lib/apt/lists/*

# Install Rust for maturin
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal

WORKDIR /workspace

# Install maturin
RUN pip install --no-cache-dir maturin

# Copy rust_connector files
COPY rust_connector/ ./rust_connector/

# Build wheel
RUN cd rust_connector && maturin build --release

# ============================================
# Stage 3: Runtime (Lightweight)
# ============================================
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 libgomp1 libssl3 netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python dependencies file and install
COPY docker/requirements.txt /app/docker/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r docker/requirements.txt

# Copy pre-built binaries from builders
COPY --from=rust-builder /workspace/hft-grpc-server/target/release/hft-server /app/target/release/hft-server
COPY --from=python-builder /workspace/rust_connector/target/wheels/*.whl /tmp/wheels/

# Install rust_connector wheel
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Copy application code
COPY app/ /app/app/
COPY python/ /app/python/
COPY scripts/ /app/scripts/
COPY examples/ /app/examples/

EXPOSE 8501 8888 50051

# Default: Streamlit
CMD ["streamlit", "run", "app/HFT_Arbitrage_Lab.py", "--server.address=0.0.0.0"]
