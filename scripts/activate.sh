#!/usr/bin/env bash
# small convenience script: source this to activate project venv
# Navigate to project root first
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "✓ Virtual environment activated"
else
  echo ".venv not found. Create it with: python -m venv .venv"
fi