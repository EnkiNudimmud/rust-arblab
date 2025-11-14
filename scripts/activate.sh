#!/usr/bin/env bash
# small convenience script: source this to activate project venv
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo ".venv not found. Create it with: make create-venv"
fi