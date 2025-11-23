#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.password=''
