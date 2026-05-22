#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

# Install deps
pip install -q -r requirements.txt

# Launch
streamlit run app.py --server.port 8501 --server.address localhost
