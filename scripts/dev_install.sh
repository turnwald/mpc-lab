#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .[dev]
echo "Dev environment ready. Activate with: source .venv/bin/activate"
