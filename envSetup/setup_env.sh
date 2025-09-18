#!/usr/bin/env bash
set -euo pipefail

# Config
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[1/4] Creating venv at ${VENV_DIR}"
$PYTHON_BIN -m venv "${VENV_DIR}"

# Activate
if [ -f "${VENV_DIR}/bin/activate" ]; then
  source "${VENV_DIR}/bin/activate"
elif [ -f "${VENV_DIR}\Scripts\activate" ]; then
  # Windows Git Bash fallback
  source "${VENV_DIR}\Scripts\activate"
else:
  echo "Could not find activate script in ${VENV_DIR}"
  exit 1
fi

echo "[2/4] Upgrading pip & wheel"
python -m pip install --upgrade pip wheel

echo "[3/4] Installing core requirements"
pip install -r requirements.txt

echo "[4/4] Verifying CasADi solver plugins and running a smoke test"
python verify_env.py

cat <<'NOTE'

Environment setup complete.

To activate later:
  source ${VENV_DIR}/bin/activate          # Linux/macOS
  # or on Windows (PowerShell):
  #   .\\${VENV_DIR}\\Scripts\\Activate.ps1

Optional ML stack (PyTorch):
  pip install -r requirements-optional.txt

NOTE
