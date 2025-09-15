#!/usr/bin/env bash
# Usage: bash scripts/bootstrap_git.sh https://github.com/<you>/mpc-lab.git
set -euo pipefail
REMOTE_URL="${1:-}"
if [ -z "$REMOTE_URL" ]; then
  echo "Provide remote URL, e.g. https://github.com/<you>/mpc-lab.git"
  exit 1
fi
git init
git add .
git commit -m "Initial commit: MPC+Koopman+CBF scaffold"
git branch -M main
git remote add origin "$REMOTE_URL"
git push -u origin main
