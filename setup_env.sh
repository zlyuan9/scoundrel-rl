#!/usr/bin/env bash
# Create a local venv and install scoundrel-rl with common extras (tests, viewer, RL, plots).
# Usage: bash setup_env.sh && source .venv/bin/activate
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev,gui,rl,analysis]"
echo "Done. Activate with: source .venv/bin/activate"
