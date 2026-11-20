#!/usr/bin/env bash
# One-command CPU environment for the water flows (macOS Apple Silicon or Linux).
# Usage: bash flows/setup_env.sh   (run from the repository root)
set -euo pipefail
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
uv venv -p 3.12 .venv
uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv -e ./skala -r flows/requirements.txt
.venv/bin/python -c "import skala, torch, pyscf; print('skala ok | torch', torch.__version__, '| pyscf', pyscf.__version__)"
echo "Done. Try: .venv/bin/python -m flows.water opt freq"
