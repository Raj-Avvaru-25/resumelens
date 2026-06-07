#!/usr/bin/env bash
# One-command setup + launch. Creates a venv, installs deps, starts the app.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtual environment (.venv)…"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing dependencies (first run also downloads ~200MB of local models on launch)…"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

echo
echo "Launching Resume RAG → http://localhost:8501"
echo "(Optional: put ANTHROPIC_API_KEY in a .env file to enable the Claude-powered modes.)"
exec streamlit run app.py
