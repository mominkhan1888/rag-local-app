#!/usr/bin/env bash
set -euo pipefail

log() {
  printf "[setup] %s\n" "$1"
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if ! command -v python >/dev/null 2>&1; then
  log "Python not found. Please install Python 3.10+ and re-run."
  exit 1
fi

log "Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if ! command -v ollama >/dev/null 2>&1; then
  OS_NAME="$(uname -s)"
  if [[ "$OS_NAME" == "Linux" || "$OS_NAME" == "Darwin" ]]; then
    log "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
  else
    log "Ollama not found and auto-install is not supported on this OS."
    log "Please install Ollama manually, then re-run this script."
    exit 1
  fi
fi

log "Pulling Llama 3.2 3B model..."
ollama pull llama3.2:3b

log "Creating data folders..."
mkdir -p data/uploads data/chroma_db

if [[ ! -f .env ]]; then
  cp .env.example .env
  log "Created .env from .env.example"
fi

log "Setup complete. Run: streamlit run app.py --server.address 0.0.0.0 --server.port 8501"
