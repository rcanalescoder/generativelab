#!/usr/bin/env bash
# Arranca el backend en :8000 con el fallback de MPS activado (CLAUDE.md §0).
# Requiere el venv activado (ver README).
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec uvicorn app.main:app --reload --port 8000
