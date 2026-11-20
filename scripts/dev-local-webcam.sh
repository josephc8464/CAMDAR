#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${CAMDAR_VENV_DIR:-venv}"
VENV_PATH="${ROOT_DIR}/${VENV_DIR}"
PYTHON_BIN="${VENV_PATH}/bin/python3"

CONFIG="${CAMDAR_DEV_CONFIG:-configs/iter1.yaml}"
SOURCE="${CAMDAR_DEV_SOURCE:-0}"
HOST="${CAMDAR_DEV_HOST:-0.0.0.0}"
PORT="${CAMDAR_DEV_PORT:-8080}"

cd "${ROOT_DIR}"

if [[ -f "${VENV_PATH}/Scripts/python.exe" ]]; then 
  PYTHON_BIN="${VENV_PATH}/Scripts/python.exe"; 
else 
  PYTHON_BIN="${VENV_PATH}/bin/python3"; 
fi

if ! "${PYTHON_BIN}" -c "import camdar" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install -e .
fi

exec "${PYTHON_BIN}" -m camdar.ui.basic_web.app \
  --config "${CONFIG}" \
  --source "${SOURCE}" \
  --host "${HOST}" \
  --port "${PORT}"
