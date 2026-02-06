#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# GLM-OCR local server launcher (Linux/macOS)
# - Creates/uses .venv next to this script
# - Installs runtime deps (including transformers dev build)
# - Starts FastAPI on configured host/port
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODEL_CACHE_DIR="$SCRIPT_DIR/models/hf_cache"
HF_HOME_DIR="$SCRIPT_DIR/models/hf_home"
ENV_FILE="$SCRIPT_DIR/.env"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "[!] Python is not installed. Install Python 3.10+ and retry."
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  echo "[+] Loading .env from $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

mkdir -p "$MODEL_CACHE_DIR" "$HF_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export GLM_MODEL_CACHE="$MODEL_CACHE_DIR"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "[+] Creating virtual environment at $VENV_DIR ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

TORCH_CHANNEL="${TORCH_CHANNEL:-}"
if [[ -z "$TORCH_CHANNEL" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    TORCH_CHANNEL="cu126"
  else
    TORCH_CHANNEL="cpu"
  fi
fi

echo "[+] Installing/ensuring dependencies..."
python -m pip install --upgrade pip

echo "[+] Installing PyTorch ($TORCH_CHANNEL)..."
if [[ "$TORCH_CHANNEL" == "cpu" ]]; then
  python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision
else
  if ! python -m pip install --upgrade --index-url "https://download.pytorch.org/whl/$TORCH_CHANNEL" torch torchvision; then
    echo "[!] Failed with TORCH_CHANNEL=$TORCH_CHANNEL. Falling back to cpu."
    python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision
  fi
fi

python - <<'PY'
import torch
print("[torch]", torch.__version__, "cuda=", torch.version.cuda, "available=", torch.cuda.is_available())
PY

echo "[+] Installing FastAPI and image/PDF dependencies..."
python -m pip install fastapi uvicorn python-multipart pillow pypdfium2 accelerate

echo "[+] Installing transformers (development build)..."
python -m pip install git+https://github.com/huggingface/transformers.git

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "[+] Starting server at http://$HOST:$PORT"
exec uvicorn app.main:app --host "$HOST" --port "$PORT"
