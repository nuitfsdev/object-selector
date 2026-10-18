#!/usr/bin/env bash
set -euo pipefail

# CUDA 12.6 wheels run on RTX 3090 hosts whose NVIDIA driver supports CUDA >=12.6.
# Override when needed: TORCH_INDEX_URL=... bash vps_setup.sh
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install torch==2.12.1 --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the GPU. Check nvidia-smi and the CUDA wheel.")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
PY

echo "VPS environment is ready. Activate it with: source .venv/bin/activate"
