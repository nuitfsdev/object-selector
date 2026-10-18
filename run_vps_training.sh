#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
mkdir -p logs cache runs

if [[ ! -f cache/scene_objects.pt || ! -f cache/text_embeddings.pt ]]; then
  echo "Prepared caches are missing." >&2
  echo "Run: bash prepare_vps_data.sh --accept-scannet-tos" >&2
  echo "Or run the complete pipeline: bash run_vps_all.sh --accept-scannet-tos" >&2
  exit 1
fi

python train.py --config config.3090.json 2>&1 | tee "logs/train_$(date +%Y%m%d_%H%M%S).log"
