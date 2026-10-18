#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f .venv/bin/activate ]]; then
  bash vps_setup.sh
fi

if [[ ! -f cache/scene_objects.pt || ! -f cache/text_embeddings.pt ]]; then
  bash prepare_vps_data.sh "$@"
fi

bash run_vps_training.sh
