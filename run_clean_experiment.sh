#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

if [[ ! -f cache/scene_objects.pt || ! -f cache/text_embeddings.pt ]]; then
  echo "Full caches are missing. Run prepare_vps_data.sh first." >&2
  exit 1
fi

python split_train_dev.py

mkdir -p logs runs_clean
python train.py --config config.clean.3090.json \
  2>&1 | tee "logs/train_clean_$(date +%Y%m%d_%H%M%S).log"

RUN_DIR="$(ls -td runs_clean/* | head -1)"
python calibrate_threshold.py \
  --checkpoint "$RUN_DIR/best.pt" \
  --annotations ../data/scanqa/splits/seed42/ScanQA_v1.0_train_dev.json \
  --scene-cache cache/scene_objects.pt \
  --text-cache cache/text_embeddings.pt \
  --output-dir "$RUN_DIR/threshold_calibration" \
  --target-recall-all 0.75 \
  --max-average-objects 10 \
  --fallback-k 10 \
  --max-objects 10 \
  --device cuda

python -c 'import json,sys; policy=json.load(open(sys.argv[1])); assert policy["constraints_met"], "Dev calibration constraints were not met; official val was not evaluated"' \
  "$RUN_DIR/threshold_calibration/selected_policy.json"
THRESHOLD="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_threshold"])' \
  "$RUN_DIR/threshold_calibration/selected_policy.json")"
python evaluate_threshold.py \
  --checkpoint "$RUN_DIR/best.pt" \
  --annotations ../data/scanqa/ScanQA_v1.0_val.json \
  --scene-cache cache/scene_objects.pt \
  --text-cache cache/text_embeddings.pt \
  --output-dir "$RUN_DIR/final_val" \
  --threshold "$THRESHOLD" \
  --threshold-source scene_disjoint_scanqa_train_dev \
  --fallback-k 10 \
  --max-objects 10 \
  --device cuda

echo "Clean experiment completed: $RUN_DIR"
