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

mkdir -p logs runs_clean_top10 runs_clean_flexible

echo "[1/6] Training the model optimized for fixed Top-10 RecallAll..."
python train.py --config config.clean.top10.3090.json \
  2>&1 | tee "logs/train_clean_top10_$(date +%Y%m%d_%H%M%S).log"
TOP10_RUN_DIR="$(ls -td runs_clean_top10/* | head -1)"

echo "[2/6] Training the model optimized for flexible set selection..."
python train.py --config config.clean.flexible.3090.json \
  2>&1 | tee "logs/train_clean_flexible_$(date +%Y%m%d_%H%M%S).log"
FLEX_RUN_DIR="$(ls -td runs_clean_flexible/* | head -1)"

echo "[3/6] Calibrating the flexible model on the scene-disjoint dev split..."
python calibrate_threshold.py \
  --checkpoint "$FLEX_RUN_DIR/best.pt" \
  --annotations ../data/scanqa/splits/seed42/ScanQA_v1.0_train_dev.json \
  --scene-cache cache/scene_objects.pt \
  --text-cache cache/text_embeddings.pt \
  --output-dir "$FLEX_RUN_DIR/threshold_calibration" \
  --target-recall-all 0.75 \
  --max-average-objects 10 \
  --f1-weight 0.5 \
  --exact-set-weight 0.3 \
  --count-mae-weight 0.2 \
  --fallback-k 10 \
  --max-objects 10 \
  --device cuda

python -c 'import json,sys; policy=json.load(open(sys.argv[1])); assert policy["constraints_met"], "Dev calibration constraints were not met; official val was not evaluated"' \
  "$FLEX_RUN_DIR/threshold_calibration/selected_policy.json"
THRESHOLD="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_threshold"])' \
  "$FLEX_RUN_DIR/threshold_calibration/selected_policy.json")"

echo "[4/6] Evaluating metric-specific Top-K checkpoints on official val..."
for K in 1 3 5 7 10; do
  TOPK_CHECKPOINT="$TOP10_RUN_DIR/best_recall_all_at_${K}.pt"
  if [[ ! -f "$TOPK_CHECKPOINT" ]]; then
    echo "Missing Top-$K checkpoint: $TOPK_CHECKPOINT" >&2
    exit 1
  fi
  python evaluate_threshold.py \
    --checkpoint "$TOPK_CHECKPOINT" \
    --annotations ../data/scanqa/ScanQA_v1.0_val.json \
    --scene-cache cache/scene_objects.pt \
    --text-cache cache/text_embeddings.pt \
    --output-dir "$TOP10_RUN_DIR/final_val_top${K}" \
    --selection-mode topk \
    --top-k "$K" \
    --device cuda
done

echo "[5/6] Evaluating the flexible model on official val..."
python evaluate_threshold.py \
  --checkpoint "$FLEX_RUN_DIR/best.pt" \
  --annotations ../data/scanqa/ScanQA_v1.0_val.json \
  --scene-cache cache/scene_objects.pt \
  --text-cache cache/text_embeddings.pt \
  --output-dir "$FLEX_RUN_DIR/final_val" \
  --selection-mode threshold \
  --threshold "$THRESHOLD" \
  --threshold-source scene_disjoint_scanqa_train_dev \
  --fallback-k 10 \
  --max-objects 10 \
  --device cuda

echo "[6/6] Clean dual-model experiment completed."
echo "Top-10 model:   $TOP10_RUN_DIR"
echo "Flexible model: $FLEX_RUN_DIR"
