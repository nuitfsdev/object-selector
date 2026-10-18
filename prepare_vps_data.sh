#!/usr/bin/env bash
set -euo pipefail

# Prepare every input needed by train.py on a Linux VPS.
# ScanNet is license-gated, so downloading requires the caller to explicitly pass
# --accept-scannet-tos after accepting https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf

ACCEPT_TOS=0
SKIP_DOWNLOAD=0
WORKERS="${DOWNLOAD_WORKERS:-8}"

usage() {
  cat <<'EOF'
Usage:
  bash prepare_vps_data.sh --accept-scannet-tos
  bash prepare_vps_data.sh --skip-download

Options:
  --accept-scannet-tos  Confirm that you accepted the ScanNet terms and download missing files.
  --skip-download       Use ScanNet files already present under ../data/scannet/scans.
  --workers N           Parallel ScanNet downloads (default: 8).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accept-scannet-tos)
      ACCEPT_TOS=1
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --workers)
      WORKERS="${2:?--workers needs an integer}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Run: bash vps_setup.sh" >&2
  exit 1
fi
source .venv/bin/activate

TRAIN_JSON="../data/scanqa/ScanQA_v1.0_train.json"
VAL_JSON="../data/scanqa/ScanQA_v1.0_val.json"
SCANQA_DRIVE_URL="https://drive.google.com/drive/folders/1-21A3TBE0QuofEwDg5oDz2z0HEdbVgL2?usp=sharing"

if [[ ! -f "$TRAIN_JSON" || ! -f "$VAL_JSON" ]]; then
  echo "ScanQA annotations are missing; downloading the official folder..."
  mkdir -p ../data/scanqa
  python -m gdown --folder "$SCANQA_DRIVE_URL" --output ../data/scanqa

  # gdown may retain a folder level. Normalize the two files to the paths used by config.
  for name in ScanQA_v1.0_train.json ScanQA_v1.0_val.json; do
    if [[ ! -f "../data/scanqa/$name" ]]; then
      found="$(find ../data/scanqa -type f -name "$name" -print -quit)"
      if [[ -n "$found" ]]; then
        cp "$found" "../data/scanqa/$name"
      fi
    fi
  done
fi

for annotation in "$TRAIN_JSON" "$VAL_JSON"; do
  if [[ ! -s "$annotation" ]]; then
    echo "Could not obtain annotation: $annotation" >&2
    echo "Official source: $SCANQA_DRIVE_URL" >&2
    exit 1
  fi
done

mkdir -p cache logs ../data/scannet/scans

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  if [[ "$ACCEPT_TOS" -ne 1 ]]; then
    echo "ScanNet download requires acceptance of its terms:" >&2
    echo "https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf" >&2
    echo "After accepting, rerun with --accept-scannet-tos." >&2
    exit 2
  fi
  echo "[1/4] Downloading only the 3 required ScanNet files for train + val..."
  python download_scanqa_scannet.py \
    --split both \
    --workers "$WORKERS" \
    --accept-scannet-tos \
    2>&1 | tee logs/download_scannet.log
else
  echo "[1/4] Reusing existing ScanNet files."
fi

echo "[2/4] Extracting geometry, color, shape, context, and local point features..."
rm -f cache/scene_objects.pt
python prepare_objects.py 2>&1 | tee logs/prepare_objects.log

echo "[3/4] Encoding ScanQA questions and object labels..."
rm -f cache/text_embeddings.pt
python encode_text.py --device auto --batch-size 512 2>&1 | tee logs/encode_text.log

echo "[4/4] Verifying cache coverage and tensor integrity..."
python - <<'PY'
import json
from pathlib import Path

import torch

root = Path.cwd()
annotation_paths = [
    root.parent / "data/scanqa/ScanQA_v1.0_train.json",
    root.parent / "data/scanqa/ScanQA_v1.0_val.json",
]
rows = [row for path in annotation_paths for row in json.loads(path.read_text(encoding="utf-8"))]
expected_scenes = {row["scene_id"] for row in rows}
expected_questions = {str(row["question_id"]) for row in rows}

scene_cache = torch.load(root / "cache/scene_objects.pt", map_location="cpu", weights_only=False)
text_cache = torch.load(root / "cache/text_embeddings.pt", map_location="cpu", weights_only=False)
actual_scenes = set(scene_cache["scenes"])
actual_questions = set(text_cache["question_ids"])

if scene_cache.get("format_version") != 2:
    raise SystemExit("Unexpected scene cache format")
if actual_scenes != expected_scenes:
    raise SystemExit(f"Scene cache mismatch: expected={len(expected_scenes)}, actual={len(actual_scenes)}")
if actual_questions != expected_questions:
    raise SystemExit(
        f"Text cache mismatch: expected={len(expected_questions)}, actual={len(actual_questions)}"
    )
for scene_id, scene in scene_cache["scenes"].items():
    object_count = len(scene["object_ids"])
    for key in ("geometry", "appearance", "context", "object_points", "centers", "sizes"):
        if len(scene[key]) != object_count:
            raise SystemExit(f"Broken tensor length: {scene_id}/{key}")

size_mib = sum(path.stat().st_size for path in (root / "cache").glob("*.pt")) / 2**20
print(f"READY: {len(actual_scenes)} scenes, {len(actual_questions)} questions, cache={size_mib:.1f} MiB")
PY

echo "Data preparation completed. Start training with: bash run_vps_training.sh"
