#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_vps_from_drive.sh DRIVE_FILE_URL [EXPECTED_SHA256]

The Drive file must be shared as "Anyone with the link" and contain the
"scans/" directory produced by download_scannet_local.ps1 or tar -cf.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

DRIVE_URL="$1"
EXPECTED_SHA256="${2:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname -- "$SCRIPT_DIR")"
DOWNLOAD_DIR="$SCRIPT_DIR/.downloads"
ARCHIVE_PATH="$DOWNLOAD_DIR/scannet_scanqa_633_scenes.tar"
TEMP_ARCHIVE="$ARCHIVE_PATH.part"
SCANNET_ROOT="$WORKSPACE_DIR/data/scannet"

cd "$SCRIPT_DIR"
if [[ ! -f .venv/bin/activate ]]; then
  bash vps_setup.sh
fi
source .venv/bin/activate

mkdir -p "$DOWNLOAD_DIR" "$SCANNET_ROOT"
if [[ ! -s "$ARCHIVE_PATH" ]]; then
  echo "[1/4] Downloading ScanNet TAR from Google Drive..."
  rm -f "$TEMP_ARCHIVE"
  python -m gdown --fuzzy "$DRIVE_URL" --output "$TEMP_ARCHIVE"
  mv "$TEMP_ARCHIVE" "$ARCHIVE_PATH"
else
  echo "[1/4] Reusing existing archive: $ARCHIVE_PATH"
fi

if [[ -n "$EXPECTED_SHA256" ]]; then
  echo "[2/4] Checking SHA256..."
  ACTUAL_SHA256="$(sha256sum "$ARCHIVE_PATH" | awk '{print $1}')"
  if [[ "${ACTUAL_SHA256,,}" != "${EXPECTED_SHA256,,}" ]]; then
    echo "SHA256 mismatch." >&2
    echo "Expected: $EXPECTED_SHA256" >&2
    echo "Actual:   $ACTUAL_SHA256" >&2
    exit 1
  fi
  echo "SHA256 OK: $ACTUAL_SHA256"
else
  echo "[2/4] SHA256 skipped (no expected hash was supplied)."
fi

echo "[3/4] Extracting ScanNet TAR..."
tar -xf "$ARCHIVE_PATH" -C "$SCANNET_ROOT"

python - "$SCANNET_ROOT/scans" <<'PY'
import sys
from pathlib import Path

scan_root = Path(sys.argv[1])
suffixes = (
    "_vh_clean_2.ply",
    "_vh_clean_2.0.010000.segs.json",
    ".aggregation.json",
)
scene_dirs = sorted(path for path in scan_root.glob("scene*") if path.is_dir())
complete = []
missing = []
for scene_dir in scene_dirs:
    scene_id = scene_dir.name
    absent = [suffix for suffix in suffixes
              if not (scene_dir / f"{scene_id}{suffix}").is_file()]
    if absent:
        missing.append((scene_id, absent))
    else:
        complete.append(scene_id)

if len(complete) != 633 or missing:
    examples = ", ".join(scene for scene, _ in missing[:10])
    raise SystemExit(
        f"Incomplete archive: complete={len(complete)}/633, "
        f"incomplete_dirs={len(missing)}, examples={examples}"
    )
print("ARCHIVE OK: 633 scenes and 1899 required files")
PY

echo "[4/4] Starting preprocessing, clean training, calibration, and final evaluation..."
bash run_vps_all.sh --skip-download
