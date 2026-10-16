# ScanQA object selector

Lightweight multi-label object selector trained on ScanQA. It predicts a relevance
score for every ScanNet object and a dynamic number of objects to return.

## What enters the model

- frozen MiniLM embedding of the question;
- frozen MiniLM embedding of each object label;
- 31 geometry/shape/density features;
- 79 RGB/HSV appearance features;
- 17 scene-context features;
- 128 sampled local XYZ+RGB points per object;
- metric centers/sizes for explicit pairwise-relation attention.

`object_id` is retained in the dataset and exported with predictions, but it is
never passed to `ObjectSelector.forward()`. Ground-truth IDs are used only to
construct the multi-label training target.

```text
question embedding ───────────────┐
                                  ├─ 2-layer set Transformer ─ relevance/object
label + geometry + RGB + points ──┘                         └─ predicted count
```

The object branch includes a small PointNet and a relative-geometry attention
layer. Frozen text embeddings keep the overall training much smaller than a
3D-LLM. AMP is enabled automatically on CUDA.

Feature groups extracted from the three ScanNet files:

- **Semantic:** object class label and class frequency in its scene.
- **Color/material cues:** RGB mean/std/median/quartiles, circular hue statistics,
  saturation/value statistics, and RGB/HSV histograms.
- **Geometry:** normalized center, size, min/max corners, volume, surface area,
  height above floor, scene-center distance, and aspect ratios.
- **Shape:** covariance eigenvalue descriptors (linearity, planarity, scattering,
  anisotropy, omnivariance, verticality) plus PointNet over local XYZ+RGB.
- **Context:** nearest-neighbor distances, six directional object counts,
  neighborhood counts at 0.5/1/2 m, maximum/mean box IoU, same-class frequency
  and distance, and wall distance.
- **Relations:** learned attention bias from signed pairwise displacement,
  distance, size ratios, and axis overlap.

## Data required

The ScanQA annotation files already exist under `../data/scanqa`. For every
train/validation scene, only these ScanNet files are required:

```text
<scene>_vh_clean_2.ply
<scene>_vh_clean_2.0.010000.segs.json
<scene>.aggregation.json
```

They require roughly 4–4.5 GiB for the 562 ScanQA training scenes. Raw `.sens`,
RGB/depth frames, camera poses, and PoseRecover visibility files are not needed.

If the files are missing, first read and accept the ScanNet terms, then run:

```powershell
cd scanqa_object_selector
python download_scanqa_scannet.py --split train --accept-scannet-tos
```

The downloader reuses `../download-scannet.py` and skips existing files.

## Setup and preprocessing

```powershell
cd scanqa_object_selector
python -m pip install -r requirements.txt

# Build bbox/label features. This fails clearly if any scene is missing.
python prepare_objects.py

# Download MiniLM once and cache all question/label embeddings.
python encode_text.py
```

For a preprocessing smoke test using only scenes already present:

```powershell
python prepare_objects.py --allow-missing --output cache/scene_objects_smoke.pt
python encode_text.py --scene-cache cache/scene_objects_smoke.pt --output cache/text_embeddings_smoke.pt
```

## Training

Review `config.json`, then run:

```powershell
python train.py --config config.json
```

Checkpoints and `history.json` are written under `runs/<timestamp>/`. The best
checkpoint is selected using validation dynamic-F1, where the count head decides
how many objects to return.

Loss:

```text
balanced object BCE + 0.25 × count CE + 0.10 × pairwise ranking loss
```

Reported validation metrics include Precision/Recall/F1, Hit, and Recall-All at
K = 1, 3, 5, 7, 10, plus metrics for the predicted dynamic count.

## Evaluation and exported IDs

```powershell
python infer.py `
  --checkpoint runs/<timestamp>/best.pt `
  --split val `
  --output predictions_val.jsonl
```

Each output line preserves the original ScanNet IDs:

```json
{
  "question_id": "val-scene0011-2",
  "scene_id": "scene0011_00",
  "predicted_count": 2,
  "selected_object_ids": [9, 2],
  "scores": [0.94, 0.82],
  "labels": ["chair", "table"]
}
```

## SQA3D transfer

For zero-shot SQA3D, encode `situation + " [SEP] " + question` using the same
MiniLM model, reuse scene object features, and map selected tensor positions back
to `object_ids`. Add the SQA3D pose-based left/right/front/behind reranker after
this semantic selector; do not feed numeric IDs into either model.

## Evaluation cautions

- Tune thresholds or count-selection choices on a training-held-out development
  subset, not on the final validation results.
- This project does not use GT IDs to choose a camera pose, avoiding the oracle
  pose leakage in the older CAPruner/PoseAlign experiment.
- The current features use GT ScanNet instances. Detector proposals require a
  separate IoU mapping between proposal IDs and GT object IDs.
- View-dependent texture, occlusion, and image-only details cannot be recovered
  from the three downloaded files. They require raw RGB frames or multiview
  features and are intentionally outside this compact dataset configuration.
