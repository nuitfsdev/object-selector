"""Choose a score threshold on a scene-disjoint development split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ScanQAObjectDataset, collate_samples
from model import ObjectSelector


def select_indices(scores, threshold, fallback_k, max_objects):
    selected = torch.where(scores >= threshold)[0]
    fallback = len(selected) == 0
    if fallback:
        selected = scores.topk(min(fallback_k, len(scores))).indices
    else:
        selected = selected[scores[selected].argsort(descending=True)]
    capped = max_objects > 0 and len(selected) > max_objects
    if capped:
        selected = selected[:max_objects]
    return selected, fallback, capped


def evaluate(records, threshold, fallback_k, max_objects):
    rows = []
    for scores, targets in records:
        selected, fallback, capped = select_indices(
            scores, threshold, fallback_k, max_objects)
        relevant = int(targets.sum())
        hits = int(targets[selected].sum())
        precision = hits / len(selected)
        recall = hits / relevant
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((precision, recall, f1, hits > 0, hits == relevant,
                     len(selected), fallback, capped))
    matrix = np.asarray(rows, dtype=np.float64)
    return {
        "threshold": threshold,
        "precision": matrix[:, 0].mean(),
        "recall": matrix[:, 1].mean(),
        "f1": matrix[:, 2].mean(),
        "hit": matrix[:, 3].mean(),
        "recall_all": matrix[:, 4].mean(),
        "selected_count_mean": matrix[:, 5].mean(),
        "fallback_rate": matrix[:, 6].mean(),
        "capped_rate": matrix[:, 7].mean(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--scene-cache", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--target-recall-all", type=float, default=0.75)
    parser.add_argument("--max-average-objects", type=float, default=10.0)
    parser.add_argument("--fallback-k", type=int, default=10)
    parser.add_argument("--max-objects", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.threshold_step <= 0 or args.threshold_min > args.threshold_max:
        parser.error("Invalid threshold range")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device)
    dataset = ScanQAObjectDataset(args.annotations, args.scene_cache, args.text_cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_samples)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ObjectSelector(
        checkpoint["text_dim"], checkpoint["geometry_dim"],
        checkpoint["appearance_dim"], checkpoint["context_dim"],
        checkpoint["point_dim"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    records = []
    with torch.inference_mode():
        for batch in loader:
            inputs = [batch[key].to(device) for key in (
                "question_features", "label_features", "geometry", "appearance",
                "context", "object_points", "centers", "sizes", "valid_mask")]
            relevance, _ = model(*inputs)
            scores = relevance.sigmoid().cpu()
            for index in range(len(scores)):
                count = int(batch["valid_mask"][index].sum())
                records.append((scores[index, :count], batch["targets"][index, :count]))

    steps = int(round((args.threshold_max - args.threshold_min) / args.threshold_step))
    thresholds = [round(args.threshold_min + i * args.threshold_step, 10)
                  for i in range(steps + 1)]
    curve = [evaluate(records, threshold, args.fallback_k, args.max_objects)
             for threshold in thresholds]
    eligible = [row for row in curve
                if row["recall_all"] >= args.target_recall_all
                and row["selected_count_mean"] <= args.max_average_objects]
    constraints_met = bool(eligible)
    candidates = eligible or curve
    selected = max(candidates, key=lambda row: (
        row["f1"], row["recall_all"], row["precision"], -row["selected_count_mean"]))

    policy = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "calibration_annotations": str(args.annotations.resolve()),
        "num_calibration_questions": len(records),
        "selected_threshold": selected["threshold"],
        "selection_rule": "maximize macro F1 subject to constraints",
        "target_recall_all": args.target_recall_all,
        "max_average_objects": args.max_average_objects,
        "constraints_met": constraints_met,
        "fallback_k": args.fallback_k,
        "max_objects": args.max_objects,
        "dev_metrics": selected,
        "threshold_range": [args.threshold_min, args.threshold_max, args.threshold_step],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected_policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8")
    with (args.output_dir / "threshold_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    print(json.dumps(policy, indent=2))


if __name__ == "__main__":
    main()
