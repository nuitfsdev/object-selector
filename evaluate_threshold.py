"""Evaluate a fixed score threshold without tuning it on the evaluation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ScanQAObjectDataset, collate_samples
from model import ObjectSelector


def average(rows, key):
    return sum(row[key] for row in rows) / max(len(rows), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--scene-cache", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fallback-k", type=int, default=10)
    parser.add_argument("--max-objects", type=int, default=0,
                        help="0 means no cap; otherwise retain only the highest scores")
    parser.add_argument("--threshold-source", default="predeclared_not_tuned_on_eval")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if args.fallback_k < 1:
        parser.error("--fallback-k must be positive")
    if args.max_objects < 0:
        parser.error("--max-objects cannot be negative")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    dataset = ScanQAObjectDataset(args.annotations, args.scene_cache, args.text_cache)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_samples)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ObjectSelector(
        checkpoint["text_dim"], checkpoint["geometry_dim"],
        checkpoint["appearance_dim"], checkpoint["context_dim"],
        checkpoint["point_dim"], **checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    predictions = []
    with torch.inference_mode():
        for batch in loader:
            inputs = [batch[key].to(device) for key in (
                "question_features", "label_features", "geometry", "appearance",
                "context", "object_points", "centers", "sizes", "valid_mask",
            )]
            relevance_logits, _ = model(*inputs)
            scores = relevance_logits.sigmoid().cpu()
            for index in range(len(scores)):
                valid_count = int(batch["valid_mask"][index].sum())
                valid_scores = scores[index, :valid_count]
                selected = torch.where(valid_scores >= args.threshold)[0]
                used_fallback = len(selected) == 0
                if used_fallback:
                    count = min(args.fallback_k, valid_count)
                    selected = valid_scores.topk(count).indices
                else:
                    selected = selected[valid_scores[selected].argsort(descending=True)]
                was_capped = args.max_objects > 0 and len(selected) > args.max_objects
                if was_capped:
                    selected = selected[:args.max_objects]

                targets = batch["targets"][index, :valid_count]
                relevant_count = int(targets.sum())
                hits = int(targets[selected].sum())
                precision = hits / len(selected)
                recall = hits / relevant_count
                f1 = (2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
                selected_indices = selected.tolist()
                predictions.append({
                    "question_id": batch["question_ids"][index],
                    "scene_id": batch["scene_ids"][index],
                    "threshold": args.threshold,
                    "used_fallback_topk": used_fallback,
                    "was_capped": was_capped,
                    "selected_count": len(selected_indices),
                    "relevant_count": relevant_count,
                    "hits": hits,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "hit": float(hits > 0),
                    "recall_all": float(hits == relevant_count),
                    "selected_object_ids": [int(batch["object_ids"][index, j])
                                            for j in selected_indices],
                    "scores": [float(valid_scores[j]) for j in selected_indices],
                    "labels": [batch["labels"][index][j] for j in selected_indices],
                })

    counts = np.asarray([row["selected_count"] for row in predictions])
    fallback_count = sum(row["used_fallback_topk"] for row in predictions)
    capped_count = sum(row["was_capped"] for row in predictions)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "annotations": str(args.annotations.resolve()),
        "num_questions": len(predictions),
        "selection_policy": "score >= threshold; if empty, fallback Top-K; optional cap",
        "threshold": args.threshold,
        "threshold_source": args.threshold_source,
        "fallback_k": args.fallback_k,
        "max_objects": args.max_objects,
        "fallback_questions": fallback_count,
        "fallback_rate": fallback_count / max(len(predictions), 1),
        "capped_questions": capped_count,
        "capped_rate": capped_count / max(len(predictions), 1),
        "precision": average(predictions, "precision"),
        "recall": average(predictions, "recall"),
        "f1": average(predictions, "f1"),
        "hit": average(predictions, "hit"),
        "recall_all": average(predictions, "recall_all"),
        "selected_count_mean": float(counts.mean()),
        "selected_count_median": float(np.median(counts)),
        "selected_count_p90": float(np.quantile(counts, 0.9)),
        "selected_count_min": int(counts.min()),
        "selected_count_max": int(counts.max()),
        "true_count_mean": average(predictions, "relevant_count"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
