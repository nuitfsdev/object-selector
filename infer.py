"""Evaluate a checkpoint and export selected ScanNet object IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ScanQAObjectDataset, collate_samples
from metrics import average_metric_rows, batch_metric_rows
from model import ObjectSelector


HERE = Path(__file__).resolve().parent


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else (HERE / path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output", type=Path, default=HERE / "predictions_val.jsonl")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    dataset = ScanQAObjectDataset(
        resolve(data[f"{args.split}_annotations"]), resolve(data["scene_cache"]),
        resolve(data["text_cache"]),
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                        shuffle=False, num_workers=0, collate_fn=collate_samples)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ObjectSelector(
        checkpoint["text_dim"], checkpoint["geometry_dim"],
        checkpoint["appearance_dim"], checkpoint["context_dim"],
        checkpoint["point_dim"], **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    predictions, metric_rows = [], []
    with torch.inference_mode():
        for batch in loader:
            question = batch["question_features"].to(device)
            labels = batch["label_features"].to(device)
            geometry = batch["geometry"].to(device)
            appearance = batch["appearance"].to(device)
            context = batch["context"].to(device)
            object_points = batch["object_points"].to(device)
            centers = batch["centers"].to(device)
            sizes = batch["sizes"].to(device)
            valid = batch["valid_mask"].to(device)
            relevance, count_logits = model(
                question, labels, geometry, appearance, context,
                object_points, centers, sizes, valid)
            scores = relevance.sigmoid().cpu()
            metric_rows.extend(batch_metric_rows(
                scores, batch["targets"], batch["valid_mask"], count_logits.cpu()))
            counts = count_logits.argmax(-1).cpu() + 1
            for index in range(len(scores)):
                valid_count = int(batch["valid_mask"][index].sum())
                selected_count = min(int(counts[index]), valid_count)
                selected_indices = scores[index, :valid_count].topk(selected_count).indices.tolist()
                predictions.append({
                    "question_id": batch["question_ids"][index],
                    "scene_id": batch["scene_ids"][index],
                    "predicted_count": selected_count,
                    "selected_object_ids": [int(batch["object_ids"][index, j])
                                            for j in selected_indices],
                    "scores": [float(scores[index, j]) for j in selected_indices],
                    "labels": [batch["labels"][index][j] for j in selected_indices],
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(average_metric_rows(metric_rows), indent=2))
    print("Saved", len(predictions), "predictions to", args.output)


if __name__ == "__main__":
    main()

