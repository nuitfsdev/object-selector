"""Train the ScanQA object selector on cached, frozen text features."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ScanQAObjectDataset, collate_samples
from losses import combined_loss
from metrics import average_metric_rows, batch_metric_rows
from model import ObjectSelector


HERE = Path(__file__).resolve().parent


def resolve(path, base=HERE):
    path = Path(path)
    return path if path.is_absolute() else (base / path).resolve()


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_epoch(loader, model, device, training, optimizer, config, scaler=None):
    model.train(training)
    totals = {"loss": 0.0, "object_loss": 0.0, "count_loss": 0.0,
              "ranking_loss": 0.0, "samples": 0}
    metric_rows = []
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in loader:
            tensors = {
                key: batch[key].to(device)
                for key in ("question_features", "label_features", "geometry", "appearance",
                            "context", "object_points", "centers", "sizes",
                            "targets", "valid_mask")
            }
            if training:
                optimizer.zero_grad(set_to_none=True)
            amp_enabled = device.type == "cuda" and bool(config.get("amp", True))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                relevance_logits, count_logits = model(
                    tensors["question_features"], tensors["label_features"],
                    tensors["geometry"], tensors["appearance"], tensors["context"],
                    tensors["object_points"], tensors["centers"], tensors["sizes"],
                    tensors["valid_mask"],
                )
                loss, components = combined_loss(
                    relevance_logits, count_logits, tensors["targets"], tensors["valid_mask"],
                    count_weight=config["count_loss_weight"],
                    ranking_weight=config["ranking_loss_weight"],
                    ranking_margin=config["ranking_margin"],
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
            count = len(relevance_logits)
            totals["loss"] += float(loss.detach()) * count
            for key, value in components.items():
                totals[key] += float(value) * count
            totals["samples"] += count
            metric_rows.extend(batch_metric_rows(
                relevance_logits.detach().cpu(), tensors["targets"].cpu(),
                tensors["valid_mask"].cpu(), count_logits.detach().cpu(),
            ))
    samples = max(totals.pop("samples"), 1)
    result = {key: value / samples for key, value in totals.items()}
    result.update(average_metric_rows(metric_rows))
    return result


def main(config_path: Path):
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(config["training"]["device"])
    data = config["data"]
    train_set = ScanQAObjectDataset(
        resolve(data["train_annotations"]), resolve(data["scene_cache"]),
        resolve(data["text_cache"]),
    )
    evaluation_name = "dev" if "dev_annotations" in data else "val"
    evaluation_annotations = data.get("dev_annotations", data["val_annotations"])
    evaluation_set = ScanQAObjectDataset(
        resolve(evaluation_annotations), resolve(data["scene_cache"]),
        resolve(data["text_cache"]),
    )
    training = config["training"]
    train_loader = DataLoader(
        train_set, batch_size=training["batch_size"], shuffle=True,
        num_workers=training["num_workers"], collate_fn=collate_samples,
        pin_memory=device.type == "cuda",
    )
    evaluation_loader = DataLoader(
        evaluation_set, batch_size=training["batch_size"], shuffle=False,
        num_workers=training["num_workers"], collate_fn=collate_samples,
        pin_memory=device.type == "cuda",
    )
    sample = train_set[0]
    model = ObjectSelector(
        text_dim=sample["question_feature"].numel(),
        geometry_dim=sample["geometry"].shape[1],
        appearance_dim=sample["appearance"].shape[1],
        context_dim=sample["context"].shape[1],
        point_dim=sample["object_points"].shape[2],
        **config["model"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(
        device.type == "cuda" and bool(training.get("amp", True))))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = resolve(training["output_dir"]) / stamp
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"device={device} train={len(train_set):,} "
          f"{evaluation_name}={len(evaluation_set):,} "
          f"trainable_parameters={parameter_count:,}")

    selection_weights = training.get("selection_metrics", {"dynamic_f1": 1.0})
    if not selection_weights or sum(float(value) for value in selection_weights.values()) <= 0:
        raise ValueError("training.selection_metrics must contain positive weights")

    history, best_score, stale = [], -1.0, 0
    for epoch in range(1, int(training["epochs"]) + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(train_loader, model, device, True, optimizer, training, scaler)
        evaluation_metrics = run_epoch(
            evaluation_loader, model, device, False, None, training, scaler)
        row = {
            "epoch": epoch, "seconds": time.perf_counter() - started,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"{evaluation_name}_{key}": value
               for key, value in evaluation_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row))
        missing_metrics = set(selection_weights) - set(evaluation_metrics)
        if missing_metrics:
            raise KeyError(f"Unknown selection metrics: {sorted(missing_metrics)}")
        weight_sum = sum(float(value) for value in selection_weights.values())
        selection_score = sum(
            evaluation_metrics[key] * float(weight)
            for key, weight in selection_weights.items()
        ) / weight_sum
        checkpoint = {
            "model_state": model.state_dict(), "model_config": config["model"],
            "text_dim": sample["question_feature"].numel(),
            "geometry_dim": sample["geometry"].shape[1], "epoch": epoch,
            "appearance_dim": sample["appearance"].shape[1],
            "context_dim": sample["context"].shape[1],
            "point_dim": sample["object_points"].shape[2],
            "evaluation_split": evaluation_name,
            f"{evaluation_name}_metrics": evaluation_metrics,
            "selection_metrics": selection_weights,
            "selection_score": selection_score,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if selection_score > best_score:
            best_score, stale = selection_score, 0
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            stale += 1
            if stale >= int(training["patience"]):
                print("Early stopping")
                break
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("Saved run to", run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    main(args.config)

