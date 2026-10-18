"""Create a deterministic, scene-disjoint ScanQA train/dev split."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent


def split_quality(dev_scenes, grouped, all_rows, fraction):
    dev_rows = [row for scene in dev_scenes for row in grouped[scene]]

    def stats(rows):
        counts = [len(row["object_ids"]) for row in rows]
        return sum(counts) / len(counts), sum(count > 1 for count in counts) / len(counts)

    all_mean, all_multi = stats(all_rows)
    dev_mean, dev_multi = stats(dev_rows)
    question_fraction = len(dev_rows) / len(all_rows)
    score = (abs(dev_mean - all_mean) / all_mean
             + abs(dev_multi - all_multi)
             + abs(question_fraction - fraction))
    return score, dev_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=WORKSPACE / "data/scanqa/ScanQA_v1.0_train.json")
    parser.add_argument("--output-dir", type=Path,
                        default=WORKSPACE / "data/scanqa/splits/seed42")
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-attempts", type=int, default=2000)
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 1:
        parser.error("--dev-fraction must be between 0 and 1")

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scene_id"]].append(row)
    scenes = sorted(grouped)
    dev_scene_count = max(1, round(len(scenes) * args.dev_fraction))

    best = None
    for attempt in range(args.search_attempts):
        candidate = set(random.Random(args.seed + attempt).sample(scenes, dev_scene_count))
        quality, dev_rows = split_quality(candidate, grouped, rows, args.dev_fraction)
        if best is None or quality < best[0]:
            best = quality, candidate, dev_rows, attempt
    _, dev_scenes, dev_rows, best_attempt = best
    train_rows = [row for row in rows if row["scene_id"] not in dev_scenes]
    train_scenes = set(scenes) - dev_scenes
    assert train_scenes.isdisjoint(dev_scenes)
    assert len(train_rows) + len(dev_rows) == len(rows)

    def describe(partition_rows, partition_scenes):
        counts = [len(row["object_ids"]) for row in partition_rows]
        return {
            "questions": len(partition_rows),
            "scenes": len(partition_scenes),
            "mean_gt_count": sum(counts) / len(counts),
            "multi_object_rate": sum(count > 1 for count in counts) / len(counts),
            "min_gt_count": min(counts),
            "max_gt_count": max(counts),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "ScanQA_v1.0_train_fit.json"
    dev_path = args.output_dir / "ScanQA_v1.0_train_dev.json"
    train_path.write_text(json.dumps(train_rows, ensure_ascii=False), encoding="utf-8")
    dev_path.write_text(json.dumps(dev_rows, ensure_ascii=False), encoding="utf-8")
    summary = {
        "source": str(args.input.resolve()),
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "scene_disjoint": True,
        "stratification_search_attempt": best_attempt,
        "train": describe(train_rows, train_scenes),
        "dev": describe(dev_rows, dev_scenes),
        "dev_scene_ids": sorted(dev_scenes),
    }
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
