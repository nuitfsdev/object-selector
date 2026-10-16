"""Precompute frozen language embeddings for questions and object labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)


def encode(texts, tokenizer, model, device, batch_size):
    outputs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
        batch = texts[start:start + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=128,
                           return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            pooled = F.normalize(mean_pool(hidden, tokens["attention_mask"]), dim=-1)
        outputs.append(pooled.cpu().to(torch.float16))
    return torch.cat(outputs) if outputs else torch.empty(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scene-cache", type=Path, default=HERE / "cache/scene_objects.pt")
    parser.add_argument("--output", type=Path, default=HERE / "cache/text_embeddings.pt")
    parser.add_argument(
        "--annotations", nargs="+", type=Path,
        default=[WORKSPACE / "data/scanqa/ScanQA_v1.0_train.json",
                 WORKSPACE / "data/scanqa/ScanQA_v1.0_val.json"],
    )
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)

    scene_cache = torch.load(args.scene_cache, map_location="cpu", weights_only=False)["scenes"]
    labels = sorted({label for scene in scene_cache.values() for label in scene["labels"]})
    question_rows = []
    available_scenes = set(scene_cache)
    for path in args.annotations:
        for row in json.loads(path.read_text(encoding="utf-8")):
            if row["scene_id"] in available_scenes:
                question_rows.append((str(row["question_id"]), str(row["question"])))
    if len({qid for qid, _ in question_rows}) != len(question_rows):
        raise ValueError("Question IDs are not unique")

    print(f"Loading frozen text encoder {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    question_matrix = encode([text for _, text in question_rows], tokenizer, model,
                             device, args.batch_size)
    label_matrix = encode(labels, tokenizer, model, device, args.batch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "text_model": args.model,
        "embedding_dim": int(question_matrix.shape[1]),
        "question_ids": [qid for qid, _ in question_rows],
        "question_embeddings": question_matrix,
        "labels": labels,
        "label_embeddings": label_matrix,
    }, args.output)
    print(f"Saved {len(question_rows)} questions and {len(labels)} labels to {args.output}")


if __name__ == "__main__":
    main()
