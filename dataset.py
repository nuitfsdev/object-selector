"""ScanQA dataset with object IDs retained only as output metadata."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ScanQAObjectDataset(Dataset):
    def __init__(self, annotation_path: Path, scene_cache_path: Path, text_cache_path: Path):
        scene_data = torch.load(scene_cache_path, map_location="cpu", weights_only=False)
        text_data = torch.load(text_cache_path, map_location="cpu", weights_only=False)
        self.scenes = scene_data["scenes"]
        self.question_embeddings = {
            str(qid): embedding.float()
            for qid, embedding in zip(text_data["question_ids"], text_data["question_embeddings"])
        }
        self.label_embeddings = {
            label: embedding.float()
            for label, embedding in zip(text_data["labels"], text_data["label_embeddings"])
        }
        rows = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
        self.rows = [
            row for row in rows
            if row["scene_id"] in self.scenes and str(row["question_id"]) in self.question_embeddings
        ]
        if not self.rows:
            raise ValueError(f"No usable samples from {annotation_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        question_id = str(row["question_id"])
        scene = self.scenes[row["scene_id"]]
        object_ids = [int(value) for value in scene["object_ids"]]
        labels = list(scene["labels"])
        label_features = torch.stack([self.label_embeddings[label] for label in labels])
        gt_ids = {int(value) for value in row["object_ids"]}
        targets = torch.tensor([object_id in gt_ids for object_id in object_ids], dtype=torch.bool)
        if not targets.any():
            raise ValueError(f"{question_id}: no GT object is present in scene candidates")
        return {
            "question_id": question_id,
            "scene_id": str(row["scene_id"]),
            "question": str(row["question"]),
            "question_feature": self.question_embeddings[question_id],
            "label_features": label_features,
            "geometry": scene["geometry"].float(),
            "appearance": scene["appearance"].float(),
            "context": scene["context"].float(),
            "object_points": scene["object_points"].float(),
            "centers": scene["centers"].float(),
            "sizes": scene["sizes"].float(),
            "targets": targets,
            "object_ids": object_ids,  # metadata: never passed to the model
            "labels": labels,
        }


def collate_samples(samples):
    batch_size = len(samples)
    max_objects = max(len(sample["object_ids"]) for sample in samples)
    text_dim = samples[0]["question_feature"].numel()
    geometry_dim = samples[0]["geometry"].shape[1]
    appearance_dim = samples[0]["appearance"].shape[1]
    context_dim = samples[0]["context"].shape[1]
    num_points, point_dim = samples[0]["object_points"].shape[1:]
    question_features = torch.stack([sample["question_feature"] for sample in samples])
    label_features = torch.zeros(batch_size, max_objects, text_dim)
    geometry = torch.zeros(batch_size, max_objects, geometry_dim)
    appearance = torch.zeros(batch_size, max_objects, appearance_dim)
    context = torch.zeros(batch_size, max_objects, context_dim)
    object_points = torch.zeros(batch_size, max_objects, num_points, point_dim)
    centers = torch.zeros(batch_size, max_objects, 3)
    sizes = torch.zeros(batch_size, max_objects, 3)
    targets = torch.zeros(batch_size, max_objects, dtype=torch.bool)
    valid_mask = torch.zeros(batch_size, max_objects, dtype=torch.bool)
    object_ids = torch.full((batch_size, max_objects), -1, dtype=torch.long)

    for batch_index, sample in enumerate(samples):
        count = len(sample["object_ids"])
        label_features[batch_index, :count] = sample["label_features"]
        geometry[batch_index, :count] = sample["geometry"]
        appearance[batch_index, :count] = sample["appearance"]
        context[batch_index, :count] = sample["context"]
        object_points[batch_index, :count] = sample["object_points"]
        centers[batch_index, :count] = sample["centers"]
        sizes[batch_index, :count] = sample["sizes"]
        targets[batch_index, :count] = sample["targets"]
        valid_mask[batch_index, :count] = True
        object_ids[batch_index, :count] = torch.tensor(sample["object_ids"], dtype=torch.long)
    return {
        "question_features": question_features,
        "label_features": label_features,
        "geometry": geometry,
        "appearance": appearance,
        "context": context,
        "object_points": object_points,
        "centers": centers,
        "sizes": sizes,
        "targets": targets,
        "valid_mask": valid_mask,
        "object_ids": object_ids,  # metadata: training forward does not consume it
        "question_ids": [sample["question_id"] for sample in samples],
        "scene_ids": [sample["scene_id"] for sample in samples],
        "questions": [sample["question"] for sample in samples],
        "labels": [sample["labels"] for sample in samples],
    }

