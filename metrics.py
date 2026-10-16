"""Retrieval metrics for fixed K and predicted object count."""

from __future__ import annotations

import math

import torch


def _sample_metrics(order, relevant, selected_count):
    relevant = set(relevant)
    selected = order[:selected_count]
    hits = len(relevant.intersection(selected))
    precision = hits / max(len(selected), 1)
    recall = hits / len(relevant)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, float(hits > 0), float(hits == len(relevant))


def batch_metric_rows(scores, targets, valid_mask, count_logits, ks=(1, 3, 5, 7, 10)):
    rows = []
    predicted_counts = count_logits.argmax(-1) + 1
    for index in range(len(scores)):
        valid_count = int(valid_mask[index].sum())
        order = scores[index, :valid_count].argsort(descending=True).tolist()
        relevant = targets[index, :valid_count].nonzero(as_tuple=False).flatten().tolist()
        row = {}
        for k in ks:
            p, r, f1, hit, all_hit = _sample_metrics(order, relevant, min(k, valid_count))
            row.update({f"precision@{k}": p, f"recall@{k}": r, f"f1@{k}": f1,
                        f"hit@{k}": hit, f"recall_all@{k}": all_hit})
        dynamic_count = min(int(predicted_counts[index]), valid_count)
        p, r, f1, hit, all_hit = _sample_metrics(order, relevant, dynamic_count)
        row.update({"dynamic_precision": p, "dynamic_recall": r, "dynamic_f1": f1,
                    "dynamic_hit": hit, "dynamic_recall_all": all_hit,
                    "predicted_count": dynamic_count,
                    "true_count": len(relevant)})
        rows.append(row)
    return rows


def average_metric_rows(rows):
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}

