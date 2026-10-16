"""Losses robust to many negative objects per ScanQA scene."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def balanced_object_loss(logits, targets, valid_mask):
    positive = targets & valid_mask
    negative = ~targets & valid_mask
    positive_loss = (F.softplus(-logits) * positive).sum(1) / positive.sum(1).clamp_min(1)
    negative_loss = (F.softplus(logits) * negative).sum(1) / negative.sum(1).clamp_min(1)
    return ((positive_loss + negative_loss) / 2).mean()


def pairwise_ranking_loss(logits, targets, valid_mask, margin=0.5):
    positive = targets & valid_mask
    negative = ~targets & valid_mask
    positive_min = logits.masked_fill(~positive, torch.inf).min(1).values
    negative_max = logits.masked_fill(~negative, -torch.inf).max(1).values
    usable = positive.any(1) & negative.any(1)
    if not usable.any():
        return logits.sum() * 0
    return F.relu(float(margin) - positive_min[usable] + negative_max[usable]).mean()


def combined_loss(relevance_logits, count_logits, targets, valid_mask,
                  count_weight=0.25, ranking_weight=0.1, ranking_margin=0.5):
    object_loss = balanced_object_loss(relevance_logits, targets, valid_mask)
    max_count = count_logits.shape[1]
    count_targets = targets.sum(1).clamp(1, max_count).long() - 1
    count_loss = F.cross_entropy(count_logits, count_targets)
    ranking_loss = pairwise_ranking_loss(relevance_logits, targets, valid_mask, ranking_margin)
    total = object_loss + float(count_weight) * count_loss + float(ranking_weight) * ranking_loss
    return total, {"object_loss": object_loss.detach(), "count_loss": count_loss.detach(),
                   "ranking_loss": ranking_loss.detach()}
