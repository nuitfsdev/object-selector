"""Question-conditioned set Transformer for multi-label object selection."""

from __future__ import annotations

import torch
from torch import nn


class PointObjectEncoder(nn.Module):
    def __init__(self, point_dim, hidden_dim, dropout):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, hidden_dim), nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, points):
        batch, objects, samples, channels = points.shape
        encoded = self.point_mlp(points.reshape(batch * objects, samples, channels))
        pooled = encoded.max(dim=1).values.reshape(batch, objects, -1)
        return self.output_norm(pooled)


class RelativeGeometryLayer(nn.Module):
    """Object self-attention with explicit metric pairwise geometry bias."""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.relation_bias = nn.Sequential(
            nn.Linear(13, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_heads)
        )
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, tokens, centers, sizes, valid_mask):
        batch, objects, hidden = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, objects, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(2)
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # delta[i,j] points from object i to object j in metric scene coordinates.
        delta = centers[:, None, :, :] - centers[:, :, None, :]
        absolute_delta = delta.abs()
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        size_ratio = torch.log((sizes[:, None, :, :] + 1e-4) /
                               (sizes[:, :, None, :] + 1e-4))
        overlap = torch.relu((sizes[:, :, None, :] + sizes[:, None, :, :]) / 2 - absolute_delta)
        overlap = overlap / torch.minimum(sizes[:, :, None, :], sizes[:, None, :, :]).clamp_min(1e-4)
        relation = torch.cat((delta / 5.0, absolute_delta / 5.0, distance / 5.0,
                              size_ratio, overlap), dim=-1)
        bias = self.relation_bias(relation).permute(0, 3, 1, 2)
        logits = logits + bias
        pair_valid = valid_mask[:, None, :, None] & valid_mask[:, None, None, :]
        logits = logits.masked_fill(~pair_valid, -1e4)
        attention = self.dropout(logits.softmax(dim=-1))
        message = torch.matmul(attention, value).transpose(1, 2).reshape(batch, objects, hidden)
        tokens = self.norm1(tokens + self.dropout(self.output(message)))
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens.masked_fill(~valid_mask[..., None], 0)


class ObjectSelector(nn.Module):
    def __init__(self, text_dim: int, geometry_dim: int, appearance_dim: int,
                 context_dim: int, point_dim: int = 6, hidden_dim: int = 256,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1,
                 max_count: int = 25):
        super().__init__()
        self.max_count = int(max_count)
        self.question_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.label_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(geometry_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.appearance_projection = nn.Sequential(
            nn.Linear(appearance_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.context_projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.point_encoder = PointObjectEncoder(point_dim, hidden_dim, dropout)
        self.object_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.relative_geometry = RelativeGeometryLayer(hidden_dim, num_heads, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                 norm=nn.LayerNorm(hidden_dim))
        self.relevance_head = nn.Linear(hidden_dim, 1)
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.max_count),
        )

    def forward(self, question_features, label_features, geometry, appearance, context,
                object_points, centers, sizes, valid_mask):
        question = self.question_projection(question_features)
        label = self.label_projection(label_features)
        geometry = self.geometry_projection(geometry)
        appearance = self.appearance_projection(appearance)
        context = self.context_projection(context)
        points = self.point_encoder(object_points)
        expanded_question = question[:, None, :].expand_as(label)
        object_tokens = self.object_fusion(torch.cat((
            label,
            geometry,
            appearance,
            context,
            points,
            expanded_question * label,
            torch.abs(expanded_question - label),
        ), dim=-1))
        object_tokens = self.relative_geometry(object_tokens, centers, sizes, valid_mask)
        tokens = torch.cat((question[:, None, :], object_tokens), dim=1)
        padding_mask = torch.cat((
            torch.zeros(len(valid_mask), 1, dtype=torch.bool, device=valid_mask.device),
            ~valid_mask,
        ), dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        relevance_logits = self.relevance_head(encoded[:, 1:]).squeeze(-1)
        relevance_logits = relevance_logits.masked_fill(~valid_mask, -1e4)
        count_logits = self.count_head(encoded[:, 0])
        return relevance_logits, count_logits

