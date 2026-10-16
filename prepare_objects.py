"""Build per-scene object metadata without exposing object IDs to the model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
DEFAULT_SCAN_ROOTS = (
    WORKSPACE / "data/scannet/scans",
    WORKSPACE / "prepare-data/data/scannet/scans",
)
NUM_OBJECT_POINTS = 128
RGB_BINS = 8
HSV_BINS = (12, 8, 8)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB [0,1] to HSV [0,1]."""
    maximum = rgb.max(1)
    minimum = rgb.min(1)
    delta = maximum - minimum
    hue = np.zeros(len(rgb), dtype=np.float32)
    nonzero = delta > 1e-8
    red = nonzero & (maximum == rgb[:, 0])
    green = nonzero & (maximum == rgb[:, 1])
    blue = nonzero & (maximum == rgb[:, 2])
    hue[red] = ((rgb[red, 1] - rgb[red, 2]) / delta[red]) % 6
    hue[green] = (rgb[green, 2] - rgb[green, 0]) / delta[green] + 2
    hue[blue] = (rgb[blue, 0] - rgb[blue, 1]) / delta[blue] + 4
    hue /= 6
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-8)
    return np.column_stack((hue, saturation, maximum)).astype(np.float32)


def normalized_histogram(values: np.ndarray, bins: int) -> np.ndarray:
    histogram, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return (histogram / max(histogram.sum(), 1)).astype(np.float32)


def appearance_features(rgb: np.ndarray) -> np.ndarray:
    hsv = rgb_to_hsv(rgb)
    rgb_stats = np.concatenate((
        rgb.mean(0), rgb.std(0), np.median(rgb, axis=0),
        np.quantile(rgb, 0.25, axis=0), np.quantile(rgb, 0.75, axis=0),
    ))
    hue_angle = hsv[:, 0] * (2 * np.pi)
    hue_circular = np.array([np.sin(hue_angle).mean(), np.cos(hue_angle).mean()])
    sv = hsv[:, 1:]
    hsv_stats = np.concatenate((
        hue_circular, sv.mean(0), sv.std(0), np.median(sv, axis=0),
        np.quantile(sv, 0.25, axis=0), np.quantile(sv, 0.75, axis=0),
    ))
    histograms = np.concatenate((
        *(normalized_histogram(rgb[:, channel], RGB_BINS) for channel in range(3)),
        normalized_histogram(hsv[:, 0], HSV_BINS[0]),
        normalized_histogram(hsv[:, 1], HSV_BINS[1]),
        normalized_histogram(hsv[:, 2], HSV_BINS[2]),
    ))
    return np.concatenate((rgb_stats, hsv_stats, histograms)).astype(np.float32)


def shape_features(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(0)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1e-9)
    principal = eigenvectors[:, order[0]]
    first, second, third = eigenvalues
    linearity = (first - second) / first
    planarity = (second - third) / first
    scattering = third / first
    anisotropy = (first - third) / first
    omnivariance = float(np.cbrt(first * second * third))
    verticality = 1.0 - abs(float(principal[2]))
    return np.asarray((linearity, planarity, scattering, anisotropy,
                       np.log1p(omnivariance), verticality), dtype=np.float32)


def sample_object_points(scene_id: str, object_id: int, points: np.ndarray,
                         rgb: np.ndarray, center: np.ndarray, size: np.ndarray) -> np.ndarray:
    seed_text = f"{scene_id}:{object_id}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "little")
    generator = np.random.default_rng(seed)
    replace = len(points) < NUM_OBJECT_POINTS
    indices = generator.choice(len(points), size=NUM_OBJECT_POINTS, replace=replace)
    local_xyz = (points[indices] - center) / np.maximum(size, 1e-6)
    return np.concatenate((local_xyz, rgb[indices]), axis=1).astype(np.float16)


def build_context(centers: np.ndarray, sizes: np.ndarray, labels: list[str],
                  scene_scale: float) -> np.ndarray:
    count = len(centers)
    delta = centers[None, :, :] - centers[:, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)
    half = sizes / 2
    box_min, box_max = centers - half, centers + half
    intersection_min = np.maximum(box_min[:, None, :], box_min[None, :, :])
    intersection_max = np.minimum(box_max[:, None, :], box_max[None, :, :])
    intersection = np.maximum(intersection_max - intersection_min, 0).prod(-1)
    volume = sizes.prod(-1)
    union = volume[:, None] + volume[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    np.fill_diagonal(iou, 0)
    wall_indices = [index for index, label in enumerate(labels) if label == "wall"]
    features = []
    for index in range(count):
        finite_distances = np.sort(distance[index][np.isfinite(distance[index])])
        nearest = np.pad(finite_distances[:3] / scene_scale, (0, max(0, 3-len(finite_distances))),
                         constant_values=1)[:3]
        directional = np.concatenate(((delta[index] > 0).sum(0), (delta[index] < 0).sum(0)))
        directional = directional / max(count - 1, 1)
        radius_counts = np.asarray([(distance[index] < radius).sum()
                                    for radius in (0.5, 1.0, 2.0)], dtype=np.float32)
        radius_counts /= max(count - 1, 1)
        same_label = np.asarray([labels[j] == labels[index] and j != index for j in range(count)])
        same_count = same_label.sum() / max(count - 1, 1)
        same_distance = (distance[index][same_label].min() / scene_scale
                         if same_label.any() else 1.0)
        wall_distance = (distance[index, wall_indices].min() / scene_scale
                         if wall_indices and index not in wall_indices else 0.0)
        features.append(np.concatenate((
            nearest, directional, radius_counts,
            np.asarray((iou[index].max(), iou[index].mean(), same_count,
                        same_distance, wall_distance), dtype=np.float32),
        )))
    return np.asarray(features, dtype=np.float32)


def annotation_scenes(paths: list[Path]) -> list[str]:
    scenes: set[str] = set()
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        scenes.update(str(row["scene_id"]) for row in rows)
    return sorted(scenes)


def find_scene(scene_id: str, roots: list[Path]) -> Path | None:
    for root in roots:
        candidate = root / scene_id
        required = (
            candidate / f"{scene_id}_vh_clean_2.ply",
            candidate / f"{scene_id}_vh_clean_2.0.010000.segs.json",
            candidate / f"{scene_id}.aggregation.json",
        )
        if all(path.is_file() for path in required):
            return candidate
    return None


def extract_scene(scene_id: str, scene_dir: Path) -> dict:
    ply_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    seg_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    agg_path = scene_dir / f"{scene_id}.aggregation.json"

    vertices = PlyData.read(str(ply_path))["vertex"].data
    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
    vertex_rgb = np.column_stack((vertices["red"], vertices["green"], vertices["blue"])).astype(np.float32) / 255
    segment_ids = np.asarray(json.loads(seg_path.read_text(encoding="utf-8"))["segIndices"])
    groups = json.loads(agg_path.read_text(encoding="utf-8"))["segGroups"]

    scene_min, scene_max = xyz.min(0), xyz.max(0)
    scene_center = (scene_min + scene_max) / 2
    scene_scale = max(float((scene_max - scene_min).max()), 1e-6)

    object_ids, labels, geometry, appearance, object_points = [], [], [], [], []
    metric_centers, metric_sizes = [], []
    for group in sorted(groups, key=lambda item: int(item["objectId"])):
        vertex_mask = np.isin(segment_ids, np.asarray(group["segments"]))
        points = xyz[vertex_mask]
        colors = vertex_rgb[vertex_mask]
        if not len(points):
            continue
        box_min, box_max = points.min(0), points.max(0)
        center = (box_min + box_max) / 2
        size = np.maximum(box_max - box_min, 1e-6)
        box_min_norm = (box_min - scene_center) / scene_scale
        box_max_norm = (box_max - scene_center) / scene_scale
        volume = float(size.prod())
        surface_area = float(2 * (size[0]*size[1] + size[1]*size[2] + size[0]*size[2]))
        aspect = size / max(float(size.max()), 1e-6)
        density = len(points) / max(volume, 1e-6)
        # Rich geometry derived only from points/bboxes. object_id is never included.
        feature = np.concatenate((
            (center - scene_center) / scene_scale,
            size / scene_scale,
            np.log1p(size),
            box_min_norm, box_max_norm,
            np.asarray((np.log1p(volume), np.log1p(surface_area),
                        np.log1p(len(points)), np.log1p(density),
                        (box_min[2] - scene_min[2]) / scene_scale,
                        (center[2] - scene_min[2]) / scene_scale,
                        np.linalg.norm(center - scene_center) / scene_scale), dtype=np.float32),
            aspect,
            shape_features(points),
        )).astype(np.float32)
        object_id = int(group["objectId"])
        object_ids.append(object_id)
        labels.append(str(group["label"]).strip().lower())
        geometry.append(feature)
        appearance.append(appearance_features(colors))
        object_points.append(sample_object_points(scene_id, object_id, points, colors, center, size))
        metric_centers.append(center)
        metric_sizes.append(size)

    if not object_ids:
        raise ValueError(f"{scene_id} contains no valid objects")
    metric_centers = np.stack(metric_centers).astype(np.float32)
    metric_sizes = np.stack(metric_sizes).astype(np.float32)
    context = build_context(metric_centers, metric_sizes, labels, scene_scale)
    return {
        "object_ids": object_ids,  # metadata only
        "labels": labels,
        "geometry": torch.from_numpy(np.stack(geometry)),
        "appearance": torch.from_numpy(np.stack(appearance)),
        "context": torch.from_numpy(context),
        "object_points": torch.from_numpy(np.stack(object_points)),
        "centers": torch.from_numpy(metric_centers),
        "sizes": torch.from_numpy(metric_sizes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", nargs="+", type=Path,
        default=[WORKSPACE / "data/scanqa/ScanQA_v1.0_train.json",
                 WORKSPACE / "data/scanqa/ScanQA_v1.0_val.json"],
    )
    parser.add_argument("--scan-root", action="append", type=Path, dest="scan_roots")
    parser.add_argument("--output", type=Path, default=HERE / "cache/scene_objects.pt")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    annotations = [path.expanduser().resolve() for path in args.annotations]
    roots = [path.expanduser().resolve() for path in (args.scan_roots or DEFAULT_SCAN_ROOTS)]
    scenes = annotation_scenes(annotations)
    cache, missing = {}, []
    for scene_id in tqdm(scenes, desc="Extracting ScanNet objects"):
        scene_dir = find_scene(scene_id, roots)
        if scene_dir is None:
            missing.append(scene_id)
            continue
        cache[scene_id] = extract_scene(scene_id, scene_dir)

    if missing and not args.allow_missing:
        examples = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Missing ScanNet files for {len(missing)}/{len(scenes)} scenes. "
            f"Examples: {examples}. Download them or use --allow-missing for a smoke test."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 2,
        "geometry_dim": int(next(iter(cache.values()))["geometry"].shape[1]),
        "appearance_dim": int(next(iter(cache.values()))["appearance"].shape[1]),
        "context_dim": int(next(iter(cache.values()))["context"].shape[1]),
        "point_dim": 6,
        "num_object_points": NUM_OBJECT_POINTS,
        "scenes": cache,
        "missing_scenes": missing,
        "scan_roots": [str(path) for path in roots],
    }, args.output)
    print(f"Saved {len(cache)}/{len(scenes)} scenes to {args.output}")
    if missing:
        print(f"Skipped {len(missing)} missing scenes")


if __name__ == "__main__":
    main()

