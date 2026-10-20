"""Download only the three ScanNet files needed for ScanQA train/val scenes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
TOS_URL = "http://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf"
SCANNET_BASE_URL = "http://kaldir.vc.cit.tum.de/scannet/v2/scans"
SUFFIXES = ("_vh_clean_2.ply", "_vh_clean_2.0.010000.segs.json", ".aggregation.json")
# ScanNet's official download-scannet.py uses the same context because its
# HTTP endpoint redirects to an HTTPS server with an incomplete certificate chain.
SCANNET_SSL_CONTEXT = ssl._create_unverified_context()


def verify_scene(scene: str, output_root: Path) -> list[str]:
    """Parse the three required files and return any integrity errors."""
    scene_dir = output_root / "scans" / scene
    errors: list[str] = []
    paths = {suffix: scene_dir / f"{scene}{suffix}" for suffix in SUFFIXES}
    for suffix, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"{path}: missing or empty")
    if errors:
        return errors

    try:
        aggregation = json.loads(paths[".aggregation.json"].read_text(encoding="utf-8"))
        if not isinstance(aggregation.get("segGroups"), list):
            raise ValueError("missing list field 'segGroups'")
    except Exception as exc:
        errors.append(f"{paths['.aggregation.json']}: {exc}")

    segs_suffix = "_vh_clean_2.0.010000.segs.json"
    try:
        segmentation = json.loads(paths[segs_suffix].read_text(encoding="utf-8"))
        if not isinstance(segmentation.get("segIndices"), list):
            raise ValueError("missing list field 'segIndices'")
    except Exception as exc:
        errors.append(f"{paths[segs_suffix]}: {exc}")

    ply_suffix = "_vh_clean_2.ply"
    try:
        from plyfile import PlyData

        ply = PlyData.read(str(paths[ply_suffix]))
        if len(ply["vertex"]) == 0:
            raise ValueError("PLY has no vertices")
        if len(ply["face"]) == 0:
            raise ValueError("PLY has no faces")
    except Exception as exc:
        errors.append(f"{paths[ply_suffix]}: {exc}")
    return errors


def download_file(scene: str, suffix: str, destination: Path, retries: int = 3) -> None:
    """Download one public ScanNet file atomically, with bounded retries."""
    url = f"{SCANNET_BASE_URL}/{scene}/{scene}{suffix}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ScanQA-object-selector/1.0"})
            with urllib.request.urlopen(
                request, timeout=120, context=SCANNET_SSL_CONTEXT
            ) as response:
                content_length = response.headers.get("Content-Length")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            downloaded_size = temporary.stat().st_size
            if downloaded_size == 0:
                raise OSError("downloaded an empty file")
            if content_length is not None and downloaded_size != int(content_length):
                raise OSError(
                    f"incomplete download: expected {content_length} bytes, got {downloaded_size}"
                )
            temporary.replace(destination)
            return
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            print(
                f"Download failed ({attempt}/{retries}) {url}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "both"), default="train")
    parser.add_argument(
        "--scene-id",
        action="append",
        help="Download one specific scene; may be passed more than once.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--accept-scannet-tos", action="store_true")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "data/scannet")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Parse and validate every requested PLY and JSON file after downloading.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate existing files without downloading; implies --verify.",
    )
    args = parser.parse_args()
    if not args.accept_scannet_tos:
        parser.error(f"Read and accept {TOS_URL}, then pass --accept-scannet-tos")
    scenes = set(args.scene_id or [])
    if not scenes:
        splits = ("train", "val") if args.split == "both" else (args.split,)
        for split in splits:
            path = WORKSPACE / f"data/scanqa/ScanQA_v1.0_{split}.json"
            scenes.update(row["scene_id"] for row in json.loads(path.read_text(encoding="utf-8")))

    jobs = []
    if not args.verify_only:
        for scene in sorted(scenes):
            for suffix in SUFFIXES:
                destination = args.output / "scans" / scene / f"{scene}{suffix}"
                if not destination.is_file() or destination.stat().st_size == 0:
                    jobs.append((scene, suffix))
    print(f"Scenes={len(scenes)} files_to_download={len(jobs)}")

    def download(job):
        scene, suffix = job
        destination = args.output / "scans" / scene / f"{scene}{suffix}"
        download_file(scene, suffix, destination)

    if jobs:
        pool = ThreadPoolExecutor(max_workers=args.workers)
        futures = [pool.submit(download, job) for job in jobs]
        try:
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                if index % 50 == 0 or index == len(futures):
                    print(f"Downloaded {index}/{len(futures)} files")
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            print("Download cancelled. Completed files were kept; run again to resume.", file=sys.stderr)
            os._exit(130)
        except BaseException:
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

    if args.verify or args.verify_only:
        print(f"Verifying {len(scenes)} scenes by parsing PLY and JSON files...")
        integrity_errors: list[str] = []
        for index, scene in enumerate(sorted(scenes), 1):
            integrity_errors.extend(verify_scene(scene, args.output))
            if index % 25 == 0 or index == len(scenes):
                print(f"Verified {index}/{len(scenes)} scenes")
        if integrity_errors:
            print(f"Verification failed with {len(integrity_errors)} invalid files:", file=sys.stderr)
            for error in integrity_errors:
                print(f"  {error}", file=sys.stderr)
            raise SystemExit(1)
        print(f"VERIFIED: {len(scenes)} scenes, {len(scenes) * len(SUFFIXES)} files")


if __name__ == "__main__":
    main()
