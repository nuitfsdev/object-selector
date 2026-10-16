"""Download only the three ScanNet files needed for ScanQA train/val scenes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
TOS_URL = "https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf"
SUFFIXES = ("_vh_clean_2.ply", "_vh_clean_2.0.010000.segs.json", ".aggregation.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "both"), default="train")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--accept-scannet-tos", action="store_true")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "data/scannet")
    args = parser.parse_args()
    if not args.accept_scannet_tos:
        parser.error(f"Read and accept {TOS_URL}, then pass --accept-scannet-tos")
    downloader = WORKSPACE / "download-scannet.py"
    splits = ("train", "val") if args.split == "both" else (args.split,)
    scenes = set()
    for split in splits:
        path = WORKSPACE / f"data/scanqa/ScanQA_v1.0_{split}.json"
        scenes.update(row["scene_id"] for row in json.loads(path.read_text(encoding="utf-8")))

    jobs = []
    for scene in sorted(scenes):
        for suffix in SUFFIXES:
            destination = args.output / "scans" / scene / f"{scene}{suffix}"
            if not destination.is_file():
                jobs.append((scene, suffix))
    print(f"Scenes={len(scenes)} files_to_download={len(jobs)}")

    def download(job):
        scene, suffix = job
        subprocess.run([
            sys.executable, str(downloader), "--skip_existing", "--id", scene,
            "--type", suffix, "-o", str(args.output),
        ], input="\n", text=True, check=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 50 == 0 or index == len(futures):
                print(f"Downloaded {index}/{len(futures)} files")


if __name__ == "__main__":
    main()
