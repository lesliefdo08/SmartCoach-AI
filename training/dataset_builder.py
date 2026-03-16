from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


QUERIES: Dict[str, str] = {
    "defensive": "cricket defensive shot slow motion",
    "drive": "cricket cover drive slow motion",
    "lofted": "cricket lofted shot slow motion",
    "pull": "cricket pull shot slow motion",
    "cut": "cricket cut shot slow motion",
}


def _download_label_videos(label: str, query: str, out_dir: Path, max_videos: int) -> None:
    from yt_dlp import YoutubeDL

    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "mp4/bestvideo+bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "merge_output_format": "mp4",
        "ignoreerrors": True,
        "match_filter": lambda info, *args, **kwargs: "video too long" if info.get("duration") and info["duration"] > 240 else None,
    }

    search = f"ytsearch{max_videos}:{query}"
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([search])


def _extract_frames(video_path: Path, frames_dir: Path, sample_step: int = 5) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    idx = 0
    saved = 0
    stem = video_path.stem
    clip_dir = frames_dir / stem
    clip_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_step == 0:
            out = clip_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out), frame)
            saved += 1
        idx += 1

    cap.release()
    return saved


def build_dataset(dataset_root: Path, max_videos_per_class: int, frame_step: int) -> None:
    for label, query in QUERIES.items():
        label_dir = dataset_root / label
        _download_label_videos(label, query, label_dir, max_videos=max_videos_per_class)

        videos = list(label_dir.glob("*.mp4")) + list(label_dir.glob("*.mkv")) + list(label_dir.glob("*.webm"))
        frames_dir = label_dir / "frames"

        total_saved = 0
        for video in videos:
            total_saved += _extract_frames(video, frames_dir, sample_step=frame_step)

        print(f"[{label}] videos={len(videos)} extracted_frames={total_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build cricket shot dataset from YouTube clips.")
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--max-videos", type=int, default=8)
    parser.add_argument("--frame-step", type=int, default=5)
    args = parser.parse_args()

    build_dataset(Path(args.dataset_root), max_videos_per_class=args.max_videos, frame_step=args.frame_step)
