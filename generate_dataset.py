from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from modules.video_processor import CricketAnalyticsPipeline


REQUIRED_LABEL_DIRS = ["defensive", "drive", "pull", "lofted"]
SUPPORTED_LABELS = ["defensive", "drive", "pull", "lofted", "cut", "sweep"]
VIDEO_EXTS = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm")


def _load_metadata_map(metadata_csv: Path) -> Dict[str, Dict[str, str]]:
    if not metadata_csv.exists():
        return {}

    df = pd.read_csv(metadata_csv)
    if "video_file" not in df.columns:
        return {}

    out: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        video_file = str(row.get("video_file", "")).strip()
        if not video_file:
            continue
        out[video_file] = {
            "shot_type": str(row.get("shot_type", "")).strip(),
            "camera_angle": str(row.get("camera_angle", "unknown")).strip() or "unknown",
            "lighting": str(row.get("lighting", "unknown")).strip() or "unknown",
            "handedness": str(row.get("handedness", "unknown")).strip() or "unknown",
        }
    return out


def _sidecar_metadata(video_path: Path) -> Dict[str, str]:
    sidecar = video_path.with_suffix(".meta.json")
    if not sidecar.exists():
        return {}
    try:
        with sidecar.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    return {
        "shot_type": str(payload.get("shot_type", "")).strip(),
        "camera_angle": str(payload.get("camera_angle", "unknown")).strip() or "unknown",
        "lighting": str(payload.get("lighting", "unknown")).strip() or "unknown",
        "handedness": str(payload.get("handedness", "unknown")).strip() or "unknown",
    }


def _collect_videos(label_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in VIDEO_EXTS:
        files.extend(sorted(label_dir.glob(ext)))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structured feature dataset from labeled cricket videos.")
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--metadata-csv", type=str, default="dataset/metadata.csv")
    parser.add_argument("--output-csv", type=str, default="dataset/features_dataset.csv")
    parser.add_argument("--frame-output-csv", type=str, default="dataset/frame_features.csv")
    parser.add_argument("--output-jsonl", type=str, default="dataset/features_dataset.jsonl")
    parser.add_argument("--sample-rate", type=int, default=2)
    parser.add_argument("--strict-filter", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    for label in REQUIRED_LABEL_DIRS:
        (dataset_root / label).mkdir(parents=True, exist_ok=True)

    metadata_map = _load_metadata_map(Path(args.metadata_csv))
    pipeline = CricketAnalyticsPipeline(sample_rate=max(1, args.sample_rate))

    rows: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []

    try:
        label_dirs = [d for d in dataset_root.iterdir() if d.is_dir() and d.name.lower() in SUPPORTED_LABELS]
        for label_dir in sorted(label_dirs):
            label = label_dir.name.lower()
            videos = _collect_videos(label_dir)
            for video in videos:
                rel_video = str(video.relative_to(dataset_root)).replace("\\", "/")

                meta = {
                    "shot_type": label,
                    "camera_angle": "unknown",
                    "lighting": "unknown",
                    "handedness": "unknown",
                }
                csv_meta = metadata_map.get(rel_video, metadata_map.get(video.name, {}))
                sidecar_meta = _sidecar_metadata(video)
                meta.update({k: v for k, v in csv_meta.items() if v})
                meta.update({k: v for k, v in sidecar_meta.items() if v})
                if not meta.get("shot_type"):
                    meta["shot_type"] = label

                out = pipeline.process_video_filtered(video, strict_filter=args.strict_filter)
                windows = out.get("window_features", [])
                frame_features = out.get("features_by_frame", [])

                for i, feat in enumerate(windows, start=1):
                    row: Dict[str, object] = {
                        "video_file": rel_video,
                        "window_index": i,
                        "shot_type": str(meta.get("shot_type", label)),
                        "camera_angle": str(meta.get("camera_angle", "unknown")),
                        "lighting": str(meta.get("lighting", "unknown")),
                        "handedness": str(meta.get("handedness", "unknown")),
                        "valid_frame_count": int(len(out.get("key_features", []))),
                        "total_frame_count": int(len(frame_features)),
                    }
                    row.update({k: float(v) for k, v in feat.items()})
                    rows.append(row)

                for i, feat in enumerate(frame_features, start=1):
                    row_f: Dict[str, object] = {
                        "video_file": rel_video,
                        "frame_index": i,
                        "shot_type": str(meta.get("shot_type", label)),
                        "camera_angle": str(meta.get("camera_angle", "unknown")),
                        "lighting": str(meta.get("lighting", "unknown")),
                        "handedness": str(meta.get("handedness", "unknown")),
                    }
                    row_f.update({k: float(v) for k, v in feat.items() if isinstance(v, (int, float))})
                    frame_rows.append(row_f)
    finally:
        pipeline.close()

    if not rows:
        print("No feature windows generated. Add labeled videos under dataset/<shot_type>/ and rerun.")
        return

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    frame_output_csv = Path(args.frame_output_csv)
    frame_output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame_rows).to_csv(frame_output_csv, index=False)

    output_jsonl = Path(args.output_jsonl)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Saved window dataset: {output_csv} ({len(df)} rows)")
    print(f"Saved frame dataset : {frame_output_csv} ({len(frame_rows)} rows)")
    print(f"Saved JSONL dataset : {output_jsonl}")


if __name__ == "__main__":
    main()
