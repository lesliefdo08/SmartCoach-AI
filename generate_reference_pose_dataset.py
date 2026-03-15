from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from core.feature_extractor import extract_pose_features
from core.pose_detector import PoseDetector
from core.video_processor import extract_frames, load_video, preprocess_frame


SHOT_MAP = {
    "cover_drive": "cover_drive_pro.json",
    "straight_drive": "straight_drive_pro.json",
    "pull_shot": "pull_shot_pro.json",
    "defense": "defense_pro.json",
}


def _ideal_range(mean: float, std: float, width_scale: float = 1.5) -> List[float]:
    lower = max(0.0, mean - width_scale * std)
    upper = min(180.0, mean + width_scale * std)
    return [round(lower, 2), round(upper, 2)]


def process_video(video_path: Path, detector: PoseDetector, sample_rate: int = 2) -> List[Dict[str, float]]:
    cap = load_video(str(video_path))
    frames, _ = extract_frames(cap, sample_rate=sample_rate)
    cap.release()

    features_list: List[Dict[str, float]] = []
    for frame in frames:
        proc = preprocess_frame(frame)
        keypoints = detector.detect(proc)
        if not keypoints:
            continue
        features = extract_pose_features(keypoints)
        if features:
            features_list.append(features)

    return features_list


def aggregate_features(all_features: List[Dict[str, float]]) -> Dict[str, Dict[str, float | List[float]]]:
    if not all_features:
        return {
            "joint_angles_mean": {},
            "joint_angles_std": {},
            "ideal_ranges": {},
        }

    keys = sorted({k for row in all_features for k in row.keys()})
    mean_dict: Dict[str, float] = {}
    std_dict: Dict[str, float] = {}
    range_dict: Dict[str, List[float]] = {}

    for k in keys:
        vals = np.array([row.get(k, np.nan) for row in all_features], dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue

        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        std_v = max(std_v, 2.0)

        mean_dict[k] = round(mean_v, 2)
        std_dict[k] = round(std_v, 2)
        range_dict[k] = _ideal_range(mean_v, std_v)

    return {
        "joint_angles_mean": mean_dict,
        "joint_angles_std": std_dict,
        "ideal_ranges": range_dict,
    }


def generate_dataset(
    source_dir: Path,
    reference_dir: Path,
    sample_rate: int = 2,
) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    detector = PoseDetector(model_complexity=1)

    try:
        for shot, out_name in SHOT_MAP.items():
            shot_dir = source_dir / shot
            videos = []
            for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
                videos.extend(shot_dir.glob(ext))

            all_features: List[Dict[str, float]] = []
            for video in videos:
                print(f"Processing [{shot}] video: {video.name}")
                all_features.extend(process_video(video, detector=detector, sample_rate=sample_rate))

            profile = aggregate_features(all_features)
            profile["shot_type"] = shot
            profile["source_videos"] = [v.name for v in videos]
            profile["frames_used"] = len(all_features)

            out_path = reference_dir / out_name
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2)

            print(f"Saved {out_path} | videos={len(videos)} frames_used={len(all_features)}")
    finally:
        detector.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate professional statistical reference pose dataset.")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="assets/pro_videos",
        help="Directory containing shot subfolders with pro videos.",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="reference_data",
        help="Output directory for *_pro.json files.",
    )
    parser.add_argument("--sample-rate", type=int, default=2, help="Use every Nth frame.")

    args = parser.parse_args()

    generate_dataset(
        source_dir=Path(args.source_dir),
        reference_dir=Path(args.reference_dir),
        sample_rate=max(1, int(args.sample_rate)),
    )


if __name__ == "__main__":
    main()
