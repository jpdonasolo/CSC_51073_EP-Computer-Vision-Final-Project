import os
import sys
from pathlib import Path
import glob

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification

# Make sure we can import from src/base
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_DIR))

from base.utils import load_finetuned_model


"""
Evaluate fine-tuned Timesformer on your finetuning dataset.

- Loads best checkpoint from checkpoints/timesformer_best.pt
- Uses class order saved in the checkpoint
- Computes per-class accuracy + overall accuracy
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROOT = PROJECT_ROOT / "finetuning" / "train"   # or "train"

MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "timesformer_best.pt"
NUM_FRAMES = 8


def load_frames(path: Path, num_frames: int = 8):
    """
    Load evenly-spaced frames from a video using OpenCV.
    - Returns a list of exactly `num_frames` RGB frames.
    - If something goes wrong, returns None and the caller can skip this video.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print(f"[WARN] No frames in video: {path}")
        cap.release()
        return None

    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Failed to read frame {idx} from {path}")
            cap.release()
            return None

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        print(
            f"[WARN] Only got {len(frames)} frames (need {num_frames}) from {path}, skipping"
        )
        return None

    return frames


def main():
    print(f"Dataset root: {ROOT}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model, classes = load_finetuned_model(MODEL_NAME, CKPT_PATH, device)
    model.to(device)

    print(f"Checkpoint classes: {classes}")

    # List categories (folder names under ROOT)
    categories = sorted(d.name for d in ROOT.iterdir() if d.is_dir())
    print(f"Folders (categories): {categories}")

    # Stats container for each category
    stats = {cat: {"correct": 0, "total": 0} for cat in categories}

    # Iterate through each category and its .mp4 videos
    for category in categories:
        cat_dir = ROOT / category
        # only use .mp4 files (avoid .mov, .avi, etc.)
        video_paths = list(cat_dir.glob("*.mp4")) + list(cat_dir.glob("*.MP4"))

        for video_path in video_paths:
            frames = load_frames(video_path, NUM_FRAMES)
            if frames is None:
                # unreadable / broken video -> skip
                continue

            # processor expects a list of frames (list of HxWxC)
            inputs = processor(frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = logits.softmax(dim=-1)[0]

            top_idx = probs.argmax().item()
            pred_label = model.config.id2label[top_idx]  # should match ckpt['classes']

            stats[category]["total"] += 1
            if pred_label == category:
                stats[category]["correct"] += 1

    # Print summary
    total_correct = 0
    total_total = 0

    print("\n=== Fine-tuned accuracy ===")
    for cat, s in stats.items():
        acc = (s["correct"] / s["total"] * 100) if s["total"] else 0.0
        total_correct += s["correct"]
        total_total += s["total"]
        print(f"{cat:15s} : {acc:6.2f}%  ({s['correct']}/{s['total']})")

    overall_acc = (total_correct / total_total * 100) if total_total else 0.0
    print(f"\nOverall : {overall_acc:6.2f}%  ({total_correct}/{total_total})")


if __name__ == "__main__":
    main()