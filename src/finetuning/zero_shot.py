import os
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification
import warnings
warnings.filterwarnings(
    "ignore",
    message="Creating a tensor from a list of numpy.ndarrays is extremely slow*",
)

"""
- Load each video in finetuning/train/<class>/*
- Run pretrained Timesformer (K400)
- Take top-1 predicted label
- Compare with folder name
- Compute per-class accuracy and overall accuracy
"""

# ===== Settings =====

# This file is:  src/finetuning/zero_shot.py
# Project root is two levels up from here
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "finetuning", "train"))
# If you want to evaluate on val instead, use:
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "finetuning", "val"))

MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
NUM_FRAMES = 8  # how many frames to sample from each video


# Map K400 labels to your folder names
LABEL_NORMALIZATION = {
    "push up": "push-up",
    "squat": "squat",
    "plank": "plank",
    "russian twist": "russian-twist",
}


# ===== load frames with OpenCV =====

def load_frames_with_cv2(path, num_frames=8):
    """
    Load evenly-spaced frames from a video using OpenCV.
    - Returns a list of exactly `num_frames` RGB numpy arrays.
    - If the video cannot be read or we get fewer than `num_frames`, returns None.
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

    # Choose indices evenly from [0, frame_count - 1]
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Failed to read frame {idx} from {path}")
            # If any frame failed, treat the whole video as unusable
            cap.release()
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    # Extra safety: ensure we really have num_frames frames
    if len(frames) < num_frames:
        print(
            f"[WARN] Only got {len(frames)} frames (need {num_frames}) from {path}, skipping"
        )
        return None

    return frames


# ===== Main evaluation =====

def main():
    print(f"Dataset root: {ROOT}")
    print(f"Loading pretrained model: {MODEL_NAME}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = TimesformerForVideoClassification.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Get category names from folder structure
    categories = sorted(
        d for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d))
    )
    print("Categories:", categories)

    # Accuracy stats per category
    stats = {cat: {"correct": 0, "total": 0} for cat in categories}

    # Loop over each category and each video
    for category in categories:
        video_paths = glob.glob(os.path.join(ROOT, category, "*"))

        for video_path in video_paths:
            frames = load_frames_with_cv2(video_path, NUM_FRAMES)
            if frames is None:
                # Skip unreadable / problematic videos
                continue

            # Prepare inputs for Timesformer
            inputs = processor(frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = logits.softmax(dim=-1)[0]

            top_idx = probs.argmax().item()
            pred_label = model.config.id2label[top_idx]  # e.g. "push up"
            pred_norm = LABEL_NORMALIZATION.get(pred_label, None)

            stats[category]["total"] += 1
            if pred_norm == category:
                stats[category]["correct"] += 1

    # ===== Print results =====
    print("\n=== Zero-shot accuracy ===")
    total_correct = 0
    total_total = 0

    for cat, s in stats.items():
        if s["total"] > 0:
            acc = 100.0 * s["correct"] / s["total"]
        else:
            acc = 0.0
        total_correct += s["correct"]
        total_total += s["total"]
        print(f"{cat:15s} : {acc:6.2f}%  ({s['correct']}/{s['total']})")

    if total_total > 0:
        overall_acc = 100.0 * total_correct / total_total
    else:
        overall_acc = 0.0

    print(f"\nOverall accuracy : {overall_acc:6.2f}%  ({total_correct}/{total_total})")


if __name__ == "__main__":
    main()