import torch
import os
import glob
from pathlib import Path
import decord
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, TimesformerForVideoClassification

from base.utils import load_finetuned_model

"""
Evaluate fine-tuned Timesformer on your finetuning dataset.

- Loads best checkpoint from checkpoints/timesformer_best.pt
- Uses class order saved in the checkpoint
- Computes per-class accuracy + overall accuracy
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(SCRIPT_DIR).resolve().parent
ROOT = PROJECT_ROOT / "finetuning" / "val"   # "train" or "val"
MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "timesformer_best.pt"
NUM_FRAMES = 8


def load_frames(path, num_frames=8):
    """Load evenly-spaced frames from a video using decord."""
    vr = VideoReader(str(path), ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise RuntimeError(f"No frames found in video: {path}")
    indices = torch.linspace(0, total - 1, steps=min(num_frames, total)).long()
    frames = [vr[int(i)].asnumpy() for i in indices]
    return frames



def main():
    print(f"Dataset root: {ROOT}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model, classes = load_finetuned_model(MODEL_NAME, CKPT_PATH, device)

    # List categories from folder structure (same names as classes)
    categories = sorted(
        d.name for d in ROOT.iterdir() if d.is_dir()
    )
    print(f"Folders (categories): {categories}")

    # Stats container for each category
    stats = {cat: {"correct": 0, "total": 0} for cat in categories}

    # Iterate through each category and video
    for category in categories:
        video_paths = glob.glob(str(ROOT / category / "*"))

        for video_path in video_paths:
            frames = load_frames(video_path, NUM_FRAMES)

            inputs = processor([frames], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                probs = logits.softmax(dim=-1)[0]

            top_idx = probs.argmax().item()
            pred_label = model.config.id2label[top_idx]  # this now matches ckpt['classes']

            stats[category]["total"] += 1
            if pred_label == category:
                stats[category]["correct"] += 1

    # Print summary
    total_correct = 0
    total_total = 0

    print("\n=== Fine-tuned model performance (per class) ===")
    for cat, s in stats.items():
        acc = (s["correct"] / s["total"] * 100) if s["total"] else 0.0
        total_correct += s["correct"]
        total_total += s["total"]
        print(f"{cat:7s} : {acc:6.2f}%  ({s['correct']}/{s['total']})")

    overall_acc = (total_correct / total_total * 100) if total_total else 0.0
    print(f"Overall : {overall_acc:6.2f}%  ({total_correct}/{total_total})")


if __name__ == "__main__":
    main()
