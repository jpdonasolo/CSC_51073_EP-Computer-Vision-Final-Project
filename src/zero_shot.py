import torch
import os
import glob
from pathlib import Path
import decord
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, TimesformerForVideoClassification

'''
load each video
run pretrained Timesformer (K400)
take top-1 predicted label
map pretrained labels → finetuning folder names
"pull ups" → "pull-up"
etc.
check correctness
compute per-category accuracy & overall accuracy
print nicely
'''


# Settings
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "finetuning", "train"))  #train #val
MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
NUM_FRAMES = 8


# Normalize pretrained K400 labels → Finetuning folder names
# Example:
#   pretrained: "pull ups"  →  folder: "pull-up"
LABEL_NORMALIZATION = {
    "pull ups": "pull-up",
    "push up": "push-up",
    "squat": "squat",
    "plank": "plank"
}


def load_frames(path, num_frames=8):
    """Load evenly-spaced frames from a video using decord."""
    vr = VideoReader(str(path), ctx=cpu(0))
    total = len(vr)
    indices = torch.linspace(0, total - 1, steps=min(num_frames, total)).long().tolist()
    frames = [vr[i].asnumpy() for i in indices]
    return frames


def main():
    print(f"Loading pretrained model: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = TimesformerForVideoClassification.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # List action categories from folder structure
    categories = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])

    # Stats container for each category
    stats = {cat: {"correct": 0, "total": 0} for cat in categories}

    # Iterate through each category and each video
    for category in categories:
        video_paths = glob.glob(f"{ROOT}/{category}/*")

        for video_path in video_paths:
            frames = load_frames(video_path, NUM_FRAMES)

            inputs = processor(frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = logits.softmax(dim=-1)[0]

            # Top-1 predicted index → label string
            top_idx = probs.argmax().item()
            pred_label = model.config.id2label[top_idx]    # e.g., "pull ups"

            # Convert K400 label → our folder name
            pred_norm = LABEL_NORMALIZATION.get(pred_label, None)

            stats[category]["total"] += 1
            is_correct = (pred_norm == category)

            if is_correct:
                stats[category]["correct"] += 1


# Print summary results
    total_correct = 0
    total_total = 0

    for cat, s in stats.items():
        acc = (s["correct"] / s["total"] * 100) if s["total"] else 0
        total_correct += s["correct"]
        total_total += s["total"]
        print(f"{cat:7s} : {acc:6.2f}%  ({s['correct']}/{s['total']})")

    # Overall
    overall_acc = (total_correct / total_total * 100) if total_total else 0
    print(f"Overall : {overall_acc:6.2f}%  ({total_correct}/{total_total})")

if __name__ == "__main__":
    main()