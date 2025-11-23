import torch
from transformers import TimesformerForVideoClassification
import pathlib
from typing import List, Tuple
from pathlib import Path
from torch.utils.data import Dataset
import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


class VideoFolderDataset(Dataset):
    """
    Dataset that expects a folder structure like:
        root/
            plank/
                video1.mp4
                video2.mp4
            pull-up/
                ...
            push-up/
                ...
            squat/
                ...
    Each subfolder name is treated as a class label.
    """

    def __init__(self, root: Path, num_frames: int = 8):
        self.root = Path(root)
        self.num_frames = num_frames

        # class names = subfolder names, sorted for stable label mapping
        self.classes: List[str] = sorted(
            [d.name for d in self.root.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # collect all (video_path, label_idx)
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for p in cls_dir.iterdir():
                if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
                    self.samples.append((p, self.class_to_idx[cls_name]))

        print(f"[Dataset] root={self.root}  classes={self.classes}")
        print(f"[Dataset] Found {len(self.samples)} videos")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames(self, path: Path) -> List:
        """Load evenly spaced frames from a video using decord."""
        # decord needs to be imported here to avoid OpenCV GUI conflicts
        from decord import VideoReader, cpu


        vr = VideoReader(str(path), ctx=cpu(0))
        total = len(vr)
        if total == 0:
            raise RuntimeError(f"No frames found in video: {path}")
        indices = torch.linspace(0, total - 1, steps=min(self.num_frames, total)).long()
        frames = [vr[int(i)].asnumpy() for i in indices]
        return frames

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)
        return frames, label, str(video_path)


def make_collate_fn(processor):
    """
    Collate function that:
      - takes a list of (frames, label, path)
      - passes the list of frames directly to the processor
    """

    def collate(batch):
        frames_list, labels, paths = zip(*batch)  # frames_list: tuple of [list of frames]
        # processor can handle a list of videos, each video = list of frames (np.ndarrays)
        inputs = processor(list(frames_list), return_tensors="pt")
        pixel_values = inputs["pixel_values"]  # (B, T, C, H, W)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return pixel_values, labels_tensor, list(paths)

    return collate


def load_finetuned_model(base_model_name, ckpt_path, device):
    """
    Load Timesformer with correct 4-class head and fine-tuned weights,
    using the class order stored in the checkpoint.
    """
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    if "classes" in ckpt:
        classes = ckpt["classes"]
    else:
        # fallback: infer from ROOT if not saved (but you *did* save it)
        classes = sorted(
            d.name for d in (PROJECT_ROOT / "finetuning" / "train").iterdir() if d.is_dir()
        )

    num_classes = len(classes)

    id2label = {i: cls_name for i, cls_name in enumerate(classes)}
    label2id = {v: k for k, v in id2label.items()}

    # Initialize model with correct head & label mapping
    model = TimesformerForVideoClassification.from_pretrained(
        base_model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Clean state_dict keys (strip 'model.' / 'module.' if present)
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model, classes