from pathlib import Path
from typing import List, Tuple
import logging
import re

import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, TimesformerForVideoClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


logging.basicConfig(
    level=logging.INFO, 
    format='[base/utils.py] %(message)s'
)
logger = logging.getLogger()



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

    def __init__(self, root: Path, num_frames: int = 8, exts=(".mp4", ".MP4"), use_short_segments: bool = False):
        """
        root_dir: directory that contains class subfolders (e.g. train/plank, train/squat, ...)
        num_frames: how many frames to sample per video
        exts: tuple of allowed video file extensions
        """
        self.root = Path(root)
        self.num_frames = num_frames
        self.use_short_segments = use_short_segments
        
        # normalize extensions to lowercase, e.g. ".MP4" -> ".mp4"
        self.exts = {ext.lower() for ext in exts}

        # class names = subfolder names, sorted for stable label mapping
        self.classes: List[str] = sorted(
            [d.name for d in self.root.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

        # collect all (video_path, label_idx) for allowed extensions only
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for p in cls_dir.iterdir():
                if not (p.is_file() and p.suffix.lower() in self.exts):
                    continue
                
                if self.use_short_segments: #short  video; expects XXXX_segNNN"
                    if not re.search(r"_seg\d{3}$", p.stem):
                        continue
                
                self.samples.append((p, self.class_to_idx[cls_name]))

        print(f"[Dataset] root={self.root}  classes={self.classes}")
        print(f"[Dataset] Found {len(self.samples)} videos (exts={self.exts})")
        
        if self.use_short_segments and len(self.samples) == 0:
            logger.warning(
                "[Dataset] use_short_segments=True, but no videos matching the pattern "
                "'*_segNNN.<ext>' were found under %s. "
                "Please check that your short video files are named like 'XXXX_seg001.mp4'.",
                self.root,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames(self, path: Path) -> List:
        """Load evenly spaced frames from a video using decord."""
        from decord import VideoReader, cpu

        vr = VideoReader(str(path), ctx=cpu(0))
        total = len(vr)
        if total == 0:
            raise RuntimeError(f"No frames found in video: {path}")

        indices = torch.linspace(
            0, total - 1, steps=min(self.num_frames, total)
        ).long()
        frames = [vr[int(i)].asnumpy() for i in indices]
        return frames

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)
        return frames, label, str(video_path)


def make_collate_fn(processor, transform=None):
    """
    Collate function that:
      - takes a list of (frames, label, path)
      -  (optionally) applies transform to frames
      - passes the list of frames directly to the processor
    """

    def collate(batch):
        frames_list, labels, paths = zip(*batch)  # frames_list: tuple of [list of frames]
        
        # Augmentation
        if transform is not None:
            frames_list_proc = [transform(frames) for frames in frames_list]
        else:
            frames_list_proc = frames_list
        
        
        # processor can handle a list of videos, each video = list of frames (np.ndarrays)
        inputs = processor(list(frames_list_proc), return_tensors="pt")
        pixel_values = inputs["pixel_values"]  # (B, T, C, H, W)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return pixel_values, labels_tensor, list(paths)

    return collate

def load_model(model_flag: str, checkpoint_path: Path = None, device: str = "cpu"):
    """
    Load either pretrained or finetuned model.

    Available model flags:
    - timesformer: facebook/timesformer-base-finetuned-k400
    """


    if model_flag == "timesformer":
        model_name = "facebook/timesformer-base-finetuned-k400"
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)


        if checkpoint_path:
            logger.info(f"Loading finetuned {model_name} from {checkpoint_path}")
            model, classes = load_finetuned_model(model_name, checkpoint_path, device)
            return processor, model, classes
        
        else:
            logger.info(f"Loading pretrained {model_name}")
            model = TimesformerForVideoClassification.from_pretrained(model_name).to(device)
            return processor, model, None
    
    else:
        raise ValueError(f"Invalid model: {model_flag}")

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