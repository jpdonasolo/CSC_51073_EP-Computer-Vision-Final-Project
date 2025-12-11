import re
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import random
from collections import Counter
import wandb

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import decord
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, TimesformerForVideoClassification

import warnings
warnings.filterwarnings(
    "ignore",
    message="Creating a tensor from a list of numpy.ndarrays is extremely slow*",
)

SRC_DIR = Path(__file__).resolve().parents[1]  # .../src
sys.path.append(str(SRC_DIR))

from base.utils import VideoFolderDataset, make_collate_fn
from base.transforms import VideoRandomAugment #change augmentation method here

MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
BATCH_SIZE = 4
NUM_EPOCHS = 8
NUM_WORKERS = 4
NUM_FRAMES = 8
LR = 1e-4

BALANCE_MODE = "none"       # "none" / "min" / "2min_flip" / "max_full"
FRAME_SAMPLING = "even"     # "even", "first", "random", "center"
NUM_UNFROZEN_LAYERS = 2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = PROJECT_ROOT / "finetuning" / "train"
VAL_ROOT   = PROJECT_ROOT / "finetuning" / "val"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def freeze_layers(model, num_unfrozen_layers):
    # TimesFormer encoder layers
    encoder_layers = model.timesformer.encoder.layer
    total_layers = len(encoder_layers)
    print(f"Total encoder layers: {total_layers}")
    print(f"Unfreezing last {num_unfrozen_layers} layers")

    for name, param in model.named_parameters():

        # Encoder layers
        if "timesformer.encoder.layer" in name:
            match = re.search(r"layer\.(\d+)", name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx < total_layers - num_unfrozen_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        # Classification head + embeddings
        else:
            param.requires_grad = True

    # Debug summary
    trainable = sum(p.requires_grad for p in model.parameters())
    total = sum(1 for _ in model.parameters())
    print(f"Trainable params: {trainable}/{total}")


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for pixel_values, labels, _ in dataloader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


def evaluate(
    model,
    dataloader,
    device,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for pixel_values, labels, _ in dataloader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


# calculate weight
def make_sample_weights_and_num_samples(train_dataset: VideoFolderDataset, mode: str = "none") -> Tuple[Optional[List[float]], int, bool]:
    """
    mode:
    - "none"      : no class balancing (use the original distribution)
    - "min"       : match the smallest class (downsampling)
    - "2min_flip" : target = 2 × min_count
    - "max_full"  : target = max_count

    Returns:
    - sample_weights: weights passed to WeightedRandomSampler (None when mode="none")
    - num_samples   : num_samples for the sampler
    - use_sampler   : whether to use the sampler
    """
    labels = []
    for i in range(len(train_dataset)):
        _, lab, _ = train_dataset[i]
        labels.append(lab)

    label_counts = Counter(labels)
    num_classes = len(label_counts)
    min_count = min(label_counts.values())
    max_count = max(label_counts.values())
    print("Label counts:", label_counts)
    print(f"[Balance mode={mode}] min={min_count}, max={max_count}, num_classes={num_classes}")
    
    if mode == "none":
        # W/O sampler
        num_samples = len(train_dataset)
        return None, num_samples, False

    if mode == "min":
        target_per_class = min_count
    elif mode == "2min_flip":
        target_per_class = 2 * min_count
    elif mode == "max_full":
        target_per_class = max_count
    else:
        raise ValueError(f"Unknown balance mode: {mode}")
    
    class_weights = {cls_idx: 1.0 / cnt for cls_idx, cnt in label_counts.items()}
    sample_weights = [class_weights[lab] for lab in labels]

    num_samples = target_per_class * num_classes

    print(f"[Balance mode={mode}] num_samples per epoch = {num_samples}")
    print("Class weights:", class_weights)
    
    return sample_weights, num_samples, True


def run_experiment(
    balance_mode: str,
    frame_sampling: str,
    num_unfrozen_layers: int,
    seed: int = 42):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ---------- wandb init ----------
    wandb.init(
        project="CV2025 - Video Classification",
        config={
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "num_workers": NUM_WORKERS,
            "num_frames": NUM_FRAMES,
            "lr": LR,
            "balance_mode": balance_mode,
            "frame_sampling": frame_sampling,
            "num_unfrozen_layers": num_unfrozen_layers,
        },
    )
    # -------------------------------
    
    # Load processor and model
    print(f"Loading model: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    # Build datasets
    train_dataset = VideoFolderDataset(TRAIN_ROOT, num_frames=NUM_FRAMES, exts=(".mp4", ".MP4"),  use_short_segments = True, frame_sampling=frame_sampling)
    val_dataset = VideoFolderDataset(VAL_ROOT, num_frames=NUM_FRAMES, exts=(".mp4", ".MP4"), use_short_segments = True, frame_sampling=frame_sampling)

    num_classes = len(train_dataset.classes)
    print(f"Classes: {train_dataset.classes}  (num_classes={num_classes})")

    # Load pretrained Timesformer and adapt classifier to 4 classes
    model = TimesformerForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # replace original K400 head
    )
    model.to(device)
    
    #Unfreeze layers
    freeze_layers(model, num_unfrozen_layers=num_unfrozen_layers)
    
    # Define augumentation
    if balance_mode == "none":
        train_transform = VideoRandomAugment(
            num_frames=NUM_FRAMES,
            flip_prob=0.0, #0.5
            color_jitter_prob=0.0, #True
            use_augmentation=False
        )
        
    elif balance_mode == "min": #down sampling
        train_transform = VideoRandomAugment(
            num_frames=NUM_FRAMES,
            flip_prob=0.0,
            color_jitter_prob=0.0,
            use_augmentation=False,
        )
        
    elif balance_mode == "2min_flip":
        # target = 2 * min_class
        train_transform = VideoRandomAugment(
            num_frames=NUM_FRAMES,
            flip_prob=0.5,       # flip only
            color_jitter_prob=0.0,
            use_augmentation=True,
        )
        
    elif balance_mode == "max_full":
        # target = max_class, flip + temporal crop
        train_transform = VideoRandomAugment(
            num_frames=NUM_FRAMES,
            flip_prob=0.5,
            color_jitter_prob=0.5,
            use_augmentation=True,
        )
    else:
        raise ValueError(f"Unknown BALANCE_MODE: {balance_mode}")


    # Dataloaders
    train_collate_fn = make_collate_fn(processor, transform=train_transform)
    val_collate_fn   = make_collate_fn(processor, transform=None) # No augmentation for validation

    # Sampler for class balance
    sample_weights, num_samples, use_sampler = make_sample_weights_and_num_samples(
        train_dataset,
        mode=balance_mode,
    )
    if use_sampler:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True  # BALANCE_MODE == "none": shuffle
  
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=NUM_WORKERS, # with sampler -> False
        collate_fn=train_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=val_collate_fn,
    )

    # Optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=LR) # exclude freezed layers 

    best_val_acc = 0.0
    best_ckpt_path = PROJECT_ROOT / "checkpoints"
    best_ckpt_path.mkdir(parents=True, exist_ok=True)
    best_ckpt_file = best_ckpt_path / f"timesformer_{balance_mode}_{frame_sampling}_{num_unfrozen_layers}.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Train] loss={train_loss:.4f}  acc={train_acc*100:.2f}%")

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[Val]   loss={val_loss:.4f}  acc={val_acc*100:.2f}%")
        
        # ---------- wandb log ----------
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
            }
        )
        # -------------------------------

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_dataset.classes,
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                best_ckpt_file,
            )
            print(f"New best model saved to: {best_ckpt_file} (val_acc={val_acc*100:.2f}%)")

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Best checkpoint: {best_ckpt_file}")
    wandb.finish()


def main():
    run_experiment(
        balance_mode=BALANCE_MODE,
        frame_sampling=FRAME_SAMPLING,
        num_unfrozen_layers=NUM_UNFROZEN_LAYERS,
    )
    
if __name__ == "__main__":
    main()