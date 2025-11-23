import numpy as np
import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

import decord
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, TimesformerForVideoClassification

from base.utils import VideoFolderDataset, make_collate_fn

import warnings
warnings.filterwarnings(
    "ignore",
    message="Creating a tensor from a list of numpy.ndarrays is extremely slow*",
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_ROOT = PROJECT_ROOT / "finetuning" / "train"
VAL_ROOT = PROJECT_ROOT / "finetuning" / "val"

MODEL_NAME = "facebook/timesformer-base-finetuned-k400"
BATCH_SIZE = 4
NUM_EPOCHS = 8
NUM_WORKERS = 4
NUM_FRAMES = 8
LR = 1e-4


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



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load processor and model
    print(f"Loading model: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    # Build datasets
    train_dataset = VideoFolderDataset(TRAIN_ROOT, num_frames=NUM_FRAMES)
    val_dataset = VideoFolderDataset(VAL_ROOT, num_frames=NUM_FRAMES)

    num_classes = len(train_dataset.classes)
    print(f"Classes: {train_dataset.classes}  (num_classes={num_classes})")

    # Load pretrained Timesformer and adapt classifier to 4 classes
    model = TimesformerForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # replace original K400 head
    )
    model.to(device)

    # Dataloaders
    collate_fn = make_collate_fn(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_ckpt_path = PROJECT_ROOT / "checkpoints"
    best_ckpt_path.mkdir(parents=True, exist_ok=True)
    best_ckpt_file = best_ckpt_path / "timesformer_best.pt"

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Train] loss={train_loss:.4f}  acc={train_acc*100:.2f}%")

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[Val]   loss={val_loss:.4f}  acc={val_acc*100:.2f}%")

        # Save best checkpoint
        if val_acc >= best_val_acc:
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


if __name__ == "__main__":
    main()