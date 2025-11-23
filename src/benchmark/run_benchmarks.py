import sys
from pathlib import Path
from typing import List
import argparse
import time
import logging
import random

import torch
from torch.utils.data import Subset

logging.basicConfig(
    level=logging.INFO, 
    format='[benchmark/run_benchmarks.py] %(message)s'
)
logger = logging.getLogger()


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src/"))

from utils import load_model, calculate_dataset_statistics, calculate_gpu_memory_usage
from base.utils import VideoFolderDataset, make_collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="timesformer")
    parser.add_argument("--batch_sizes", type=List[int], default=[1, 4])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    return parser.parse_args()


def benchmark_model(model, dataloader, device: str):
    total_time = 0
    model.to(device)
    model.eval()

    peak_memory = 0
    memory_usage_before = calculate_gpu_memory_usage()

    with torch.no_grad():
        for pixel_values, labels, _ in dataloader:

            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            
            time_start = time.time()
            _ = model(pixel_values=pixel_values, labels=labels)
            total_time += time.time() - time_start

            allocated_memory = calculate_gpu_memory_usage()
            if allocated_memory - memory_usage_before > peak_memory:
                peak_memory = allocated_memory - memory_usage_before

    return total_time, peak_memory


def main(
    model_name: str,
    batch_sizes: List[int],
    device: str,
    sample_fraction: float,
):

    CKPT_PATH = PROJECT_ROOT / "checkpoints" / "timesformer_best.pt"

    allocated_memory_before = calculate_gpu_memory_usage()
    processor, model, model_classes = load_model(model_name, CKPT_PATH, device)

    logger.info(f"GPU memory usage after loading the model: {calculate_gpu_memory_usage() - allocated_memory_before:.2f}MB")

    dataset_root = PROJECT_ROOT / "finetuning" / "train"
    dataset_categories = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
    logger.info(f"Dataset categories: {dataset_categories}")
    logger.info(f"Model classes: {model_classes}")

    dataset = VideoFolderDataset(dataset_root)
    if sample_fraction < 1.0:
        num_samples = int(len(dataset) * sample_fraction)
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = Subset(dataset, indices)
    
    num_videos, average_duration_seconds = calculate_dataset_statistics(dataset)
    
    logger.info(f"Number of videos: {num_videos}")
    logger.info(f"Average video duration: {average_duration_seconds:.2f}s")

    collate_fn = make_collate_fn(processor)
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size={batch_size} using device={device}")

        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=collate_fn
        )
        total_time, peak_memory = benchmark_model(model, dataloader, device)
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Average second(s) to process a video: {total_time / num_videos:.2f}s")
        logger.info(f"Average second(s) to process a second of video: {total_time / (average_duration_seconds * num_videos):.2f}s")
        logger.info(f"Peak dataset memory usage during inference: {peak_memory:.2f}MB")
    

if __name__ == "__main__":
    args = parse_args()
    main(**args.__dict__)