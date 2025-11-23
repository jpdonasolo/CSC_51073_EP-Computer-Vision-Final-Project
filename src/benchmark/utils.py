import logging
import sys
from pathlib import Path
from typing import Tuple
import cv2
import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src/"))

# User imports
from base.utils import load_finetuned_model, VideoFolderDataset

# Logging
logging.basicConfig(
    level=logging.INFO, 
    format='[benchmark/utils.py] %(message)s'
)
logger = logging.getLogger()


CKPT_PATH = PROJECT_ROOT / "checkpoints" / "timesformer_best.pt"


def calculate_dataset_statistics(dataset: VideoFolderDataset):
    """
    Given a VideoFolderDataset, calculate:
    - Number of videos
    - Average video duration in seconds
    """
    num_videos = len(dataset)
    total_duration_seconds = 0

    for _, _, video_path in dataset:
        video = cv2.VideoCapture(video_path)
        
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        total_duration_seconds += duration

    return num_videos, total_duration_seconds


def calculate_gpu_memory_usage() -> Tuple[float, float, float]:
    MB_CONVERSION_FACTOR = 1024 ** 2
    allocated_memory = torch.cuda.memory_allocated() / MB_CONVERSION_FACTOR
    return allocated_memory

def load_model(model_flag: str, checkpoint_path: Path = None, device: str = "cpu"):
    """
    Load either pretrained or finetuned model.

    Available model flags:
    - timesformer: facebook/timesformer-base-finetuned-k400
    """


    if model_flag == "timesformer":
        model_name = "facebook/timesformer-base-finetuned-k400"
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)


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