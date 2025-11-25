import logging
import sys
from pathlib import Path
import cv2

# User imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src/"))
from base.utils import VideoFolderDataset

# Logging
logging.basicConfig(
    level=logging.INFO, 
    format='[benchmark/utils.py] %(message)s'
)
logger = logging.getLogger()


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
