"""
Create the finetuning dataset. The folder structure will be:
```
finetuning/
├── train/
│   ├── label_1
│   ├── label_2/
│   └── ...
└── val/
    ├── label_1
    ├── label_2/
    └── ...
```

Videos names are normalized using unique IDs within each label.

Usage:
```
uv run create_finetuning_folders.py [--kaggle-dir <path/to/kaggle/dir> --local-datasets-dir <path/to/local/datasets/dir> --outdir <path/to/out/dir> --train-size train_size --clear-outdir]
```
"""

import os
import shutil
import argparse
from pathlib import Path

import numpy as np


RANDOM_SEED = 1
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


LABELS = ["push-up", "pull-up", "plank", "squat", "russian-twist"]
DEFAULT_LABEL_TO_DIR = {x: x for x in LABELS}


def parse_args():
    parser = argparse.ArgumentParser()
    default_kaggle_dir = os.path.expanduser("~/.cache/kagglehub/datasets/")
    default_local_datasets_dir = os.path.join(ROOT_DIR, "datasets")
    default_outdir = os.path.join(ROOT_DIR, "finetuning")
    
    parser.add_argument("--kaggle-dir", type=str, default=default_kaggle_dir)
    parser.add_argument("--local-datasets-dir", type=str, default=default_local_datasets_dir)
    parser.add_argument("--outdir", type=str, default=default_outdir)
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--clear-outdir", action="store_false")
    args = parser.parse_args()
    
    assert 0 <= args.train_size <= 1, "Train size must be between 0 and 1"
    assert os.path.exists(args.kaggle_dir)
    assert os.path.exists(args.local_datasets_dir)

    return args


def create_output_dirs(
    outdir: str,
    labels: list[str] = LABELS
):
    """Create directory for the finetuning dataset and subdirectories for each label.
    """
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "train"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "val"), exist_ok=True)

    for label in labels:
        os.makedirs(os.path.join(outdir, "train", label), exist_ok=True)
        os.makedirs(os.path.join(outdir, "val", label), exist_ok=True)

def split_videos(dataset_folder_path: str, train_size: float):
    """ Split videos in train and val.

    Args:
        dataset_folder_path: Path to the folder with the videos files.
        train_size: Size of the train split.

    Returns:
        train_videos: List of train videos filenames.
        val_videos: List of val videos filenames.
    """
    videos = os.listdir(dataset_folder_path)
    n_videos = len(videos)
    n_train = int(n_videos * train_size)
    
    video_indices = np.random.permutation(list(range(n_videos)))
    train_indices = video_indices[:n_train]
    val_indices = video_indices[n_train:]

    train_videos = [videos[idx] for idx in train_indices]
    val_videos = [videos[idx] for idx in val_indices]

    return train_videos, val_videos


class VideoCopier:
    """Tracks video counts per label (across all splits), and copies videos with unique names."""
    
    def __init__(self):
        # Dictionary to track counts per label: {label: count}
        # Videos with the same label share a unique sequence regardless of split
        self.counters = {}
    
    def copy_video_with_unique_name(
        self,
        src_path: str,
        label: str,
        split: str,
        outdir: Path,
    ) -> str:
        """
        Copy a video with a unique name based on label.
        """
        # Initialize counter for this label if needed
        if label not in self.counters:
            self.counters[label] = 0
        
        # Increment counter and generate unique filename
        self.counters[label] += 1
        video_id = self.counters[label]
        
        # Get original file extension
        original_filename = os.path.basename(src_path)
        _, ext = os.path.splitext(original_filename)
        
        # Generate new filename: label_video_id.ext
        new_filename = f"{label}_{video_id}{ext}"
        
        # Destination path
        dst_path = os.path.join(outdir, split, label, new_filename)
        
        # Copy the file
        shutil.copy(src_path, dst_path)
        
        return new_filename

def main(
    kaggle_dir: str,
    local_datasets_dir: str,
    outdir: str,
    train_size: float,
    clear_outdir: bool,
):
    np.random.seed(RANDOM_SEED)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if clear_outdir:
        print(f"Clearing outdir: {outdir}")
        for file in os.listdir(outdir):
            shutil.rmtree(os.path.join(outdir, file), ignore_errors=True)

    create_output_dirs(outdir)

    # Initialize video copier to track unique video IDs
    video_copier = VideoCopier()

    workoutfitness_dir = os.path.join(kaggle_dir, "hasyimabdillah/workoutfitness-video/versions/5")
    assert os.path.exists(workoutfitness_dir)
    print(f"Processing Kaggle dataset: workoutfitness-video")
    
    label_to_dir = {
        "pull-up": "pull Up",
        "russian-twist": "russian twist",
    }
    # Use default label to dir for the rest of the labels
    label_to_dir = {k: label_to_dir.get(k, v) for k, v in DEFAULT_LABEL_TO_DIR.items()}
    assert set(label_to_dir.keys()) == set(LABELS)
    
    for label, dataset_folder_name in label_to_dir.items():
        
        # Create path to dataset folder
        dataset_folder_path = os.path.join(workoutfitness_dir, dataset_folder_name)
        
        # Split videos in train and val
        train_videos, val_videos = split_videos(dataset_folder_path, train_size)

        # Copy train and val videos to outdir with unique names
        for video in train_videos:
            src_path = os.path.join(dataset_folder_path, video)
            video_copier.copy_video_with_unique_name(src_path, label, "train", outdir)
        for video in val_videos:
            src_path = os.path.join(dataset_folder_path, video)
            video_copier.copy_video_with_unique_name(src_path, label, "val", outdir)

    
    dataset_bao2_dir = os.path.join(kaggle_dir, "bluebirdss/dataset-bao2/versions/1/data_mae")
    assert os.path.exists(dataset_bao2_dir)
    print(f"Processing Kaggle dataset: dataset-bao2")
    
    label_to_dir = {
        "pull-up": "pull up",
        "russian-twist": "russian twist",
    }
    label_to_dir = {k: label_to_dir.get(k, v) for k, v in DEFAULT_LABEL_TO_DIR.items()}
    assert set(label_to_dir.keys()) == set(LABELS)

    # Hardcoded train/val/test counts for dataset-bao2
    bao2_counts = {
        "plank": (16, 5, 1),
        "push-up": (44, 13, 2),
        "pull-up": (12, 5, 0),
        "squat": (40, 12, 1),
        "russian-twist": (14, 5, 0),
    }
    assert set(bao2_counts.keys()) == set(label_to_dir.keys())
    
    train_root_dir = os.path.join(dataset_bao2_dir, "train")
    val_root_dir = os.path.join(dataset_bao2_dir, "val")
    test_root_dir = os.path.join(dataset_bao2_dir, "test")
    
    for label, dataset_folder_name in label_to_dir.items():
            
        train_count, val_count, test_count = bao2_counts[label]
        total_count = train_count + val_count + test_count
        desired_train_count = int(total_count * train_size)
        
        train_label_dir = os.path.join(train_root_dir, dataset_folder_name)
        val_label_dir = os.path.join(val_root_dir, dataset_folder_name)
        test_label_dir = os.path.join(test_root_dir, dataset_folder_name)
        

        train_videos = []
        val_videos = []
        test_videos = []
        # Get all videos from each split
        if os.path.exists(train_label_dir):
            train_videos = [(f, train_label_dir) for f in os.listdir(train_label_dir) 
                        if os.path.isfile(os.path.join(train_label_dir, f))]
        if os.path.exists(val_label_dir):
            val_videos = [(f, val_label_dir) for f in os.listdir(val_label_dir) 
                         if os.path.isfile(os.path.join(val_label_dir, f))]
        if os.path.exists(test_label_dir):
            test_videos = [(f, test_label_dir) for f in os.listdir(test_label_dir) 
                          if os.path.isfile(os.path.join(test_label_dir, f))]
        
        # Adjust train count if needed (only move from train to test, never the opposite)
        if len(train_videos) > desired_train_count:
            np.random.shuffle(train_videos)
            videos_to_move = train_videos[desired_train_count:]
            train_videos = train_videos[:desired_train_count]
            test_videos.extend(videos_to_move)
        
        # Copy videos to outdir with unique names
        for video, src_dir in train_videos:
            src_path = os.path.join(src_dir, video)
            video_copier.copy_video_with_unique_name(src_path, label, "train", outdir)
        for video, src_dir in val_videos:
            src_path = os.path.join(src_dir, video)
            video_copier.copy_video_with_unique_name(src_path, label, "val", outdir)
        for video, src_dir in test_videos:
            src_path = os.path.join(src_dir, video)
            video_copier.copy_video_with_unique_name(src_path, label, "val", outdir)

if __name__ == "__main__":
    main(**parse_args().__dict__)