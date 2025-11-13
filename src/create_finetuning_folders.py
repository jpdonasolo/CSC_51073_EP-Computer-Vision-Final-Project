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

Usage:
```
uv run create_finetuning_folders.py [--kaggle-dir <path/to/kaggle/dir> --local-datasets-dir <path/to/local/datasets/dir> --outdir <path/to/out/dir> --train-size train_size --clear-outdir]
```
"""

import os
import shutil
import argparse
from glob import glob


import numpy as np


RANDOM_SEED = 1
HERE = os.path.dirname(os.path.abspath(__file__))




def parse_args():
    parser = argparse.ArgumentParser()
    default_kaggle_dir = os.path.expanduser("~/.cache/kagglehub/datasets/")
    default_local_datasets_dir = os.path.join(HERE, "../datasets")
    default_outdir = os.path.join(HERE, "../finetuning")
    
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
    labels: list[str] = ["push-up", "pull-up", "plank", "squat"]
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

def main(
    kaggle_dir: str,
    local_datasets_dir: str,
    outdir: str,
    train_size: float,
    clear_outdir: bool,
):
    np.random.seed(RANDOM_SEED)


    if clear_outdir:
        print(f"Clearing outdir: {outdir}")
        for file in os.listdir(outdir):
            shutil.rmtree(os.path.join(outdir, file), ignore_errors=True)

    create_output_dirs(outdir)


    workoutfitness_dir = os.path.join(kaggle_dir, "hasyimabdillah/workoutfitness-video/versions/5")
    assert os.path.exists(workoutfitness_dir)
    print(f"Processing Kaggle dataset: workoutfitness-video")
    
    label_to_dir = {
        "push-up": "push-up",
        "pull-up": "pull Up",
        "plank": "plank",
        "squat": "squat",
    }
    
    for label, dataset_folder_name in label_to_dir.items():
        
        # Create path to dataset folder
        dataset_folder_path = os.path.join(workoutfitness_dir, dataset_folder_name)
        
        # Split videos in train and val
        train_videos, val_videos = split_videos(dataset_folder_path, train_size)

        # Copy train and val videos to outdir
        for video in train_videos:
            shutil.copy(os.path.join(dataset_folder_path, video), os.path.join(outdir, "train", label, video))
        for video in val_videos:
            shutil.copy(os.path.join(dataset_folder_path, video), os.path.join(outdir, "val", label, video))



if __name__ == "__main__":
    main(**parse_args().__dict__)