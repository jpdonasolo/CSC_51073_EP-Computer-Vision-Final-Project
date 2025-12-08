# src/base/transforms.py
import random
import numpy as np
import torch
from typing import List, Tuple

class VideoRandomAugment:
    """
    - holizontal flip (p: flip_prob)
    - Random temporal crop (align to num_frames)
    """

    def __init__(self, num_frames: int, flip_prob: float = 0.5, temporal_jitter: bool = True):
        self.num_frames = num_frames
        self.flip_prob = flip_prob
        self.temporal_jitter = temporal_jitter

    def __call__(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        # frames: list of HxWxC numpy arrays

        # ---- temporal crop ----
        if self.temporal_jitter and len(frames) > self.num_frames:
            max_start = len(frames) - self.num_frames
            start = random.randint(0, max_start)
            frames = frames[start:start + self.num_frames]

        # if length is not enough, repeat the final frame
        if len(frames) < self.num_frames and len(frames) > 0:
            last = frames[-1]
            frames = list(frames) + [last] * (self.num_frames - len(frames))

        # ---- horizontal flip ----
        if random.random() < self.flip_prob:
            frames = [np.fliplr(f) for f in frames]

        return frames