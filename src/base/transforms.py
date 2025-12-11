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

    def __init__(self, num_frames: int, 
                 flip_prob: float = 0.5, 
                 color_jitter_prob: bool = True, 
                 use_augmentation: bool = True,
                 brightness: float = 0.1,  # approximately ±10% variation
                 contrast: float = 0.1):
        
        self.num_frames = num_frames
        self.flip_prob = flip_prob
        self.use_augmentation = use_augmentation
        self.color_jitter_prob = color_jitter_prob
        self.brightness = brightness
        self.contrast = contrast
        
    
    def _color_jitter_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        frame: H x W x C, uint8 想定
        brightness と contrast を少しだけ変える
        """
        # float
        img = frame.astype(np.float32)

        # brightness:
        if self.brightness > 0:
            b_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        else:
            b_factor = 1.0

        # contrast:
        if self.contrast > 0:
            c_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        else:
            c_factor = 1.0

        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * c_factor + mean
        img = img * b_factor

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    
    def __call__(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        # frames: list of HxWxC numpy arrays

        # ---- no augmentation mode ----
        if not self.use_augmentation:
            return frames

        # ---- horizontal flip ----
        if random.random() < self.flip_prob:
            frames = [np.fliplr(f) for f in frames]
            
        # ---- slight color jitter ----
        if self.color_jitter_prob > 0 and random.random() < self.color_jitter_prob:
            frames = [self._color_jitter_frame(f) for f in frames]

        return frames