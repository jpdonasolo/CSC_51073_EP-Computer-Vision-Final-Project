import argparse
import sys
import time
import os
from typing import Literal
from pathlib import Path


import cv2
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src/workout_tracker"))


CHECKPOINTS_PATH = Path("/Data/masayo.tomita/CV/CSC_51073_EP-Computer-Vision-Final-Project/checkpoints/")

from workout_tracker.workout_model import WorkoutModel


BUFFER_SIZE_SECONDS = 5
PREDICTION_INTERVAL = 0.15
WARMUP_TIME = max(2.0, BUFFER_SIZE_SECONDS + 0.2)
CONFIDENCE_THRESHOLD = 0.95
ALPHA = 0.8
TIMEOUT = 0.8
NUM_FRAMES = 8
TEMPERATURE = .8

USE_FINETUNED_MODEL = True


def build_output_path(
    use_finetuned_model: bool, 
    ckpt_path: Path, 
    video_path: Path,
    temperature: float,
):
    base_path = PROJECT_ROOT / "src" / "benchmark" / "manual_inspection" / "results"
    
    if use_finetuned_model:
        stem = f"{ckpt_path.stem}_{video_path.stem}"
    else:
        stem = f"pretrained_{video_path.stem}"
    
    # Add temperature string to stem if temperature is not the module-level constant
    if temperature != 1.:
        stem += f"_temperature={temperature}"

    output_path = base_path / f"{stem}.csv"
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance-mode", type=str, default="none")
    parser.add_argument("--frame-sampling", type=str, default="even")
    parser.add_argument("--num-unfrozen-layers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    return parser.parse_args()


def main(
    balance_mode: Literal["none", "min", "2min_flip", "max_full"],
    frame_sampling: Literal["even", "first", "random", "center"],
    num_unfrozen_layers: int = Literal[2, 4, 6],
    temperature: float = TEMPERATURE,
):


    print("Loading model...")
    ckpt_path = CHECKPOINTS_PATH / f"timesformer_{balance_mode}_{frame_sampling}_{num_unfrozen_layers}.pt" if USE_FINETUNED_MODEL else None
    model = WorkoutModel(
            "timesformer",
            ckpt_path=ckpt_path,
            temperature=temperature,
            num_frames=NUM_FRAMES,
            timeout=TIMEOUT,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            alpha=ALPHA,
        )
    print(f"Loaded {ckpt_path} model")
    print(f"Parameters: {temperature=}, {NUM_FRAMES=}, {TIMEOUT=}, {CONFIDENCE_THRESHOLD=}, {ALPHA=}")


    videos = os.listdir(PROJECT_ROOT / "src" / "benchmark" / "manual_inspection" / "videos")
    for video_path in videos:
        video_path = PROJECT_ROOT / "src" / "benchmark" / "manual_inspection" / "videos" / video_path
        print(f"Processing video {video_path}")
    

        out_path = build_output_path(USE_FINETUNED_MODEL, ckpt_path, video_path, temperature)
        
        model.set_output(out_path)

        cap = cv2.VideoCapture(str(video_path))
        
        
        camera_framerate = cap.get(cv2.CAP_PROP_FPS)
        buffer_size = int(BUFFER_SIZE_SECONDS * camera_framerate)

        print(f"Camera framerate: {camera_framerate:.2f}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        print(f"Video duration: {len(frames) / camera_framerate:.2f} seconds")
        
        last_infer_time = time.time()
        cur_frame = WARMUP_TIME * camera_framerate


        while True:
            current_time = time.time()
            
            if current_time - last_infer_time >= PREDICTION_INTERVAL:
                elapsed_time = current_time - last_infer_time
                elapsed_frames = elapsed_time * camera_framerate
                cur_frame = cur_frame + elapsed_frames
                print(f"Starting inference at {cur_frame / camera_framerate:.2f} seconds")
                
                if cur_frame >= len(frames):
                    break

                model.predict(frames[int(cur_frame-buffer_size):int(cur_frame)])
                last_infer_time = current_time
            
            else:
                time.sleep(0.01)

        # Wait for the model to finish predicting
        time.sleep(.5)

if __name__ == "__main__":
    args = parse_args()
    main(**args.__dict__)