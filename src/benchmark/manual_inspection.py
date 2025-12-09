import sys
from pathlib import Path
import time
import os

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src/workout_tracker"))

from workout_tracker.workout_model import WorkoutModel


BUFFER_SIZE_SECONDS = 5
PREDICTION_INTERVAL = 0.15
WARMUP_TIME = max(2.0, BUFFER_SIZE_SECONDS + 0.2)
CONFIDENCE_THRESHOLD = 0.95
ALPHA = 0.8
TIMEOUT = 0.8
NUM_FRAMES = 8
TEMPERATURE = 1

USE_FINETUNED_MODEL = False


def main():


    print("Loading model...")
    ckpt_path = PROJECT_ROOT / "checkpoints" / "timesformer_best.pt" if USE_FINETUNED_MODEL else None
    filename = "finetuned" if USE_FINETUNED_MODEL else "pretrained"
    model = WorkoutModel(
            "timesformer",
            ckpt_path=ckpt_path,
            temperature=TEMPERATURE,
            num_frames=NUM_FRAMES,
            timeout=TIMEOUT,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            alpha=ALPHA,
        )
    print(f"Loaded {filename} model")
    print(f"Parameters: {TEMPERATURE=}, {NUM_FRAMES=}, {TIMEOUT=}, {CONFIDENCE_THRESHOLD=}, {ALPHA=}")


    videos = os.listdir(PROJECT_ROOT / "src" / "benchmark" / "manual_inspection" / "videos")
    for video_path in videos:
        video_path = PROJECT_ROOT / "src" / "benchmark" / "manual_inspection" / "videos" / video_path
        print(f"Processing video {video_path}")
    
        model.set_output(PROJECT_ROOT / "src" / "benchmark" / "manual_inspection" / "results" / f"{filename}_{video_path.stem}.csv")

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
    main()