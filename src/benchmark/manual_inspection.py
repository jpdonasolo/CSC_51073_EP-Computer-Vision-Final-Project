import sys
from pathlib import Path
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src/workout_tracker"))

from workout_tracker.workout_model import WorkoutModel

VIDEO_TO_FPS = {
    1: 29.72,
    2: 29.94,
    3: 29.98,
    4: 29.88,
    5: 29.95,
    6: 18.56,
}

BUFFER_SIZE_SECONDS = 3
PREDICTION_INTERVAL = 0.2
WARMUP_TIME = 2.0
UP_CONFIDENCE_THRESHOLD = 0.95
ALPHA = 0.8
TIMEOUT = 0.5
NUM_FRAMES = 8
TEMPERATURE = 0.9


def main():


    for video_id in range(1, 7):
        print(f"Processing video {video_id}")

        model = WorkoutModel(
                "timesformer",
                ckpt_path=PROJECT_ROOT / "checkpoints" / "timesformer_best.pt",
                output=PROJECT_ROOT / "src" / "benchmark" / "manual_inspection" / "results" / f"finetuned_{video_id}.csv",
                temperature=TEMPERATURE,
                num_frames=NUM_FRAMES,
                timeout=TIMEOUT,
                up_confidence=UP_CONFIDENCE_THRESHOLD,
                alpha=ALPHA,
            )
    
        # Load video from "src/benchmark/manual_inspection"
        video_path = PROJECT_ROOT / "src" / "benchmark" / "manual_inspection" / "videos" / f"{video_id}.mp4"
        
        # Option 1: Load all frames as numpy arrays
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        camera_framerate = VIDEO_TO_FPS[video_id]
        buffer_size = int(BUFFER_SIZE_SECONDS * camera_framerate)
        
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