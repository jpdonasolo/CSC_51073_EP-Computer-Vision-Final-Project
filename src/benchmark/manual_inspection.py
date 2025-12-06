import sys
from pathlib import Path
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src/workout_tracker"))

from workout_tracker.workout_model import WorkoutModel
import workout_tracker.constants as c



VIDEO_ID = 5


def main():
    model = WorkoutModel(
            "timesformer",
            output=PROJECT_ROOT / "src" / "benchmark" / f"manual_inspection_{VIDEO_ID}.csv",
        )
    
    # Load video from "src/benchmark/manual_inspection"
    video_path = PROJECT_ROOT / "src" / "benchmark" / "manual_inspection_videos" / f"{VIDEO_ID}.mp4"
    
    # Option 1: Load all frames as numpy arrays
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    print(f"Loaded {len(frames)} frames")
    print(f"First frame shape: {frames[0].shape}")  # (H, W, 3)


    fps = 30
    cur_frame = int(fps * c.WARMUP_TIME)

    while cur_frame < len(frames) - c.BUFFER_SIZE:
        model.predict(frames[cur_frame:cur_frame + c.BUFFER_SIZE])
        cur_frame += int(fps * c.PREDICTION_INTERVAL)
        time.sleep(c.PREDICTION_INTERVAL)

    print("Done")



if __name__ == "__main__":
    main()