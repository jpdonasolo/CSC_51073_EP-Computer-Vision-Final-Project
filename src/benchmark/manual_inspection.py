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



VIDEO_ID = 6
VIDEO_TO_FPS = {
    1: 29.72,
    2: 29.94,
    3: 29.98,
    4: 29.88,
    5: 29.95,
    6: 18.56,
}


def main():
    model = WorkoutModel(
            "timesformer",
            # ckpt_path=PROJECT_ROOT / "checkpoints" / "timesformer_best.pt",
            output=PROJECT_ROOT / "src" / "benchmark" / "manual_inspection" / "results" / f"pretrained_{VIDEO_ID}.csv",
        )
    
    # Load video from "src/benchmark/manual_inspection"
    video_path = PROJECT_ROOT / "src" / "benchmark" / "manual_inspection" / "videos" / f"{VIDEO_ID}.mp4"
    
    # Option 1: Load all frames as numpy arrays
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # Resize frames to 640x480
    # frames = [cv2.resize(frame, (640, 480)) for frame in frames]

    print(f"Loaded {len(frames)} frames")
    print(f"First frame shape: {frames[0].shape}")  # (H, W, 3)


    camera_framerate = VIDEO_TO_FPS[VIDEO_ID]
    normalized_buffer_size = int(c.BUFFER_SIZE / 30 * camera_framerate)
    
    last_infer_time = time.time()
    cur_frame = c.WARMUP_TIME * camera_framerate


    while True:
        current_time = time.time()
        
        if current_time - last_infer_time >= c.PREDICTION_INTERVAL:
            elapsed_time = current_time - last_infer_time
            elapsed_frames = elapsed_time * camera_framerate
            cur_frame = cur_frame + elapsed_frames
            print(f"Starting inference at {cur_frame / camera_framerate:.2f} seconds")
            
            if cur_frame >= len(frames):
                break

            model.predict(frames[int(cur_frame-normalized_buffer_size):int(cur_frame)])
            last_infer_time = current_time
        
        else:
            time.sleep(0.01)

    # Wait for the model to finish predicting
    time.sleep(.5)
    print("Done")



if __name__ == "__main__":
    main()