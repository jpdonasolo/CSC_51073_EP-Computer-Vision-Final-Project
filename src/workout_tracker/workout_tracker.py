import cv2
import time
from collections import deque
import argparse

from workout_model import WorkoutModel
import constants as c


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model", type=str, default="finetuned")
    return parser.parse_args()

def format_time(seconds):
    """Format seconds as mm:ss for display."""
    seconds = int(seconds)
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"


def main(output: str = None):
    # Initialize camera (on macOS, AVFOUNDATION is usually the stable backend)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set display size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Buffer to store recent frames (last BUFFER_SIZE_SECONDS seconds)
    # Assume max 60fps for buffer
    frame_buffer = deque(maxlen=c.BUFFER_SIZE_SECONDS * 60)
    
    # Strat FPS at 30fps
    fps = 30
    
    # Smooth FPS over time
    alpha = 0.98

    # Initialize dummy model
    model = WorkoutModel(
        model_flag=model,
        output=output,
        alpha=0.8
    )

    # Current label and inference timing
    last_infer_time = time.time()
    warmup_time = time.time()
    last_frame_cap = 0

    current_label = "initializing model..."
    current_prob = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Warning: Failed to read frame from camera.")
                break

            now = time.time()
            fps = alpha * fps + (1 - alpha) * 1 / (now - last_frame_cap)
            last_frame_cap = now

            # Store frame in buffer (later used for model inference)
            frame_buffer.append(frame.copy())

            # If enough time has passed, run inference on the recent clip
            if now - warmup_time >= c.WARMUP_TIME and now - last_infer_time >= c.PREDICTION_INTERVAL:
                last_infer_time = now

                n_trames = int(c.BUFFER_SIZE_SECONDS * fps)
                frames_for_model = list(frame_buffer)[-n_trames:]

                # Call dummy model (replace with real model later)
                model.predict(frames_for_model)
            # ===== Rendering overlay text =====

            if (last_prediction:=model.get_last_prediction()) != (None, None):
                current_label, current_prob = last_prediction

            # Show current label at the top
            cv2.putText(
                frame,
                f"Current: {current_label} ({current_prob:.2%})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Show fps at the right corner
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                .7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Show cumulative time for each label on the left side
            y0 = 60
            dy = 25
            for i, label in enumerate(c.LABEL_TO_COUNT.keys()):
                t_str = format_time(c.LABEL_TO_COUNT[label])
                text = f"{label}: {t_str}"
                cv2.putText(
                    frame,
                    text,
                    (10, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Display the frame
            cv2.imshow("Workout Tracker", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Pressed 'q'. Exiting.")
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args.output, args.model)
