import cv2
import time
import random
from collections import deque

# Candidate labels
LABELS = ["push-up", "pull-up", "squat", "plank", "pause"]


class DummyWorkoutModel:
    """
    Dummy model used for testing.

    In the future, replace this with your fine-tuned model class.
    Implement a similar interface:

        model = YourRealModel(...)
        label = model.predict(frames)

    where `frames` is a list of numpy arrays (H, W, 3).
    """

    def __init__(self, labels):
        self.labels = labels

    def predict(self, frames):
        """
        frames: list of frames (np.ndarray), representing a short video clip.
        Return a label string.
        """
        if not frames:
            return "pause"

        # For now, randomly choose a label.
        # Replace this logic with actual model inference.
        return random.choice(self.labels)


def format_time(seconds):
    """Format seconds as mm:ss for display."""
    seconds = int(seconds)
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"


def main():
    # Initialize camera (on macOS, AVFOUNDATION is usually the stable backend)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set display size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Buffer to store recent frames (last 5 seconds)
    # Assuming ~30 fps, 5 seconds ≈ 150 frames
    frame_buffer = deque(maxlen=150)

    # Cumulative time (in seconds) for each label
    cumulative_times = {label: 0.0 for label in LABELS}

    # Current label and inference timing
    current_label = "pause"
    last_infer_time = time.time()
    infer_interval = 5.0  # Run inference every 5 seconds

    # Initialize dummy model
    model = DummyWorkoutModel(LABELS)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to read frame from camera.")
                break

            now = time.time()

            # Store frame in buffer (later used for model inference)
            frame_buffer.append(frame.copy())

            # If enough time has passed, run inference on the recent clip
            if now - last_infer_time >= infer_interval:
                frames_for_model = list(frame_buffer)

                # Call dummy model (replace with real model later)
                predicted_label = model.predict(frames_for_model)

                current_label = predicted_label

                # Add the last interval duration to the predicted label's cumulative time
                cumulative_times[predicted_label] += now - last_infer_time

                last_infer_time = now
                # Optionally, you can clear the buffer here if you want non-overlapping clips:
                # frame_buffer.clear()

            # ===== Rendering overlay text =====

            # Show current label at the top
            cv2.putText(
                frame,
                f"Current: {current_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Show cumulative time for each label on the left side
            y0 = 60
            dy = 25
            for i, label in enumerate(LABELS):
                t_str = format_time(cumulative_times[label])
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
    main()
