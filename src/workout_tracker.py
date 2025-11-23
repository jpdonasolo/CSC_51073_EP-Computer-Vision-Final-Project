import cv2
import time
import random
from collections import deque
import threading

# Candidate labels
LABEL_TO_COUNT = {
    "push-up": 0,
    "pull-up": 0,
    "squat": 0,
    "plank": 0,
    "pause": 0,
}
TIMEOUT = .8 # seconds


class DummyWorkoutModel:
    """
    Dummy model used for testing.

    In the future, replace this with your fine-tuned model class.
    Implement a similar interface:

        model = YourRealModel(...)
        label = model.predict(frames)

    where `frames` is a list of numpy arrays (H, W, 3).
    """

    def __init__(self, labels, timeout=TIMEOUT):
        self.labels = labels
        self.timeout = timeout

    def predict(self, *args, **kwargs):
        """
        Non blocking prediction. Starts a daemon thread to perform the prediction.
        """
        t = threading.Thread(target=self._predict_with_timeout, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()

    def _predict_with_timeout(self, frames):
        """
        Uses a thread to perform the prediction. Timesout after self.timeout seconds.

        frames: list of frames (np.ndarray), representing a short video clip.
        Return a label string.
        """
        def predict_thread(frames):
            # For now, sleeps to simulate inference time, then returns a random label.  
            inference_time = random.uniform(0, 1)
            print(f"Inference time: {inference_time:.2f} seconds")
            time.sleep(inference_time)

            if not frames:
                LABEL_TO_COUNT["pause"] += 1
                return

            predicted_label = random.choice(self.labels)
            LABEL_TO_COUNT[predicted_label] += 1

        t = threading.Thread(target=predict_thread, args=(frames,))
        t.start()
        t.join(timeout=self.timeout)

        if t.is_alive():
            print("Thread timed out")

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

    # Current label and inference timing
    current_label = "pause"
    last_infer_time = time.time()
    infer_interval = 5.0  # Run inference every 5 seconds

    # Initialize dummy model
    model = DummyWorkoutModel(list(LABEL_TO_COUNT.keys()), timeout=TIMEOUT)

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
                last_infer_time = now
                frames_for_model = list(frame_buffer)

                # Call dummy model (replace with real model later)
                model.predict(frames_for_model)
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
            for i, label in enumerate(LABEL_TO_COUNT.keys()):
                t_str = format_time(LABEL_TO_COUNT[label])
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
