import random
import time
import threading
from pathlib import Path
import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

import constants as c
from benchmark.utils import load_model


def sample_frames(frames, num_frames=8):
    """Sample evenly-spaced frames from a list of frames."""
    if len(frames) <= num_frames:
        return frames
    indices = torch.linspace(0, len(frames) - 1, steps=num_frames).long().tolist()
    return [frames[int(i)] for i in indices]


class WorkoutBaseModel:
    """
    Base class for all workout models.
    """
    def predict(self, *args, **kwargs):
        """
        Non blocking prediction. Starts a daemon thread to perform the prediction.
        """
        t = threading.Thread(target=self._predict_with_timeout, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()

        def _predict_thread(frames):
            raise NotImplementedError("Subclasses must implement this method")

    def _predict_with_timeout(self, frames):
        """
        Uses a thread to perform the prediction. Timesout after self.timeout seconds.

        frames: list of frames (np.ndarray), representing a short video clip.
        Return a label string.
        """
        t = threading.Thread(target=self._predict_thread, args=(frames,))
        t.start()
        t.join(timeout=self.timeout)

        if t.is_alive():
            print("Thread timed out")
        

class WorkoutModel(WorkoutBaseModel):
    """
    Dummy model used for testing.

    In the future, replace this with your fine-tuned model class.
    Implement a similar interface:

        model = YourRealModel(...)
        label = model.predict(frames)

    where `frames` is a list of numpy arrays (H, W, 3).
    """

    def __init__(self, labels, timeout=c.INFERENCE_TIMEOUT):
        self.labels = labels
        self.timeout = timeout

    def _predict_thread(self, frames):
        # For now, sleeps to simulate inference time, then returns a random label.  
        inference_time = random.uniform(0, 1)
        print(f"Inference time: {inference_time:.2f} seconds")
        time.sleep(inference_time)

        if not frames:
            c.LABEL_TO_COUNT["pause"] += 1
            return

        predicted_label = random.choice(self.labels)
        c.LABEL_TO_COUNT[predicted_label] += 1


class WorkoutModel(WorkoutBaseModel):
    """
    Model used to predict the current workout.
    """

    def __init__(
            self, 
            model_flag,
            labels, 
            device="cuda", 
            timeout=c.INFERENCE_TIMEOUT,
            num_frames=c.NUM_FRAMES
        ):
        self.labels = labels
        self.timeout = timeout
        self.device = device
        self.num_frames = num_frames
        self.processor, self.model, self.classes = load_model(model_flag, device=device)

        self.model.to(device)
        self.model.eval()

    def _predict_thread(self, frames):
        frames = sample_frames(frames, num_frames=self.num_frames)
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            logits = self.model(pixel_values=pixel_values).logits
            probs = logits.softmax(dim=-1)[0]
            top_idx = probs.argmax().item()
        
        pred_label = self.model.config.id2label[top_idx]
        predicted_label = c.TIMESFORMER_LABEL_NORMALIZATION.get(pred_label, "pause")
        
        c.LABEL_TO_COUNT[predicted_label] += 1