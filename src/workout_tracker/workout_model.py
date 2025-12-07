import random
import time
import threading
from queue import Queue
from pathlib import Path
import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
import logging

import torch

import constants as c
from recorder import Recorder
from base.utils import load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(
    level=logging.INFO, 
    format='[workout_model.py] %(message)s'
)
logger = logging.getLogger()


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

    def __init__(self, output: str = None, alpha: float = 0.):

        self._last_prediction_timestamp = 0
        self._raw_last_prediction_probs = dict()
        self._smoothed_last_prediction_probs = dict()
        self._prediction_lock = threading.Lock()
        self._label_count_lock = threading.Lock()
        self._alpha = alpha
        self._recorder = Recorder(output)


    def predict(self, *args, **kwargs):
        """
        Non blocking prediction. Starts a daemon thread to perform the prediction.
        """
        prediction_thread = threading.Thread(target=self._predict_with_timeout, args=args, kwargs=kwargs)
        prediction_thread.daemon = True
        prediction_thread.start()

    def _predict_thread(self, frames, prediction_result_queue):
        raise NotImplementedError("Subclasses must implement this method")

    def _predict_with_timeout(self, frames):
        """
        Uses a thread to perform the prediction. Timesout after self.timeout seconds.

        frames: list of frames (np.ndarray), representing a short video clip.
        Return a label string.
        """
        prediction_start = time.time()
        
        prediction_result_queue = Queue() # Queue where the thread should store the result
        prediction_thread = threading.Thread(target=self._predict_thread, args=(frames, prediction_result_queue))
        
        prediction_thread.start()
        prediction_thread.join(timeout=self.timeout)

        if prediction_thread.is_alive():
            logger.error(f"Thread timed out")
            return
    
        probs = prediction_result_queue.get()
        prediction_label = max(probs, key=probs.get)
        prediction_prob = probs[prediction_label]
        
        if prediction_prob < c.CONFIDENCE_THRESHOLD:
            prediction_label = "pause"
            prediction_prob = 1 - max(probs.values())
        

        self.increment_label_count(prediction_label)
        updated = self.update_probs(prediction_start, probs)

        if not updated:
            logger.warning(f"Thread was too slow and failed to update last prediction")
            return

        logger.debug(f"Predicted label: {prediction_label}")


        prediction_start_str = time.strftime('%H:%M:%S.%s', time.localtime())
        probs["timestamp"] = prediction_start_str
        self._recorder.record(probs)
        
        return prediction_label
    
    def increment_label_count(self, label):
        """ 
        Thread safe increment of the label count.
        """
        with self._label_count_lock:
            c.LABEL_TO_COUNT[label] += c.PREDICTION_INTERVAL
        
    def update_probs(self, starting_time: float, probs: dict):
        """
        Thread safe update of the last prediction.
        """

        with self._prediction_lock:
            if starting_time < self._last_prediction_timestamp:
                return False
            
            self._last_prediction_timestamp = starting_time
            self._raw_last_prediction_probs = probs

            default_prob = 1 / len(c.LABEL_TO_COUNT.keys())
            self._smoothed_last_prediction_probs = {
                label: self._alpha * self._smoothed_last_prediction_probs.get(label, default_prob) + (1 - self._alpha) * prob \
                    for label, prob in probs.items()
            }
            return True
        
    def get_last_prediction(self) -> tuple[str, float]:
        probs = self._smoothed_last_prediction_probs
        if len(probs) == 0:
            return (None, None)
        prob = max(probs.values())
        label = max(probs, key=probs.get)
        if prob < c.CONFIDENCE_THRESHOLD:
            return ("pause", 1 - prob)
        return label, prob

class WorkoutModel(WorkoutBaseModel):
    """
    Model used to predict the current workout.
    """

    def __init__(
            self, 
            model_flag,
            device=None,  #default: None (for local)
            ckpt_path=None,
            timeout=c.INFERENCE_TIMEOUT,
            num_frames=c.NUM_FRAMES,
            temperature=c.TEMPERATURE,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        elif isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
            # if cuda not available -> automatically use cpu
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

        self.device = device
        self.timeout = timeout
        self.num_frames = num_frames
        self.temperature = temperature
        self.processor, self.model, self.classes = load_model(
            model_flag,
            checkpoint_path=ckpt_path,
            device=self.device,
        )

        self.model.to(self.device)
        self.model.eval()

    def _predict_thread(self, frames, prediction_result_queue):
        frames = sample_frames(frames, num_frames=self.num_frames)
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            logits = self.model(pixel_values=pixel_values).logits / self.temperature
        

        tracked_probs = {}
        for label in c.LABEL_TO_COUNT.keys():
            if label == "pause":
                continue

            model_label = c.INVERSE_TIMESFORMER_LABEL_NORMALIZATION.get(label)
            label_id = self.model.config.label2id.get(model_label, None)

            if label_id is not None:
                tracked_probs[label] = logits[0, label_id].item()

            else:
                tracked_probs[label] = -float("inf")
        
        # Get items and sort alphabetically by label
        tracked_items = sorted(tracked_probs.items(), key=lambda x: x[0])
        
        labels = [label for label, _ in tracked_items]
        probs = [prob for _, prob in tracked_items]

        softmax_probs = torch.softmax(torch.tensor(probs), dim=-1)
        probs = {label: softmax_probs[idx].item() for idx, label in enumerate(labels)}

        prediction_result_queue.put(probs)