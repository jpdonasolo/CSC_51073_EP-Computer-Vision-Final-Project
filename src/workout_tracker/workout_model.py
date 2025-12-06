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

    def __init__(self, output: str = None):

        self._last_prediction_timestamp = 0
        self._last_prediction_label = None
        self._prediction_lock = threading.Lock()
        # Lock to protect the c.LABEL_TO_COUNT dictionary
        self._label_count_lock = threading.Lock()

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
    
        (prediction_label, probs) = prediction_result_queue.get()

        prediction_start_str = time.strftime('%H:%M:%S.%s', time.localtime())
        probs["timestamp"] = prediction_start_str
        self._recorder.record(probs)

        self.increment_label_count(prediction_label)
        updated = self.update_last_prediction(prediction_start, prediction_label)
        
        if not updated:
            logger.warning(f"Thread was too slow andfailed to update last prediction")
            return
        
        logger.info(f"Predicted label: {prediction_label}")
        
        return prediction_label
    
    def increment_label_count(self, label):
        """ 
        Thread safe increment of the label count.
        """
        with self._label_count_lock:
            c.LABEL_TO_COUNT[label] += c.PREDICTION_INTERVAL
        
    def update_last_prediction(self, starting_time: float, label: str):
        """
        Thread safe update of the last prediction.
        """

        with self._prediction_lock:
            if starting_time < self._last_prediction_timestamp:
                return False
            
            self._last_prediction_timestamp = starting_time
            self._last_prediction_label = label
            return True
        
    def get_last_prediction(self) -> str:
        return self._last_prediction_label
    
class WorkoutDummyModel(WorkoutBaseModel):
    """
    Dummy model used for testing.

    In the future, replace this with your fine-tuned model class.
    Implement a similar interface:

        model = YourRealModel(...)
        label = model.predict(frames)

    where `frames` is a list of numpy arrays (H, W, 3).
    """

    def __init__(self, labels, timeout=c.INFERENCE_TIMEOUT, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels
        self.timeout = timeout

    def _predict_thread(self, frames, prediction_result_queue):
        # For now, sleeps to simulate inference time, then returns a random label.  
        inference_time = random.uniform(0, 1)
        print(f"Inference time: {inference_time:.2f} seconds")
        time.sleep(inference_time)

        if not frames:
            c.LABEL_TO_COUNT["pause"] += c.PREDICTION_INTERVAL
            return

        predicted_label = random.choice(self.labels)
        prediction_result_queue.put((predicted_label, {label: 1/len(self.labels) for label in self.labels}))


class WorkoutModel(WorkoutBaseModel):
    """
    Model used to predict the current workout.
    """

    def __init__(
            self, 
            model_flag,
            labels, 
            device=None,  #default: None (for local)
            timeout=c.INFERENCE_TIMEOUT,
            num_frames=c.NUM_FRAMES,
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
        self.labels = labels
        self.timeout = timeout
        self.device = device
        self.num_frames = num_frames
        
        ckpt_path = PROJECT_ROOT / "checkpoints" / "timesformer_best.pt"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found at {ckpt_path}. Falling back to pretrained model.")
            ckpt_path = None
            
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
            logits = self.model(pixel_values=pixel_values).logits / c.TEMPERATURE
            probs = logits.softmax(dim=-1)[0]
            top_idx = probs.argmax().item()

        tracked_probs = {}
        for label in sorted(c.TIMESFORMER_LABEL_NORMALIZATION.keys()):
            
            label_id = self.model.config.label2id.get(label, None)
            if label_id is None:
                tracked_probs[label] = 0
                continue
            
            prob = probs[label_id].item()

            label = c.TIMESFORMER_LABEL_NORMALIZATION.get(label, "pause")
            tracked_probs[label] = prob
        
        # Add the probability of the pause
        tracked_probs["pause"] = (probs.sum().item()) - sum(tracked_probs.values())

        
        pred_label = self.model.config.id2label[top_idx]
        predicted_label = c.TIMESFORMER_LABEL_NORMALIZATION.get(pred_label, "pause")
        
        prediction_result_queue.put((predicted_label, tracked_probs))