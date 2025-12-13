import random
import time
import threading
from queue import Queue
from pathlib import Path
import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
import logging
import warnings
from typing import Literal
import subprocess
import shutil
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

    def __init__(self, output: str = None, alpha: float = 0., confidence_threshold: float = c.CONFIDENCE_THRESHOLD):

        self._last_prediction_timestamp = 0
        self._raw_last_prediction_probs = {"pause": 1.0}
        self._smoothed_last_prediction_probs = {"pause": 1.0}
        self._prediction_lock = threading.Lock()
        self._label_count_lock = threading.Lock()
        self._alpha = alpha
        self._confidence_threshold = confidence_threshold
        self._recorder = Recorder(output)

    def set_output(self, output: str):
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
        smoothed_probs = self._smooth_probs(probs)
        
        prediction_label = max(smoothed_probs, key=smoothed_probs.get)
        prediction_prob = smoothed_probs[prediction_label]

        prediction_label, prediction_prob = self.apply_confidence_threshold(prediction_label, prediction_prob)

        updated = self.update_probs(prediction_start, probs, smoothed_probs)
        self.increment_label_count(prediction_label)

        if not updated:
            logger.warning(f"Thread was too slow and failed to update last prediction")
            return

        logger.debug(f"Predicted label: {prediction_label}")


        ms = int((prediction_start % 1) * 1000)
        prediction_start_str = time.strftime('%H:%M:%S', time.localtime(prediction_start)) + f'.{ms:03d}'
        probs["timestamp"] = prediction_start_str
        self._recorder.record(probs)
        
        return prediction_label

    def _smooth_probs(self, probs: dict) -> dict:
        """
        Smooths the probabilities using the alpha parameter.
        """
        default_prob = 1 / len(c.LABEL_TO_COUNT.keys())
        smoothed_probs = {
            label: self._alpha * self._smoothed_last_prediction_probs.get(label, default_prob) + (1 - self._alpha) * prob \
                for label, prob in probs.items()
        }
        return smoothed_probs
    
    def apply_confidence_threshold(self, prediction_label: str, prediction_prob: float) -> tuple[str, float]:
        """
        Applies the confidence thresholds to the prediction. If the prediction is not confident enough,
        it is set to pause, and probability is set to 1 - prediction_prob.
        """
        probs = self._smoothed_last_prediction_probs
        current_label = max(probs, key=probs.get)

        if current_label != "pause" and prediction_prob < self._confidence_threshold:
            prediction_label = "pause"
            prediction_prob = 1 - prediction_prob

        return prediction_label, prediction_prob
    
    def increment_label_count(self, label):
        """ 
        Thread safe increment of the label count.
        """
        with self._label_count_lock:
            c.LABEL_TO_COUNT[label] += c.PREDICTION_INTERVAL
        
    def update_probs(self, starting_time: float, probs: dict, smoothed_probs: dict):
        """
        Thread safe update of the last prediction.
        """

        with self._prediction_lock:
            if starting_time < self._last_prediction_timestamp:
                return False
            
            self._last_prediction_timestamp = starting_time
            self._raw_last_prediction_probs = probs

            self._smoothed_last_prediction_probs = smoothed_probs

            return True
        
    def get_last_prediction(self) -> tuple[str, float]:
        probs = self._smoothed_last_prediction_probs
        if len(probs) == 0:
            return (None, None)
        prob = max(probs.values())
        label = max(probs, key=probs.get)
        if prob < self._confidence_threshold:
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
            pause_strategy: Literal["sum_untracked_heads", "confidence_threshold"] = "sum_untracked_heads",
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
        self.pause_strategy = pause_strategy

        if model_flag == "finetuned":
            ckpt_path = PROJECT_ROOT / "checkpoints" / "timesformer_max_full_even_6.pt"
            if not ckpt_path.exists():
                print("Downloading finetuned model...")
                
                
                temp_folder_name = Path("pretrained_model")
                shutil.rmtree(temp_folder_name, ignore_errors=True)
                
                command = ["git", "clone", "https://huggingface.co/MTomita/CSC_51073_EP-Computer-Vision-Final-Project", temp_folder_name]
                completed_process = subprocess.run(command, check=True)
                completed_process.check_returncode()
                
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(temp_folder_name / "timesformer_max_full_even_6.pt", ckpt_path)
                shutil.rmtree(temp_folder_name, ignore_errors=True)
        
        elif model_flag == "pretrained":
            ckpt_path = None

        self.processor, self.model, self.classes = load_model(
            "timesformer",
            checkpoint_path=ckpt_path,
            device=self.device,
        )

        self.model.to(self.device)
        self.model.eval()

        if ckpt_path is None:
            self.inverse_label_normalization = c.INVERSE_PRETRAINED_TIMESFORMER_LABEL_NORMALIZATION
        else:
            self.inverse_label_normalization = c.INVERSE_FINETUNED_TIMESFORMER_LABEL_NORMALIZATION

        # Make sure all labels are in the model's label2id
        untracked_labels = []
        for label in c.LABEL_TO_COUNT.keys():
            model_label = self.inverse_label_normalization.get(label)
            label_id = self.model.config.label2id.get(model_label, None)
            if label_id is None:
                untracked_labels.append(label)

        if len(untracked_labels) > 0:
            warnings.warn(f"The following labels were not found in model's label2id and will not be tracked: {untracked_labels}")

    def _predict_thread(self, frames, prediction_result_queue):
        frames = sample_frames(frames, num_frames=self.num_frames)
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            logits = self.model(pixel_values=pixel_values).logits / self.temperature


        tracked_logits = {}       
        if self.pause_strategy == "sum_untracked_heads":
            # Sum the logits of all labels that are not in LABEL_TO_COUNT.keys()
            # and use it as the pause probability
            untracked_labels = [label for label in self.model.config.label2id.keys() if label not in c.LABEL_TO_COUNT.keys()]
            untracked_label_ids = [self.model.config.label2id[label] for label in untracked_labels]

            if len(untracked_label_ids) < 0:
                warnings.warn(f"No untracked labels found. Setting pause logit to -inf")
                tracked_logits["pause"] = float("-inf")

            else:
                tracked_logits["pause"] = logits[0, untracked_label_ids].sum()
        
        else:
            # Otherwise, set it as -inf for now, as we will update it later
            tracked_logits["pause"] = float("-inf")

        for label in c.LABEL_TO_COUNT.keys():
            
            # Pause is not a label in any of our models
            if label == "pause":
                continue

            model_label = self.inverse_label_normalization.get(label)
            label_id = self.model.config.label2id.get(model_label, None)

            if label_id is not None:
                tracked_logits[label] = logits[0, label_id].item()

            else:
                tracked_logits[label] = -float("inf")
        
        # Get items and sort alphabetically by label
        tracked_items = sorted(tracked_logits.items(), key=lambda x: x[0])
        
        labels = [label for label, _ in tracked_items]
        probs = [prob for _, prob in tracked_items]

        softmax_probs = torch.softmax(torch.tensor(probs), dim=-1)
        probs = {label: softmax_probs[idx].item() for idx, label in enumerate(labels)}

        if self.pause_strategy == "confidence_threshold":
            # Set the pause probability to 1 - the probability of the most likely label
            probs["pause"] = 1 - max(probs.values())

        prediction_result_queue.put(probs)