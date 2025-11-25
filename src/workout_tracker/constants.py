LABEL_TO_COUNT = {
    "push-up": 0,
    "pull-up": 0,
    "squat": 0,
    "plank": 0,
    "pause": 0,
}

TIMESFORMER_LABEL_NORMALIZATION = {
    "pull ups": "pull-up",
    "push up": "push-up",
    "squat": "squat",
    "plank": "plank"
}

# Timeout for the inference thread
INFERENCE_TIMEOUT = .8

# Number of frames to sample from the video
NUM_FRAMES = 8

# Time interval between predictions (seconds)
PREDICTION_INTERVAL = 0.5

# Time before first prediction (seconds)
WARMUP_TIME = 2.0