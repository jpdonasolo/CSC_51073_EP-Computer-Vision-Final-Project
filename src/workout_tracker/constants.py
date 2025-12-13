LABEL_TO_COUNT = {
    "push-up": 0,
    "russian-twist": 0,
    "squat": 0,
    "plank": 0,
    "pause": 0,
}

PRETRAINED_TIMESFORMER_LABEL_NORMALIZATION = {
    "russian-twist": "russian-twist",
    "push up": "push-up",
    "squat": "squat",
    "plank": "plank"
}

FINETUNED_TIMESFORMER_LABEL_NORMALIZATION = {
    "russian-twist": "russian-twist",
    "push-up": "push-up",
    "squat": "squat",
    "plank": "plank"
}

INVERSE_PRETRAINED_TIMESFORMER_LABEL_NORMALIZATION = {v: k for k, v in PRETRAINED_TIMESFORMER_LABEL_NORMALIZATION.items()}
INVERSE_FINETUNED_TIMESFORMER_LABEL_NORMALIZATION = {v: k for k, v in FINETUNED_TIMESFORMER_LABEL_NORMALIZATION.items()}

# Minimum probability for a prediction to be made
CONFIDENCE_THRESHOLD = 0.9

# Timeout for the inference thread
INFERENCE_TIMEOUT = .5

# Number of frames to sample from the video
NUM_FRAMES = 8

# Buffer size for the frame buffer 
BUFFER_SIZE_SECONDS = 3

# Time interval between predictions (seconds)
PREDICTION_INTERVAL = 0.2

# Time before first prediction (seconds)
WARMUP_TIME = 2.0

# Temeprature lower than one to counterweight the fact that
# "pause" is the most likely label.
TEMPERATURE = 0.9