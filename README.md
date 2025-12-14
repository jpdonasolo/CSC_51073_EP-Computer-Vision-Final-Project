# CSC_51073_EP Final Project

If the cloning repository takes too long, try downloading only the latest version with

`git clone --depth 1 https://github.com/jpdonasolo/CSC_51073_EP-Computer-Vision-Final-Project.git`

This will ignore heavy videos that were uploaded in previous commits.

## Running the code
This project uses `uv` as a package manager. You can download it [here](https://docs.astral.sh/uv/guides/install-python/).

IF running from mac, please change the following line in `src/workout_trackerworkout_tracker.py` from

`cap = cv2.VideoCapture(0)`

to

`cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)`

Then, run from the root directory:
`uv run src/workout_tracker/workout_tracker.py`

By default, the code will download and run the finetuned model. To use the pretrained model, just add `--model pretrained`.

⚠️ WARNING ⚠️
Depending on your machine, the predictions might not run fast enough, making threads accumulate and GPU memory to explode. If threads start to timeout and the times on the screen stop increasing, try increasing `PREDICTION_INTERVAL` in `src/workout_tracker/constants.py`.

## Finetuning
If you intend to run the finetuning code, make sure to follow these steps:

### Kagglehub
Finetuning downloads datasets from [Kaggle](https://www.kaggle.com/) using `kagglehub`. In order to work properly, your machine must have an autentication API token. You can read about it [here](https://www.kaggle.com/docs/api).
