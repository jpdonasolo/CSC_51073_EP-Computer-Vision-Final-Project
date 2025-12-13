# CSC_51073_EP Final Project

If the cloning the repository takes too long, try downloading only the latest version with

`git clone --depth 1 https://github.com/jpdonasolo/CSC_51073_EP-Computer-Vision-Final-Project.git`

This will ignore heavy videos that were uploaded to the repository in previous versions.

## Running the code
This project uses `uv` as a package manager. You can download it [here](https://docs.astral.sh/uv/guides/install-python/).

Then, just run from the root directory:

`uv run src/workout_tracker/workout_tracker.py`

By default, the code will download and run the finetuned model. To use the pretrained model, just add `--model pretrained`.

## Finetuning
If you intend to run the finetuning code, make sure to follow this steps:

### rar
Some of the datasets are downloaded as `.rar` files. To `unrar` them, make sure to download the [executable](https://www.win-rar.com/download.html?&L=0) and unzip it in the root folder.

### Kagglehub
The project also downloads datasets from [Kaggle](https://www.kaggle.com/) using `kagglehub`. In order to work properly, your machine must have an autentication API token. You can read about it [here](https://www.kaggle.com/docs/api).

## Running the code
Before running the code, your root folder should look like this:
