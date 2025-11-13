# CSC_51073_EP Final Project

## Instalation

### UV
This project uses `uv` as a package manager. You can download it [here](https://docs.astral.sh/uv/guides/install-python/).

### rar
Some of the datasets are downloaded as `.rar` files. To `unrar` them, make sure to download the [executable](https://www.win-rar.com/download.html?&L=0) and unzip it in the root folder.

### Kagglehub
The project also downloads datasets from [Kaggle](https://www.kaggle.com/) using `kagglehub`. In order to work properly, your machine must have an autentication API token. You can read about it [here](https://www.kaggle.com/docs/api).

## Running the code
Before running the code, your root folder should look like this:

```
.
├── download_datasets.py
├── pyproject.toml
├── rar
│   ├── rar
│   ├── unrar
│   └── ...
├── README.md
└── uv.lock
```

Then, you can rode the code with `uv run download_datasets.py`