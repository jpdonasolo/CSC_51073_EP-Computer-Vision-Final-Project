import kagglehub
import requests
import os
import urllib3
import subprocess
import argparse


if not os.path.exists("rar/unrar"):
    raise FileNotFoundError("`unrar` executable not found. Download and unpack it from https://www.win-rar.com/download.html")


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abs-path-outdir", type=str, default=os.path.join(ROOT_DIR, "datasets"))
    parser.add_argument("--force-download", action="store_true")
    
    return parser.parse_args()


def main(
    abs_path_outdir: str,
    force_download: bool,
):
    os.makedirs(abs_path_outdir, exist_ok=True)

    # For requests download without SSL certificate verification
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video
    print("Downloading dataset: hasyimabdillah/workoutfitness-video")
    path = kagglehub.dataset_download("hasyimabdillah/workoutfitness-video")

    # https://www.kaggle.com/datasets/mohamadashrafsalama/pushup
    print("Downloading dataset: mohamadashrafsalama/pushup")
    path = kagglehub.dataset_download("mohamadashrafsalama/pushup")


    # https://www.crcv.ucf.edu/data/UCF101.php
    print("Downloading dataset: UCF101")
    url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    local_path = os.path.join(abs_path_outdir, "UCF101.rar")
    
    if not os.path.exists(local_path) or force_download:
        with requests.get(url, verify=False, stream=True) as response:
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0)) / 1024 / 1024
            downloaded = 0

            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk) / 1024 / 1024
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}% ({downloaded:.1f}/{total_size:.1f} MB)", end='', flush=True)
    
    else:
        print("Dataset UCF101 already exists. Skipping download")

    print("Extracting dataset: UCF101")
    if os.path.exists(os.path.join(abs_path_outdir, "UCF-101")):
        print("Dataset UCF101 already extracted. Skipping extraction")
        return
    subprocess.run([f"{ROOT_DIR}/rar/unrar",  "x", local_path, abs_path_outdir], check=True)

if __name__ == "__main__":
    args = parse_args()
    main(**args.__dict__)