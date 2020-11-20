import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import requests
import shutil
import time
import os


from uuid import uuid4
from pathlib import Path

from .settings import VERBOSITY, SAMPLE_LABELS_URL, SAMPLE_IMAGES_URL


def fancy_print(to_print, verbosity=0):
    if VERBOSITY >= verbosity:
        print(to_print)


def append_image_data_chunk(image_frame: pd.DataFrame, file_to_append: str):
    # this function load a parquet data chunk and 'add' it to image_frame
    to_add = pd.read_parquet(file_to_append, engine="pyarrow")
    return pd.concat([image_frame, to_add.T], axis=0)


def load_and_combine_dataset(dataset_path: str, nums: [int]):
    # i wanted to make sure the memory for each image data files are released after the combined dataframe is built
    images = pd.DataFrame()
    # just in case where base_path has /
    if dataset_path[-1] == "/":
        dataset_path = dataset_path[:-1]

    for num in nums:
        images = append_image_data_chunk(images, f"{dataset_path}/dataset_{num}.parquet")

    labels = pd.concat([pd.read_csv(f"{dataset_path}/dataset_labels_{each}.csv",index_col=0) for each in nums], axis=0, ignore_index=True)

    return images, labels


def string_contains(__str: str, __words: list):
    # catching empty captioned images here
    if not __str:
        return []

    # i think using the regex module is better than my own half assed implementation
    __output = []
    for each_word in __words:
        result = re.findall(each_word, __str)
        if result.__len__() != 0:
            __output.append(each_word)
    return __output


def plot_image_batch(shared_id=None,batch_images=None):
    # plot out images for manual inspection
    grid_size = int(np.floor(np.sqrt(batch_images.shape[0]))) + 1
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(6,6))  # <= note that axs is a ndarray shape grid,grid with subplots inside

    fig.suptitle(f"found {batch_images.shape[0]} images for id {shared_id}")
    print(f"identifying for id {shared_id}")

    # plot each image and i in subplot
    for i in range(batch_images.shape[0]):
        axs[i // grid_size, i % grid_size].imshow(batch_images[i])
        axs[i // grid_size, i % grid_size].set_title(f"{i}")

    plt.show()
    plt.close()
    time.sleep(0.5)  # <= this is here to stop image plot from being shown after text input


def download_sample_dataset(image_url=SAMPLE_IMAGES_URL, label_url=SAMPLE_LABELS_URL):
    # download and save to disk
    target_dir = "./data/sample"
    image_path = download_to_path(target_path=Path(target_dir + "/dataset_0.parquet"),target_url=image_url)
    label_path = download_to_path(target_path=Path(target_dir + "/dataset_labels_0.parquet"),target_url=label_url)
    # return path of dataset directory
    return target_dir


# getting files from hosted storage
def download_to_path(target_url, target_path, human_name="file"):
    if isinstance(target_path, str):
        target_path = Path(target_path)

    # ensure weights exist:
    if os.path.isfile(target_path):
        return fancy_print(f"existing file found at {target_path}, download skipped")
    else:
        fancy_print(f"downloading {human_name} from {target_url}")


    # ensure we have a dir to put downloaded file
    os.makedirs(target_path.parent, exist_ok=True)

    # do request
    with requests.get(target_url, stream=True) as r:
        with open(target_path, "wb") as file:
            shutil.copyfileobj(r.raw, file)

    fancy_print(f"saved {human_name} to {target_path}")
    return target_path


def new_short_id():
    return str(uuid4())[-12:]
