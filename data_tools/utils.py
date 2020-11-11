import pandas as pd
import numpy as np
import re


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
