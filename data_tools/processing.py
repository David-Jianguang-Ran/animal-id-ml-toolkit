"""
here are some functions for
* data pre-processing
* pre-computing training data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from uuid import uuid4

from .object_detection import YOLO
from .settings import *


class RestartInput(Exception):
    pass


def _new_short_id():
    return str(uuid4())[-12:]


def _animal_id_suffixed(prev_id: float, num: int, addition_base=0.5) -> float:
    """adds a decimal number to make a new animal_id distinct from prev_id"""
    if num == 0:
        return prev_id
    else:
        return prev_id + addition_base ** num


def _group_input(n, seperator=","):
    id_string = input(f"input image numbers of {n}th animal:")

    # handle special cases
    if len(id_string) == 0:
        return []
    elif id_string[:4] == "back":
        raise RestartInput

    try:
        ids = [int(each)for each in id_string.split(seperator)]
        print(f"ids received {ids}")
    except ValueError:
        return []

    return ids


def get_identity_input(image_count):
    # receive human input
    i = 0
    grouped_by_identity = []
    while i < image_count:
        returned = _group_input(i)
        if len(returned) == 0:
            break
        elif any(returned) > image_count:
            print(f"input image id exceeds maximum, check input and re-enter")
        else:
            grouped_by_identity.append(returned)
            i += 1

    # ask user if they are really done
    done = input(f"done? typing n restarts input for this group again[y/n]")
    if done == "y":
        return grouped_by_identity
    elif done == "n":
        raise RestartInput
    elif done == "skip":
        return []
    else:
        print("that's a No i guess")
        raise RestartInput


def manual_identify(labels: pd.DataFrame, images: pd.DataFrame, input_shape=(240,240,3), ):
    # first we aggregate images from each insta id together by dropping id suffix
    labels["user_id"] = np.floor(labels["animal_id"].values)
    unique_ids = labels['user_id'].unique()

    out_images = []
    out_image_ids = []
    out_animal_ids = []

    # iterate through images grouped by user_id
    for user_id in unique_ids:
        # select all images with same animal_id
        selected_images = labels[labels["user_id"] == user_id]["image_id"]
        selected_images = images.loc[selected_images].values.reshape((-1,*input_shape))

        # plot out images for manual inspection
        grid_size = int(np.floor(np.sqrt(selected_images.shape[0]))) + 1
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(6,6))  # <= note that axs is a ndarray shape grid,grid with subplots inside

        fig.suptitle(f"found {selected_images.shape[0]} images for id {user_id}")
        print(f"identifying for id {user_id}")
        # plot each image and i in subplot
        for i in range(selected_images.shape[0]):
            axs[i // grid_size, i % grid_size].imshow(selected_images[i])
            axs[i // grid_size, i % grid_size].set_title(f"{i}")
        plt.show()
        plt.close()

        # receive input from human operator
        human_input = []
        while True:
            try:
                human_input = get_identity_input(selected_images.shape[0])
                break
            except RestartInput:
                continue

        # make new ids,  mark unselected as 0.
        new_animal_ids = [0.0 for i in range(selected_images.shape[0])]
        for i, selected_indexes in enumerate(human_input):
            for each_index in selected_indexes:
                new_animal_ids[each_index] = _animal_id_suffixed(user_id, i)

        # if image is selected, add to output collection
        for i, each in enumerate(new_animal_ids):
            if each == 0.0:
                pass
            else:
                out_images.append(selected_images[i].reshape((1, -1)))
                out_animal_ids.append(each)
                out_image_ids.append(_new_short_id())

    # will the same concat value error happen here?
    out_images = pd.DataFrame(np.concatenate(out_images, axis=0))
    out_images['image_id'] = out_image_ids
    out_images.set_index(["image_id"], inplace=True)

    out_labels = pd.DataFrame({"image_id": out_image_ids, "animal_id": out_animal_ids})

    return out_images, out_labels


def detect_and_crop(labels: pd.DataFrame, images: pd.DataFrame, input_shape=(240,240,3), output_shape=(180, 180, 3)):
    """
    This function takes data in the almost storage format and applys object detection and get a cropped result
    *_note:_* DON"T SAVE OUTPUT AS IS. image data stored as a single row, indexed by image_id, just like input!
    :param labels: dataframe, col=animal_id, image_id
    :param images: dataframe, index=image_id, row=image pixels flattened
    :param output_shape: desired output image shape
    :return: image, label dataframe same format as input
    """
    detector = YOLO()
    unique_ids = labels["animal_id"].unique()

    out_image_id = []
    out_animal_id = []
    out_images = []
    for each_id in unique_ids:
        print(f"running detection for userid {np.floor(each_id)}")
        # select all images with same animal_id
        selected_images = labels[labels["animal_id"] == each_id]["image_id"]
        selected_images = images.loc[selected_images].values.reshape((-1,*input_shape))

        # run images through object detector
        found_images, found_labels, found_scores = detector.predict_n_crop(selected_images, output_shape=output_shape)

        if len(found_labels) == 0:
            continue

        # make and collect a new set of image ids and animal ids
        out_image_id += [_new_short_id() for each in found_labels]
        out_animal_id += [float(np.floor(each_id) + ID_SUFFIX[each]) for each in found_labels]

        # reshape images into dataset row and collect
        out_images.append(found_images.reshape((len(found_labels), -1)))

    # combine data into dataframes
    out_images = pd.DataFrame(np.concatenate(out_images, axis=0))
    out_images['image_id'] = out_image_id
    out_images.set_index(["image_id"], inplace=True)

    out_labels = pd.DataFrame({"image_id": out_image_id, "animal_id": out_animal_id})

    return out_images, out_labels
