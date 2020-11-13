"""
here are some functions for
* manual data cleaning
* crop to subject with object detector network
* data cleaning by clustering image embeddings
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from uuid import uuid4
from sklearn.cluster import DBSCAN

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.xception import preprocess_input

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


def _group_input(n, separator=","):
    id_string = input(f"input image numbers of {n}th animal:")

    # handle special cases
    if len(id_string) == 0:
        return []
    elif id_string[:4] == "back":
        raise RestartInput

    try:
        ids = [int(each) for each in id_string.split(separator)]
        print(f"ids received {ids}")
    except ValueError:
        return []

    return ids


def _get_identity_input(image_count):
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


def _batched_operations_by_id(decoratee):
    """
    TODO: These batch operations functions have weird function signitures, fix it or it will be too confusing soon!
    this decorator takes a batch wise operation and run it ove each set grouped by user_id
    *_note:_* DON"T SAVE OUTPUT AS IS. image data stored as a single row, indexed by image_id, just like input!
    """
    def _inner(labels: pd.DataFrame, images: pd.DataFrame, input_shape=(240,240,3), *args, **kwargs):
        """
        *_This function has been wrapped in a decorator_*
        the inner function will be called for every unique user_id in input labels
        the results will be collected and concat into a new pair of image and label data frames
        :param labels: labels dataframe containing column animal_id and image_id
        :param images: image dataframe with each row being a flattened image indexed by image_id
        :param input_shape: shape of the each image before flattening, 3tuple
        :param args: ...
        :param kwargs: any kwargs you'd like passed to inner
        :return: new_image: pd.Dataframe, new_labels: pd.Dataframe
        *_original doc string below_*
        ______________
        """
        # first we aggregate images from each insta id together by dropping id suffix
        labels["user_id"] = np.floor(labels["animal_id"].values)
        unique_ids = labels['user_id'].unique()

        out_images = []
        out_image_ids = []
        out_animal_ids = []

        # iterate through images grouped by user_id
        for i, user_id in enumerate(unique_ids):
            # select all images with same animal_id
            selected_images = labels[labels["user_id"] == user_id]["image_id"]
            selected_images = images.loc[selected_images].values.reshape((-1, *input_shape))

            print(f"batch{i}/{len(unique_ids) - 1}, id {user_id}")
            found_images, found_image_ids, found_animal_ids = decoratee(user_id, selected_images, *args, **kwargs)

            out_images += found_images
            out_image_ids += found_image_ids
            out_animal_ids += found_animal_ids

        if len(out_image_ids) == 0:
            print("WARNING: 0 images has been found. Something isn't right!")
            return pd.DataFrame(), pd.DataFrame(columns=["image_id", "animal_id"])

        # combine data into dataframes
        out_images = pd.DataFrame(np.concatenate(out_images, axis=0))
        out_images['image_id'] = out_image_ids
        out_images.set_index(["image_id"], inplace=True)

        out_labels = pd.DataFrame({"image_id": out_image_ids, "animal_id": out_animal_ids})

        return out_images, out_labels

    # setting doc string on decoratee for eaiser usage
    decoratee.__doc__ = _inner.__doc__ + str(decoratee.__doc__)
    return _inner


@_batched_operations_by_id
def manual_identify(shared_id, batch_images):
    """
    *_This function has been wrapped in a decorator_*
    see source code of decorator for usage
    this function processes a single batch grouped by user_id
    """
    # plot out images for manual inspection
    grid_size = int(np.floor(np.sqrt(batch_images.shape[0]))) + 1
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(6,6))  # <= note that axs is a ndarray shape grid,grid with subplots inside

    fig.suptitle(f"found {batch_images.shape[0]} images for id {shared_id}")
    print(f"identifying for id {shared_id}")

    out_images = []
    out_image_ids = []
    out_animal_ids = []
    # plot each image and i in subplot
    for i in range(batch_images.shape[0]):
        axs[i // grid_size, i % grid_size].imshow(batch_images[i])
        axs[i // grid_size, i % grid_size].set_title(f"{i}")

    plt.show()
    plt.close()

    # receive input from human operator
    human_input = []
    while True:
        try:
            human_input = _get_identity_input(batch_images.shape[0])
            break
        except RestartInput:
            continue

    # make new ids,  mark unselected as 0.
    new_animal_ids = [0.0 for i in range(batch_images.shape[0])]
    for i, selected_indexes in enumerate(human_input):
        for each_index in selected_indexes:
            new_animal_ids[each_index] = _animal_id_suffixed(shared_id, i)

    # if image is selected, add to output collection
    for i, each in enumerate(new_animal_ids):
        if each == 0.0:
            pass
        else:
            out_images.append(batch_images[i].reshape((1, -1)))
            out_image_ids.append(_new_short_id())
            out_animal_ids.append(each)

    return out_images,  out_image_ids, out_animal_ids,


@_batched_operations_by_id
def detect_and_crop(shared_id, batch_images, output_shape=(180, 180, 3)):
    """
    *_This function has been wrapped in a decorator_*
    see source code of decorator for usage
    This function runs YOLO over image and outputs the found images with new shape, with labels
    """
    detector = YOLO()

    # run images through object detector
    found_images, found_labels, found_scores = detector.predict_n_crop(batch_images, output_shape=output_shape)

    if len(found_labels) == 0:
        return [], [], []
    else:
        # reshape images into dataset row and collect,
        # make image ids and animal ids
        return [found_images.reshape((len(found_labels), -1))], \
            [_new_short_id() for each in found_labels] , \
            [float(np.floor(shared_id) + ID_SUFFIX[each]) for each in found_labels]


@_batched_operations_by_id
def clean_with_encoder(shared_id, batch_images, encoder: Model = None):
    """
    *_This function has been wrapped in a decorator_*
    see source code of decorator for usage
    This function runs encoder over image and outputs images with outlier removed, with labels
    """
    if encoder is None:
        raise ValueError("A vector encoder is required for this operation")

    # input images needs to be preprocessed
    # obtain batch embeddings
    embeddings = encoder.predict_on_batch(preprocess_input(batch_images))

    # do clustering on embeddings
    possible_identities = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMP).fit_predict(embeddings)

    out_images = []
    out_image_ids = []
    out_animal_ids = []

    # if image is selected, add to output collection
    for i, learned_label in enumerate(possible_identities):
        if learned_label == -1:  # <= learned label -1 means noise or outlier
            pass
        else:
            out_images.append(batch_images[i].reshape((1, -1)))
            out_image_ids.append(_new_short_id())
            out_animal_ids.append(_animal_id_suffixed(shared_id, learned_label))

    return out_images,  out_image_ids, out_animal_ids,
