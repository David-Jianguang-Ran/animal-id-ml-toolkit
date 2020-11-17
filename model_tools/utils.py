import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, load_model

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from .generators import SequentialGenerator
from .encoder import proximity_hits

# TODO : there is a lot of redundant code here. refactor!


def compute_embeddings(data: SequentialGenerator, model: Model, include_labels=True):
    # note this is meant to be used with a keras.Sequence
    embeddings = []
    labels = []
    # predict for each batch, save labels
    for batch_image, batch_labels in data:
        embeddings.append(model.predict_on_batch(batch_image))

        if include_labels:
            labels.append(batch_labels)
    # save to a data frame, return
    frame = pd.DataFrame(np.concatenate(embeddings, axis=0))
    if include_labels:
        frame['image_id'] = data._labels['image_id'].values
        frame.set_index(["image_id"], inplace=True)
    return frame


def plot_embeddings(encodings,labels,key_str):
    pca = PCA(n_components=3)

    pri_comp = pca.fit_transform(encodings)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(pri_comp[:,0],pri_comp[:,1],pri_comp[:,2], c=np.equal(labels, labels[0]).astype("int8"))
    fig.suptitle(key_str)
    fig.show()
    plt.close()


def inspect_embeddings(model, generator, batches, note=""):
    labels, embeddings = get_predictions(model, generator, batches)
    print(embeddings[:5,:])
    plot_embeddings(embeddings, labels, f"count:{len(labels)} - {note}")


def get_predictions(model, generator, batches: int = None):
    """
    helper function for running
    :param model: Keras.Model
    :param generator: Keras.Sequence, outputing data, label each batch
    :param batches: int # of batches to predict, can't be more than generator batches
    :return: true, pred
    """
    encoding_out = []
    label_out = []
    for num in range(len(generator) if batches is None else min(batches, len(generator))):
        x, y = generator.__getitem__(num)
        pred = model.predict_on_batch(x)
        encoding_out.append(pred)
        label_out.append(y)

    encoding_out = np.concatenate(encoding_out,axis=0)
    label_out = np.concatenate(label_out,axis=0)

    return label_out, encoding_out


def run_proximity_hits(model, data_gen, prox=4.0, k=6):
    results = []
    metric = proximity_hits(prox, k)
    for x, y in data_gen:
        em = model.predict_on_batch(x)
        results.append(metric(y, em))

    return np.average(results)


def _normalize_meanstd(input_array, axis=(1, 2)):
    """
    normalize input value to std_dev to batch mean
    :param input_array: ndarray shape (batch_size , height, width, channels),
    :param axis: tuple, axis for normalization, default to height width axis of image
    :return:
    """
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(input_array, axis=axis, keepdims=True)
    std = np.sqrt(((input_array - mean) ** 2).mean(axis=axis, keepdims=True))
    return (input_array - mean) / std

