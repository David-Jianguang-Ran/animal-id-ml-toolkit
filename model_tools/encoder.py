"""

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from uuid import uuid4
from pathlib import Path

from tensorflow.keras.callbacks import Callback
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from keras.callbacks import Callback

from scipy.spatial import KDTree

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, Concatenate, Add
from tensorflow.keras.models import Model, load_model


# # inception module
def add_inception_module(input_node, channel_multiplier: int = 16, layer_name_prefix=None, activation="tanh"):
    """
    adds a inception like block on top of input_node,
    returns output node
    :param input_node: keras.Layer or tensor
    :param channel_multiplier: int the number of filters to use at each step
    :param layer_name_prefix: str
    :param activation: str name of activation function
    :return:
    """
    one_by_one = Conv2D(4 * channel_multiplier, (1, 1), activation=activation, name=layer_name_prefix + "_1x1")(input_node)

    a = Conv2D(4 * channel_multiplier, (1, 1), activation=activation, name=layer_name_prefix + "_pre_3x3")(input_node)
    three_by_three = Conv2D(8 * channel_multiplier, (3, 3), padding="same", activation=activation, name=layer_name_prefix + "_3x3")(a)

    b = Conv2D(1 * channel_multiplier, (1, 1), activation=activation, name=layer_name_prefix + "_pre_5x5")(input_node)
    five_by_five = Conv2D(2 * channel_multiplier, (5, 5), padding="same", activation=activation, name=layer_name_prefix + "_5x5")(b)

    c = MaxPool2D((3, 3), padding="same", strides=1, name=layer_name_prefix + "_pooling")(input_node)
    pooled = Conv2D(2 * channel_multiplier, (1, 1), activation=activation, name=layer_name_prefix + "_pooled")(c)

    concat_output = Concatenate(axis=3,name=layer_name_prefix + "_concat")([one_by_one, three_by_three, five_by_five, pooled])

    return concat_output


def build_xception_encoder(image_shape=(320, 320, 3), embedding_length=16):

    encoder_input = Input(image_shape)

    # we are using xception as our base encoder
    # Xception output shape ?, 10, 10, 2048
    pre_trained = Xception(include_top=False, weights="imagenet", input_tensor=encoder_input, input_shape=image_shape)
    pre_trained.trainable = False

    # after xception we are adding conv layers
    # we are using 1x1 convolutions to reduce volume to flat vector
    # using 1x1 convolutions because it is easier to ensure output shape with varying input
    first = add_inception_module(pre_trained.output,channel_multiplier=32, layer_name_prefix="encoder_1", activation="tanh")
    second = Conv2D(256, (3, 3), activation="selu", name="encoder_2")(first)
    third = add_inception_module(second,channel_multiplier=16, layer_name_prefix="encoder_3", activation="tanh")
    forth = Conv2D(128, (3, 3), activation="selu", name="encoder_4")(third)
    fifth = add_inception_module(forth,channel_multiplier=8, layer_name_prefix="encoder_5", activation="tanh")
    sixth = Conv2D(64, (2, 2), activation="selu", name="encoder_6")(fifth)
    seventh = Conv2D(embedding_length, (3, 3), activation="selu", name="encoder_7")(sixth)

    # output layer
    embedding = Flatten()(seventh)

    # wrap the whole thing in a keras.Model instance
    encoder_model = Model(inputs=[encoder_input], outputs=[embedding])
    return encoder_model


# # evaluation metric
def _prox_hits(labels, embeddings, prox_dist: float, k: int):
    # note: this function is likely to error out due to recursion limits

    # first the embeddings needs to be stored in a kdtree
    tree = KDTree(embeddings)

    # query tree for approx. k nearest neighbor for embedding vector
    # note that the query will return inf distance and out of bounds index for not found rows
    dist, index = tree.query(embeddings, k=k, eps=1.0, distance_upper_bound=prox_dist)

    # turn the index of knn to labels
    # note since identity always comes up as nearest neighbor, first element of index is excluded
    padded_labels = np.concatenate([labels,np.array([0.0,])])
    nearest_labels = padded_labels[index[:, 1:]]

    # check if label of each row exists in nearest_labels
    # TODO re-implement in numpy operations for performance gain
    hit = 0
    count = 0
    for row_label, neighbor_labels in zip(labels, nearest_labels):
        if row_label in neighbor_labels:
            hit += 1
        count += 1

    return hit / count


def proximity_hits(prox_dist: float = 1.0, knn: int = 6, output_average: bool = True):
    """
    computes for each embedding,
    whether knn includes another embedding from the same label
    outputs average hits from all rows
    :param prox_dist: float,  maximum distance to include for each embedding
    :param knn: int, how many nearest neighbors to find
    :param output_average: bool, whether output is a ndarray same with shape(batch, 1) or single float
    """
    def _inner(labels, embeddings):
        hits = _prox_hits(labels, embeddings, prox_dist, knn)
        if output_average:
            return np.average(hits)
        else:
            return hits

    return _inner


class ProximityHitsCallback(Callback):
    """I really shouldn't use this because this is sooooo slow"""
    def __init__(self, test_data, test_label, prox_dist=1.5, k=3, *args, **kwargs):
        super().__init__()

        self.test_data = test_data
        self.test_label = test_label

        self.prox_dist = prox_dist
        self.k = k

        self.history = []

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        embeddings = self.model.predict(self.test_data)

        metric = np.average(_prox_hits(self.test_label, embeddings, self.prox_dist, self.k))
        self.history.append(metric)

    def on_epoch_end(self, epoch, logs=None):

        embeddings = self.model.predict(self.test_data)

        metric = np.average(_prox_hits(self.test_label, embeddings, self.prox_dist, self.k))
        self.history.append(metric)


# # embeddings visualizer
class EmbeddingVisualizer(Callback):
    """
    This class makes embeddings with test data,
    plots a 3d PCA projection of the embeddings and
    shows it and saves it to save_to path
    """
    def __init__(self, test_data, test_labels, save_to=None, model_id=None, highlight_one=False, two_d_plot=False, *args, **kwargs):
        super().__init__()
        self.model_id = str(uuid4())[:8] if model_id is None else model_id
        self.highlight_one = highlight_one
        self.two_d = two_d_plot

        self.test_data = test_data
        self.test_labels = test_labels

        self.history = {}
        self.frames = []

        # create save_to if doesnt exist
        if isinstance(save_to, str):
            Path(f"{save_to}/{self.model_id}").mkdir(parents=True, exist_ok=True)
            self.output_path = f"{save_to}/{self.model_id}"
        else:
            self.output_path = None

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        self.plot_embeddings(self.get_test_embeddings(), "i", 0.0)

    def on_epoch_end(self, epoch, logs=None):
        self.plot_embeddings(self.get_test_embeddings(), epoch, logs['loss'])

    def get_test_embeddings(self):
        return self.model.predict(self.test_data)

    def on_batch_end(self, batch, logs=None):
        pass

    def plot_embeddings(self, embeddings, num, loss):
        """
        first we run the test data through the model
        and save a pca projection of the resulting embeddings
        """

        # get 3d PCA projection
        pca = PCA(n_components=3)
        embeddings = pca.fit_transform(embeddings)

        # make color labels
        if self.highlight_one:
            color = np.equal(self.test_labels, self.test_labels[0]).astype("uint8")
        else:
            color = self.test_labels

        # make scatter plot
        fig = plt.figure()
        if self.two_d:
            plt.scatter(embeddings[:,0], embeddings[:,1], c=color)
        else:
            ax = Axes3D(fig)
            ax.scatter(embeddings[:,0], embeddings[:,1], embeddings[:,2], c=color)
        id_string = f"{self.model_id}_e{num}_l{loss:.3}"
        fig.suptitle(id_string)

        # show image
        try:
            fig.show()
        except:
            pass

        # save image
        if self.output_path:
            fig.savefig(f"{self.output_path}/pca_{num}")

        # close figure
        plt.close()
