import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
import time
import pandas as pd

from random import randint

from data_tools.utils import load_and_combine_dataset
from data_tools.actions import download_dataset_from_usernames, download_username_by_hashtags
from model_tools.generators import RandomBlockGenerator, SequentialGenerator, PairedEmbeddingGenerator

from data_tools.processing import detect_and_crop, manual_identify, clean_with_encoder
from data_tools.object_detection import ImageNetClassifier, YOLO

from tensorflow.keras.models import load_model
from tensorflow_addons.losses import TripletSemiHardLoss

if __name__ == "__main__":
    # call imported code here    def test_batched_decorator(self):
    #
    #         # get a mock inner method
    #         def mock_inner(labels, images):
    #             self.assertTrue(len(images.shape) == 4)
    #
    #             return [images.reshape((-1,27))], [_new_short_id() for i in range(images.shape[0])], [labels for i in range(images.shape[0])]
    #
    #         # decorate
    #         decorated = _batched_operations_by_id(mock_inner)
    #
    #         # call with a dataframe
    #         old_images, old_labels = get_dummy_data(100)
    #
    #         new_images, new_labels = decorated(old_labels, old_images, input_shape=(3,3,3))
    #
    #         # test data concat
    #         self.assertTrue(old_labels['animal_id'].all() == new_labels['animal_id'].all())
    #         self.assertTrue(old_images.shape == new_images.shape)
    pass

    # load some data
    images, labels = load_and_combine_dataset("./data",[0])
    # load encoder
    encoder = load_model("./models/0ac7d8c9-ph0.654")

    # test cropper
    new_images, new_labels = clean_with_encoder(labels, images, input_shape=(240,240,3),encoder=encoder)

    print(new_labels.head())


