import unittest
import random

import pandas as pd
import numpy as np

from unittest.mock import MagicMock

from data_tools.processing import _batched_operations_by_id, _new_short_id
from data_tools.processing import *
from data_tools.utils import load_and_combine_dataset


def get_dummy_data(row_count, row_width=27):
    """this function returns dummy images, label data in pd.DataFrame"""
    images = pd.DataFrame([ [ float(random.randint(0,10)) for j in range(row_width)] for i in range(row_count)])
    ids = [chr(65 + i // 32) + chr(65 + i % 32) for i in range(row_count)]
    images["image_id"] = ids
    images.set_index(["image_id"],inplace=True)

    labels = pd.DataFrame({
        "animal_id" : [ float(i // 4) for i in range(row_count)],
        "image_id" : ids
    })

    return images, labels


class MockModel:
    def predict_on_batch(self, x):
        """this returns something like a encoding, batch of vectors 9 wide"""
        return x.reshape((x.shape[0],-1))[:,:27:3]


class MockDataTest(unittest.TestCase):

    def setUp(self) -> None:
        self.image_data, self.label_data = get_dummy_data(50)
        self.encoder = MockModel()

    def test_batched_decorator(self):

        # get a mock inner method
        def mock_inner(labels, images):
            self.assertTrue(len(images.shape) == 4)

            return [images.reshape((-1,27))], [_new_short_id() for i in range(images.shape[0])], [labels for i in range(images.shape[0])]

        # decorate
        decorated = _batched_operations_by_id(mock_inner)

        # call with a dataframe
        new_images, new_labels = decorated(self.label_data, self.image_data, input_shape=(3,3,3))

        # test data concat
        self.assertTrue(self.label_data['animal_id'].all() == new_labels['animal_id'].all())
        self.assertTrue(self.image_data.shape == new_images.shape)

    def test_clean_with_encoder(self):
        # run images through encoder
        new_image, new_label = clean_with_encoder(self.label_data, self.image_data, input_shape=(3,3,3), encoder=self.encoder)

        self.assertTrue(new_label.shape[0] <= self.label_data.shape[0])

