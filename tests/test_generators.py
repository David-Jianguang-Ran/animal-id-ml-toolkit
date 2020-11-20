import unittest
import random

import pandas as pd
import numpy as np


from model_tools.generators import PairedEmbeddingGenerator, \
    SequentialGenerator, RandomBlockGenerator, PairedImageGenerator


def get_dummy_data(row_count, row_width=27):
    """this function returns dummy images, label data in pd.DataFrame"""

    images = pd.DataFrame([ [ random.randint(0,255) for j in range(row_width)] for i in range(row_count)])
    ids = [chr(65 + i // 32) + chr(65 + i % 32) for i in range(row_count)]
    images["image_id"] = ids
    images.set_index(["image_id"],inplace=True)

    labels = pd.DataFrame({
        "animal_id" : [ float(i // 4) for i in range(row_count)],
        "image_id" : ids
    })

    return images, labels


# monkey patch to stop loading data from disk
def no_load_peg(self,batch_size=64,data_rows=128, row_width=27, image_shape=(3,3,3)):
    if batch_size:
        self.batch_size = batch_size

    if image_shape:
        self.image_shape = image_shape

    # load data
    self._data, self._labels = get_dummy_data(data_rows, row_width)


PairedEmbeddingGenerator.__init__ = no_load_peg


class TestEmbeddingGen(unittest.TestCase):

    def setUp(self) -> None:
        self.batch_size = 256
        self.row_width = 2048
        # the class init function signiture doesn't match here becuase it's been monkey patched :P
        self.instance = PairedEmbeddingGenerator(
            batch_size=self.batch_size,
            data_rows=1067,  # <= not a even number here because i wanted one odd batch
            row_width=self.row_width
        )

    def test_random_same_id(self):
        key = "animal_id"
        shuffled = self.instance.random_but_same_id(self.instance._labels[:self.batch_size], key=key)

        # iterate through both frames and make sure keys match
        for left, right in zip(self.instance._labels[:self.batch_size][key].values, shuffled[key].values):
            self.assertTrue(left == right)

    def test_get_item(self):

        # get a random batch
        batch_num = random.randint(0,self.instance.__len__() - 1)
        (left, right), sameness = self.instance.__getitem__(batch_num)

        # check shape of both embeddings
        self.assertTrue(left.shape[0] <= self.batch_size)
        self.assertTrue(right.shape[0] <= self.batch_size)

        self.assertTrue(left.shape[1] == self.row_width)
        self.assertTrue(right.shape[1] == self.row_width)

        # check the distribution of sameness
        # as long has classes are not super imbalanced im happy
        print(f"average sameness in paired embedding = {np.average(sameness):.3}, batch_size = {sameness.shape[0]}")
        self.assertTrue(0.4 <= np.average(sameness) <= 0.6)


PairedImageGenerator.__init__ = no_load_peg


class TestImageGen(unittest.TestCase):

    def setUp(self) -> None:
        self.batch_size = 256
        self.image_shape = (3,3,3)
        # the class init function signiture doesn't match here becuase it's been monkey patched :P
        self.instance = PairedImageGenerator(
            batch_size=self.batch_size,
            data_rows=1067,  # <= not a even number here because i wanted one odd batch
            row_width=27
        )

    def test_random_same_id(self):
        key = "animal_id"
        shuffled = self.instance.random_but_same_id(self.instance._labels[:self.batch_size], key=key)

        # iterate through both frames and make sure keys match
        for left, right in zip(self.instance._labels[:self.batch_size][key].values, shuffled[key].values):
            self.assertTrue(left == right)

    def test_get_item(self):
        # get a random batch
        batch_num = random.randint(0, self.instance.__len__() - 1)
        (left, right), sameness = self.instance.__getitem__(batch_num)

        # check shape of both images
        self.assertTrue(left.shape[0] <= self.batch_size)
        self.assertTrue(right.shape[0] <= self.batch_size)

        self.assertTrue(left.shape[1:] == self.image_shape)
        self.assertTrue(right.shape[1:] == self.image_shape)

        # check if image pixel values has been normalized
        self.assertTrue(np.amax(left) <= 1.0)
        self.assertTrue(np.amin(left) >= -1.0)
        self.assertTrue(np.amax(right) <= 1.0)
        self.assertTrue(np.amin(right) >= -1.0)

        # check the distribution of sameness
        # as long has classes are not super imbalanced im happy
        print(f"average sameness in paired embedding = {np.average(sameness):.3}, batch_size = {sameness.shape[0]}")
        self.assertTrue(0.4 <= np.average(sameness) <= 0.6)


unittest.main()
