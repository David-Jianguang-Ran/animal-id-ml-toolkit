"""
keras data generator object for image data
"""
import pandas as pd
import numpy as np
import random


from keras.utils import Sequence
from keras.applications.xception import preprocess_input

from data_tools.utils import append_image_data_chunk


class SequentialGenerator(Sequence):
    """
    this generator assumes the data has following format:
    * each data chunk has two files, a label.csv and a dataset.parquet
    * label columns = image_id, animal_id
    """
    def __init__(self, base_path, chunk_nums, image_shape=(320, 320, 3), batch_size=1024):
        self.image_shape = image_shape
        self.batch_size = batch_size

        # read each image_data chunk into memory,
        # note each image was stored as a column in parquet file,
        # here we flip the axis to each image is a row and index of each image is it's image_id
        self._data = pd.DataFrame()
        for each_num in chunk_nums:
            self._data = append_image_data_chunk(self._data, f"{base_path}/dataset_{each_num}.parquet")

        # load labels
        # TODO decide whether to rely on image data and labels to have the same order
        # so far i say yes, but it's probably a bad idea
        self._labels = pd.concat([pd.read_csv(f"{base_path}/dataset_labels_{each_num}.csv", index_col=0)for each_num in chunk_nums])

    def __len__(self):
        if self._labels.shape[0] % self.batch_size != 0:
            return self._labels.shape[0] // self.batch_size + 1
        else:
            return self._labels.shape[0] // self.batch_size

    def __getitem__(self, index):
        """

        :param index:
        :return: image(batch_size, x, y, channel), label (batch_size, )
        """
        x, y = self._get_block(index, self.batch_size)
        return self._apply_preprocessing(x), y

    def _apply_preprocessing(self,data):
        return preprocess_input(data)

    def _get_block(self, block_index, block_size):
        if (block_index + 1) * block_size < self._labels.shape[0]:
            block_image = self._data.values[block_index * block_size: (block_index + 1) * block_size, :]
            block_label = self._labels["animal_id"].values[block_index * block_size: (block_index + 1) * block_size]
        else:
            block_image = self._data.values[block_index * block_size:, :]
            block_label = self._labels["animal_id"].values[block_index * block_size:]

        # reshape batch image before returning
        return block_image.reshape((-1, *self.image_shape)), block_label

    @staticmethod
    def _append_image_data_chunk(image_frame: pd.DataFrame, file_to_append: str):
        # this function load a parquet data chunk and 'add' it to image_frame
        # note the data is flipped compared to the parquet file
        to_add = pd.read_parquet(file_to_append, engine="pyarrow")
        return pd.concat([image_frame, to_add.T], axis=0)


class RandomBlockGenerator(SequentialGenerator):
    """
    Pretty much the same as SequentialGenerator,
    instead of feeding all data row in order,
    the orders are shuffled somewhat,
    but each epoch will still only go through each row once,
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # the arrangement of each block is generated now
        # this helps multiprocessing and ensuring each data row is used once per epoch
        self.block_order = [i for i in range(len(self) * 2)]
        random.shuffle(self.block_order)

    def __getitem__(self, index):
        # each batch is two randomly selected blocks
        a_x, a_y = self._get_block(self.block_order[index], self.batch_size // 2)
        b_x, b_y = self._get_block(self.block_order[index + len(self)], self.batch_size // 2)

        # combine and preprocess batch
        total_x = np.concatenate((a_x, b_x), axis=0)
        del a_x, b_x
        total_x = self._apply_preprocessing(total_x)

        total_y = np.concatenate((a_y, b_y), axis=0)

        return total_x, total_y


class PairedByLabelMixin:
    # do i actually need these useless class attributes?
    _data = None
    _labels = None
    batch_size = 0

    def __len__(self):
        return self._labels.shape[0] // (self.batch_size // 2)

    def __getitem__(self, index):
        """
        returns two embedding batches and a label on animal_id match
        this method is expected to return roughly 50/50 same/different animal_id match
        :param index:
        :return: (left embeddings, right embeddings) , animal_id match
        """
        # first let's get our anchor data, this will be the left half
        block_size = self.batch_size // 2
        if (index + 1) * block_size < self._labels.shape[0]:
            anchor = self._labels[index * block_size: (index + 1) * block_size]
        else:
            anchor = self._labels[index * block_size:]

        # right half - same id as anchor
        same = self.random_but_same_id(anchor, key="animal_id")

        # right half - randomly sampled from entire data set, likely to be different
        different = self._labels.sample(n=block_size)
        same_ness = np.equal(anchor["animal_id"].values, different["animal_id"].values).astype("float32").reshape((-1,1))

        # get embedding data based on labels
        left_data = self._data.loc[anchor["image_id"]]
        right_same = self._data.loc[same["image_id"]]
        right_diff = self._data.loc[different["image_id"]]

        # get sameness label
        batch_label = np.concatenate([
            np.full((block_size,1), fill_value=1.0, dtype="float32"),
            same_ness
        ], axis=0)

        return (
                   np.concatenate([left_data.values, left_data.values], axis=0),
                   np.concatenate([right_same.values, right_diff.values])
               ), batch_label

    @staticmethod
    def random_but_same_id(label_frame: pd.DataFrame, key="animal_id") -> pd.DataFrame:
        """
        returns a dataframe with same shape as input,
        the order of each image is scrambled but the order of selected key column should be the same
        """
        unique_keys = label_frame[key].unique()
        outputs = []
        for each_key in unique_keys:
            outputs.append(label_frame[label_frame[key] == each_key].sample(frac=1))
        return pd.concat(outputs, ignore_index=True, axis=0)


class PairedEmbeddingGenerator(PairedByLabelMixin, Sequence):
    """
    this generator expects data with each row being embedding vector for each image, indexed by image_id

    to select image by image_id
    d.T.loc[l[:50]["image_id"]]
    """
    def __init__(self, base_path, data_nums, batch_size=256, label_nums=(0,)):
        self.batch_size = batch_size

        # read each embedding_data chunk into memory,
        # note each embedding was stored as a column in parquet file,
        # here we flip the axis to each image is a row and index of each image is it's image_id
        self._data = pd.DataFrame()
        for each_num in data_nums:
            self._data = append_image_data_chunk(self._data, f"{base_path}/embeddings_{each_num}.parquet")

        self._labels = pd.concat([pd.read_csv(f"{base_path}/dataset_labels_{each_num}.csv", index_col=0)for each_num in label_nums])


class PairedImageGenerator(PairedByLabelMixin,SequentialGenerator):
    def __getitem__(self, index):
        (x1_flat, x2_flat), y = super().__getitem__(index)

        # reshape and do preprocessing
        x1 = self._apply_preprocessing(x1_flat.reshape((-1,*self.image_shape)))
        x2 = self._apply_preprocessing(x2_flat.reshape((-1,*self.image_shape)))

        return (x1, x2), y
