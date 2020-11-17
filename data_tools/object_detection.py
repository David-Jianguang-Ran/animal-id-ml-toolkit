"""
Here are some object detection utilities
built on top of experiencor/yolov3
"""
import os
import numpy as np
import cv2
import requests
import shutil
import matplotlib.pyplot as plt

from pathlib import Path

from .yolov3 import WeightReader, BoundBox, make_yolov3_model, preprocess_input_yolo, decode_netout, do_nms, correct_yolo_boxes
from .yolov3 import ANCHORS, COCO_LABELS

from .utils import fancy_print

from tensorflow.keras.applications.xception import Xception, decode_predictions, preprocess_input as preprocess_input_xception

# TODO change default weight path to project_root/models/temp
# fist make sure the pretrained weights exist
WEIGHT_PATH = Path(os.path.join(os.path.dirname(__file__), "temp","yolov3.weights"))
WEIGHT_URL = "https://pjreddie.com/media/files/yolov3.weights"


# having a real struggle to get the following class to work
class ImageNetClassifier:
    """
    singleton ish wrapper around keras.xception
    calling init multiple time will not result in multiple models in memory
    """
    __model = None

    # results other than interested labels will be ignored
    interested_labels = {"cat", "dog"}

    def __init__(self):
        if self.__model is None:
            ImageNetClassifier.__model = Xception()

    def filter_by_label(self, batch):
        """returns ([index of images that has interested subject in], [index of images without])"""
        network_out = self.__model.predict(preprocess_input_xception(batch))

        keep_index = []
        reject_index = []

        for i, each_image in enumerate(decode_predictions(network_out, top=5)):
            possible_classes = {each_pred[0] for each_pred in each_image}
            # no subject of interest found
            if len(possible_classes.intersection(self.interested_labels)) == 0:
                reject_index.append(i)
            else:
                keep_index.append(i)

        return keep_index, reject_index


class YOLO:
    """
    Singleton-ish wrapper for yolov3 application
    """
    __model = None
    __weights = None

    # results other than interested labels will be ignored
    interested_labels = {"cat", "dog"}

    @classmethod
    def _download_weights(cls, weight_path):
        # ensure weights exist:
        if os.path.isfile(weight_path):
            cls.__weights = weight_path
            return fancy_print(f"weights file found at {weight_path}")
        else:
            fancy_print(f"downloading pretrained weights from {WEIGHT_URL}")

        # ensure we have a dir to put downloaded file
        os.makedirs(WEIGHT_PATH.parent, exist_ok=True)

        # do request
        with requests.get(WEIGHT_URL, stream=True) as r:
            with open(WEIGHT_PATH, "wb") as file:
                shutil.copyfileobj(r.raw, file)

        fancy_print(f"saved file to {weight_path}")

    @classmethod
    def _load_model(cls, weight_path):
        model = make_yolov3_model()
        weight_reader = WeightReader(weight_path)
        weight_reader.load_weights(model)
        cls.__model = model

    def __init__(self, weight_path=WEIGHT_PATH):
        if self.__weights is None:
            YOLO._download_weights(weight_path)
            self.__weights = weight_path

        if self.__model is None:
            YOLO._load_model(self.__weights)
        else:
            pass

    def predict_n_crop(self, batch: np.ndarray, output_shape=(150, 150, 3), mode="pixel"):
        """
        This is the main interface with the yolo model,
        call this with a batch of images to get cropped images with labels
        __note:__ this function has no provision for keeping tracking of labels of input images
        __note:__ the pretrained yolov3/coco model is actually pretty bad having false positives
        :param batch: numpy array with shape (batch, x, y, channels)
        :param output_shape:  (ox, oy, ochannels)
        :param mode:
        :return: numpy array shape (?, ox, oy, ochannels)
        """
        # remember input dimension so we can restore it later
        batched = True if len(batch.shape) == 4 else False
        batch_x, batch_y, batch_channels = batch.shape[-3:]

        # preprocess image, resize to yolo network input size
        net_batch = preprocess_input_yolo(batch,mode=mode)

        # get predictions per batch
        predictions = self._get_bounding_boxes(net_batch)

        # crop and collect images
        found_images = []
        found_labels = []
        found_scores = []
        for i, each_prediction in enumerate(predictions):
            correct_yolo_boxes(each_prediction, batch_x, batch_y)  # <= bounding boxes are predicted in network size, fixing it here
            crops, labels, scores = self._crop_image(batch[i],each_prediction,output_shape=output_shape,batched=batched)
            found_images += crops
            found_labels += labels
            found_scores += scores

        # concat images together to a 4d array n, x, y, c
        if batched and len(found_images) == 1:
            found_images = found_images[0]
        elif batched and len(found_images) > 1:
            found_images = np.concatenate(found_images, axis=0)

        return found_images, found_labels, found_scores

    def _crop_image(self, image: np.ndarray, boxes: [BoundBox],output_shape=(180, 180, 3), batched=False):
        image_x, image_y, image_channels = image.shape[-3:]

        # crop out and reshape image for each bounding box
        out_images = []
        out_labels = []
        out_score = []
        for each_box in boxes:
            # we only care about cats and dogs here
            if each_box.text_label in self.interested_labels:

                # crop out a 1:1 section of the image and reshape to output_shape
                x_span = each_box.xmax - each_box.xmin
                y_span = each_box.ymax - each_box.ymin
                if x_span > y_span:
                    # pad y axis selection
                    crop_x = self._expand_range(each_box.xmin, each_box.xmax, x_span, 0, image_x)
                    crop_y = self._expand_range(each_box.ymin, each_box.ymax, x_span, 0, image_y)
                else:
                    # pad x axis selection
                    crop_y = self._expand_range(each_box.ymin, each_box.ymax, y_span, 0, image_y)
                    crop_x = self._expand_range(each_box.xmin, each_box.xmax, y_span, 0, image_x)

                try:
                    # do crop and resize
                    new_image = cv2.resize(image[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1], :], output_shape[:2])
                except cv2.error:
                    fancy_print(f"WARNING: cv2.error caught during new image resize, new image dimension {(crop_x, crop_y)}")
                    break

                # restore the batch dimension if that's how image was passed in
                if batched:
                    new_image = np.expand_dims(new_image, axis=0)

                out_images.append(new_image)
                out_labels.append(each_box.text_label)
                out_score.append(each_box.get_score())

        return out_images, out_labels, out_score

    def _get_bounding_boxes(self, image) -> [[BoundBox]]:
        """This method MUST be call on ONE IMAGE AT A TIME!!! UGH!"""
        # get predictions
        predictions = self.__model.predict(image)

        # decode output
        boxes = [[] for i in range(image.shape[0])]
        for j in range(len(predictions)):
            # decode the output of the network
            for i in range(image.shape[0]):
                boxes[i] += decode_netout(predictions[j][i], anchors=ANCHORS[j])

        # do nms
        boxes = [do_nms(each_pic,interested_classes=[COCO_LABELS.index(each) for each in self.interested_labels])for each_pic in boxes]
        # why are bounding boxes returned here have out of bound boxes?
        return boxes

    @staticmethod
    def _ensure_3d(image):
        # incase image has a batch dimension, only take the first image
        batch_dim = False
        if len(image.shape) == 4:
            batch_dim = True
            image = image[0, :, :, :]

        return image, batch_dim

    @staticmethod
    def _expand_range(low, high, target_width, low_bound, high_bound):
        """
        expands range low, high to target width,
        assuming high_bound - low_bound > target_width
        new low or high will not exceed low_bound, high_bound
        returns new_low, new_high
        note: sometimes low or high passed in are out of bounds
        """
        if high_bound - low_bound < target_width:
            # impossible target width specified, just return the max width
            return low_bound, high_bound

        pad_low = (target_width - high + low) // 2
        pad_high = pad_low + (target_width - high + low) % 2

        # assuming high bound - low bound >= target
        if low - pad_low < low_bound:
            # low side out of bounds
            output_low = low_bound
            output_high = min(low_bound + target_width, high_bound)
        elif high + pad_high > high_bound:
            # high side out of bounds
            output_high = high_bound
            output_low = max(high_bound - target_width, low_bound)
        else:
            # padded box fits
            output_high = high + pad_high
            output_low = low - pad_low

        return output_low, output_high


