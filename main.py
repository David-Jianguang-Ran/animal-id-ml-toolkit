import pandas as pd
import matplotlib.pyplot as plt

import time
import pathlib

from matplotlib.colors import Normalize

from random import randint

from data_tools.utils import download_to_path, download_sample_dataset
from data_tools.settings import SAMPLE_LABELS_URL

from data_tools.utils import fancy_print


if __name__ == "__main__":
    download_sample_dataset()
