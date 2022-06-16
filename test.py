import os
import numpy as np
from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from splinedist import (
    fill_label_holes,
    random_label_cmap,
    calculate_extents,
    gputools_available,
)
from splinedist.matching import matching, matching_dataset
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D

np.random.seed(42)
lbl_cmap = random_label_cmap()

import splinegenerator as sg
from splinedist.utils import phi_generator, grid_generator, get_contoursize_max
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

X = sorted(glob("data/train/images/*.tif"))
Y = sorted(glob("data/train/masks/*.tif"))
assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))

X = list(map(imread, X))
Y = list(map(imread, Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
