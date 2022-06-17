import sys
import os
import numpy as np

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible
from splinedist import fill_label_holes

from skimage import io

from splinedist import random_label_cmap, _draw_polygons, export_imagej_rois
from splinedist.models import SplineDist2D
from splinedist.utils import iou_objectwise, iou

np.random.seed(6)
lbl_cmap = random_label_cmap()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load Data
X = sorted(glob("data/*.tif"))
paths = sorted(glob("data/*.tif"))
X = list(map(imread, X))

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0, 1)  # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print(
        "Normalizing image channels %s."
        % ("jointly" if axis_norm is None or 2 in axis_norm else "independently")
    )

# Load trained model
model = SplineDist2D(None, name="splinedist01", basedir="models")

# Prediction
for idx, i in enumerate(X):
    img = normalize(i[1], 1, 99.8, axis=axis_norm)
    labels, _ = model.predict_instances(img)
    fname = os.path.basename(paths[idx])
    io.imsave(f"predictions/{fname}", labels, check_contrast=False)