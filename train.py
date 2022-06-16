import os
import sys
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
# assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))
assert len(X) == len(Y)

X = list(map(imread, X))
Y = list(map(imread, Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

# Normalize images and fill small label holes
axis_norm = (0, 1)  # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print(
        "Normalizing image channels %s."
        % ("jointly" if axis_norm is None or 2 in axis_norm else "independently")
    )
    sys.stdout.flush()

X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

# Split into train and validation datasets
assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
print("number of images: %3d" % len(X))
print("- training:       %3d" % len(X_trn))
print("- validation:     %3d" % len(X_val))

# choose the number of control points (M)
M = 8
n_params = 2 * M

# Predict on subsampled grid for increased efficiency and larger field of view
grid = (2, 2)

# compute the size of the largest contour present in the image-set
contoursize_max = get_contoursize_max(Y_trn)

conf = Config2D(
    n_params=n_params,
    grid=grid,
    n_channel_in=n_channel,
    contoursize_max=contoursize_max,
)
print(conf)
vars(conf)

phi_generator(M, conf.contoursize_max)
grid_generator(M, conf.train_patch_size, conf.grid)

model = SplineDist2D(conf, name="splinedist", basedir="models")

# Data augmentation


def random_fliprot(img, mask):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
    return x, y


# Training
model.train(
    X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter, epochs=300
)
