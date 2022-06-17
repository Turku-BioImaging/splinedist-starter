# Cell Segmentation with SplineDist

## Installation
Use either [conda](https://docs.conda.io/en/latest/miniconda.html) or [mamba](https://github.com/mamba-org/mamba) as the package manager.
1. Clone this repository
```
mamba env create -f environment.yml
mamba activate splinedist-experiments
```

## Training
1. Place training images and masks in `data/train/images` and `data/train/masks`
```
mamba activate splinedist-experiments
python train.py
```

## Prediction
1. Make predictions `python predict.py`