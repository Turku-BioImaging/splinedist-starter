# Cell Segmentation with SplineDist
<img src="tbi_logo_horizontal.jpg" alt="drawing" width="200"/>

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