<img src="tbi_logo.png" alt="drawing" width="250"/>

# Cell Segmentation with SplineDist

This is a starter repository for trying out [SplineDist](https://github.com/uhlmanngroup/splinedist) in a clean Python environment.

SplineDist is a machine-learning framework created for the purpose of cell segmentation using spline curves. It was meant to improve on the limitation of StarDist to only star-convex polygons by using spline curves instead of radial distances. The [manuscript](https://www.biorxiv.org/content/10.1101/2020.10.27.357640v1) was accepted at [ISBI 2021](https://biomedicalimaging.org/2021/).

## Usage
### Installation
Use either [conda](https://docs.conda.io/en/latest/miniconda.html) or [mamba](https://github.com/mamba-org/mamba) as the package manager.
1. Clone this repository
```
mamba env create -f environment.yml
mamba activate splinedist-starter
```

### Training
1. Place training images and masks in `data/train/images` and `data/train/masks`
```
mamba activate splinedist-starter
python train.py
```

### Prediction
1. Make predictions `python predict.py`

## About Turku BioImaging

<img src="tbi_logo.png" alt="drawing" width="250"/>


Turku BioImaging (TBI) is a broad-based, interdisciplinary science and infrastructure umbrella that aims to unite bioimaging expertise in Turku and elsewhere in Finland.

TBI is jointly operated by the University of Turku and Åbo Akademi University. TBI manages Euro-BioImaging Finland, coordinates funding acquisition for imaging instrumentation and services, operates an international MSc program in biomedical imaging, and develops new imaging-related services (e.g. image data and multimedia). TBI-associated imaging facilities provide open access services ranging from molecular and cellular imaging and high content analysis to whole animal and human imaging. TBI is led by a tripartite management team, and has a broad-based Steering Committee.

[Turku BioImaging Website](https://bioimaging.fi)
