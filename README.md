# HappyPose

[![Tests](https://github.com/agimus-project/happypose/actions/workflows/test.yml/badge.svg)](https://github.com/agimus-project/happypose/actions/workflows/test.yml)
[![Packaging](https://github.com/agimus-project/happypose/actions/workflows/packaging.yml/badge.svg)](https://github.com/agimus-project/happypose/actions/workflows/packaging.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/agimus-project/happypose/main.svg)](https://results.pre-commit.ci/latest/github/agimus-project/happypose/main)
[![Documentation Status](https://readthedocs.org/projects/happypose/badge/?version=latest)](https://happypose.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/agimus-project/happypose/branch/main/graph/badge.svg?token=TODO)](https://codecov.io/gh/agimus-project/happypose)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Toolbox and trackers for object pose-estimation. Based on the work [CosyPose](https://github.com/Simple-Robotics/cosypose) and [MegaPose](https://github.com/megapose6d/megapose6d). This directory is currently under development.


# Installation

This installation procedure will be curated.

```
git clone --recurse-submodules https://github.com/agimus-project/happypose.git
conda env create -f environment.yml
cd happypose/happypose/pose_estimators/cosypose
python setup.py install
pip install -e .
```

# Testing the install

```
python -m happypose.pose_estimators.megapose.src.megapose.scripts.run_inference_on_example barbecue-sauce --vis-outputs
python -m happypose.pose_estimators.cosypose.cosypose.scripts.run_inference_on_example cheetos --run-inference
```

# Downloading and preparing data

<details>
<summary>Click for details...</summary>

```
Create data dir in happypose/pose_estimators/cosypose/local_data
export MEGAPOSE_DATA_DIR=happypose/pose_estimators/cosypose/local_data
```

download [barbecue sauce](https://drive.google.com/drive/folders/10BIvhnrKGbNr8EKGB3KUtkSNcp460k9S) and put it in `$MEGAPOSE_DATA_DIR/examples/barbecue-sauce/`

<details>
<summary>Cosypose</summary>

All data used (datasets, models, results, ...) are stored in a directory `local_data` at the root of the repository. Create it with `mkdir local_data` or use a symlink if you want the data to be stored at a different place. We provide the utility `cosypose/scripts/download.py` for downloading required data and models. All of the files can also be [downloaded manually](https://drive.google.com/drive/folders/1JmOYbu1oqN81Dlj2lh6NCAMrC8pEdAtD?usp=sharing).

## BOP Datasets

For both T-LESS and YCB-Video, we use the datasets in the [BOP format](https://bop.felk.cvut.cz/datasets/). If you already have them on your disk, place them in `local_data/bop_datasets`. Alternatively, you can download it using :

```sh
python -m happypose.toolbox.utils.download --bop_dataset=ycbv
python -m happypose.toolbox.utils.download --bop_dataset=tless
```

Additional files that contain information about the datasets used to fairly compare with prior works on both datasets.

```sh
python -m happypose.toolbox.utils.download --bop_extra_files=ycbv
python -m happypose.toolbox.utils.download --bop_extra_files=tless
```

We use [pybullet](https://pybullet.org/wordpress/) for rendering images which requires object models to be provided in the URDF format. We provide converted URDF files, they can be downloaded using:

```sh
python -m happypose.toolbox.utils.download --urdf_models=ycbv
python -m happypose.toolbox.utils.download --urdf_models=tless.cad
```

In the BOP format, the YCB objects `002_master_chef_can` and `040_large_marker` are considered symmetric, but not by previous works such as PoseCNN, PVNet and DeepIM. To ensure a fair comparison (using ADD instead of ADD-S for ADD-(S) for these objects), these objects must *not* be considered symmetric in the evaluation. To keep the uniformity of the models format, we generate a set of YCB objects `models_bop-compat_eval` that can be used to fairly compare our approach against previous works. You can download them directly:

```sh
python -m happypose.toolbox.utils.download--ycbv_compat_models
```

Notes:

- The URDF files were obtained using these commands (requires `meshlab` to be installed):

  ```sh
  python -m happypose.toolbox.utils.download --models=ycbv
  python -m happypose.toolbox.utils.download --models=tless.cad
  ```

- Compatibility models were obtained using the following script:

  ```sh
  python -m happypose.pose_estimators.cosypose.cosypose.scripts.make_ycbv_compat_models
  ```

## Pre-trained models for single-view estimator

The pre-trained models of the single-view pose estimator can be downloaded using:

```sh
# YCB-V Single-view refiner
python -m happypose.toolbox.utils.download --model=ycbv-refiner-finetune--251020

# YCB-V Single-view refiner trained on synthetic data only
# Only download this if you are interested in retraining the above model
python -m happypose.toolbox.utils.download --model=ycbv-refiner-syntonly--596719

# T-LESS coarse and refiner models
python -m happypose.toolbox.utils.download --model=tless-coarse--10219
python -m happypose.toolbox.utils.download --model=tless-refiner--585928
```

## 2D detections

To ensure a fair comparison with prior works on both datasets, we use the same detections as DeepIM (from PoseCNN) on YCB-Video and the same as Pix2pose (from a RetinaNet model) on T-LESS. Download the saved 2D detections for both datasets using

```sh
python -m happypose.toolbox.utils.download --detections=ycbv_posecnn

# SiSo detections: 1 detection with highest per score per class per image on all images
# Available for each image of the T-LESS dataset (primesense sensor)
# These are the same detections as used in Pix2pose's experiments
python -m happypose.toolbox.utils.download --detections=tless_pix2pose_retinanet_siso_top1

# ViVo detections: All detections for a subset of 1000 images of T-LESS.
# Used in our multi-view experiments.
python -m happypose.toolbox.utils.download --detections=tless_pix2pose_retinanet_vivo_all
```

If you are interested in re-training a detector, please see the BOP 2020 section.

Notes:

- The PoseCNN detections (and coarse pose estimates) on YCB-Video were extracted and converted from [these PoseCNN results](https://github.com/yuxng/YCB_Video_toolbox/blob/master/results_PoseCNN_RSS2018.zip).
- The Pix2pose detections were extracted using [pix2pose's](https://github.com/kirumang/Pix2Pose) code. We used the detection model from their paper, see [here](https://github.com/kirumang/Pix2Pose#download-pre-trained-weights). For the ViVo detections, their code was slightly modified. The code used to extract detections can be found [here](https://github.com/ylabbe/pix2pose_cosypose).

</details>

<details>
<summary>Megapose</summary>

 ## 1. Download pre-trained pose estimation models
Download pose estimation models to $MEGAPOSE_DATA_DIR/megapose-models:

```
python -m happypose.toolbox.utils.download --megapose_models
```
