# Rover Rock Sample Analysis

Ensemble rock-type classifier for rover sample analysis. Five PyTorch models
(ResNet-18, EfficientNet-B0, MobileNetV4-Small, ShuffleNetV2, YOLO11n-cls) each
predict one of 47 classes; their softmax probabilities are averaged and the
result is written onto a copy of the input image.

## Setup

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install timm ultralytics pillow

## Model weights (not in this repo)

Download the five .pth files and place them in a `Models/` folder next to the scripts:
[add your Google Drive link here]

## Usage

    python infer_ensemble.py path/to/rock_image.jpg

The annotated image is saved to `results/`.

## Known limitations

- The checkpoints store class indices (0-46) only, not rock names. The mapping in
  `CLASS_NAMES` (infer_ensemble.py) is not yet verified, so the image shows
  "Name not set". `find_class_mapping_v2.py` is the tool for recovering the order
  from the dataset folders; it has not been run on the full dataset yet.
- The five models often disagree on the same image. Per-model accuracy has not
  been measured, so treat the ensemble output as unvalidated.
