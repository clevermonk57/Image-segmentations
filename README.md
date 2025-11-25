# Image-segmentations
Assignment 2 (Image segmentations)
Overview

This project performs semantic segmentation using two different models:

SegNet (custom implementation with VGG16 encoder and skip connections)

DeepLabV3 (using ResNet-50 encoder from segmentation_models_pytorch)

Both models are trained and evaluated on a custom dataset in COCO format.
The project runs in Google Colab and supports training, validation, visualization, and inference.

Dataset

COCO-style annotations (labels.json)

Images split into train, validation, and test folders

Only four target categories are used:

person (1)

cat (17)

sports ball (37)

book (84)

Background is class 0, giving a total of 5 classes

Annotations are filtered and converted into segmentation masks during training.

Preprocessing and Augmentation

All images and masks are resized to a fixed resolution.
Augmentations include horizontal flip, brightness/contrast, rotation, noise, and normalization.
Albumentations is used for all transformations.

SegNet Model

The SegNet notebook includes:

Custom PyTorch implementation

VGG16 encoder with pretrained ImageNet weights

Decoder with skip connections

Cross-entropy and Dice loss

Training loop with mixed precision

Saving best model based on validation mIoU

Inference code that outputs mask and overlay

DeepLabV3 Model

The DeepLab notebook includes:

DeepLabV3 from segmentation_models_pytorch

ResNet-50 encoder

Training with cross-entropy loss

Evaluation using per-class and mean IoU

Saving best model checkpoint

Inference and mask visualization on test images

Dataloader

Both models share the same COCO segmentation dataset class:

Loads images and annotations

Converts polygons to binary masks

Maps COCO category IDs to compact class indices

Applies augmentations

Returns image tensor, mask tensor, and filename

Training

Both training pipelines include:

Adam optimizer

Learning rate = 1e-4

Batch size = 8

Evaluation after each epoch

Saving the best model weights

Plotting loss curves and mIoU curves

Inference

For both SegNet and DeepLab:

Load the best saved checkpoint

Predict segmentation mask

Resize prediction back to original image size

Generate two visual outputs:

colorized mask

image–mask overlay

Results

Each notebook produces the following files:

best_model.pth checkpoint

loss.png

miou.png

Predicted masks and overlay images for test samples
