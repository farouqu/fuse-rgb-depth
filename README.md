# RGB-Depth Fusion Architecture for Semantic Segmentation using FCN

## Overview

This project focuses on developing a fusion architecture for semantic segmentation utilizing RGB and depth information. The architecture is based on the Fully Convolutional Network (FCN). By integrating both RGB and depth modalities, the goal is to enhance the segmentation accuracy and robustness, particularly in scenarios like road scene analysis.

## Setup and Requirements

Ensure you have the following libraries and modules installed:
- `numpy`
- `cv2`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `keras`

## Dataset

The dataset comprises 1100 images per modality, focusing on road scenes. It is partitioned into train (600 images), test (200 images), and validation (300 images) sets. Each image is resized to 256x256 pixels, and labels are encoded using one-hot encoding. A custom DataLoader is implemented to facilitate efficient loading during model training.

Download the dataset here: [Fusion_RGB_Depth_Dataset](https://drive.google.com/file/d/1FqEyWwFU7vH_L6kayUr2ONg6cnut0kH3/view?usp=sharing)

## Architecture

The fusion architecture consists of two streams, each processing RGB and depth modalities, respectively. Key components include:

1. Pretrained ResNet50 as backbone for feature extraction.
2. Convolutional layers for further feature refinemet.
3. Dropout layers to prevent overfitting.
4. Concatenation of features from both streams.
5. Transposed convolutional layer for upsampling.
6. Softmax activation for pixel-wise classification.

## Model Compilation and Training

The FCN model is compiled using Stochastic Gradient Descent (SGD) optimizer with a defined learning rate schedule. Training is conducted using the provided DataLoader for a specified number of epochs.

## Evaluation

Evaluation of the trained model involves:
- Computing loss and accuracy on the test dataset.
- Generating semantic segmentation predictions on random test examples.
- Visualizing ground truth labels alongside predicted masks for qualitative assessment.

## Extra Experiment
An additional experiment you can do involves implementing FCNs for each modality independently and comparing their performance against the fusion model. 

Happy coding !!