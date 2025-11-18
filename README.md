# ASL Alphabet Recognition: MobileNetV2 vs. Custom CNN

This repository contains two Jupyter notebooks demonstrating different Deep Learning approaches for recognizing American Sign Language (ASL) alphabets from images.

## Project Overview
The goal is to classify hand gesture images into **29 categories**: letters A-Z, plus 'del', 'space', and 'nothing'.

## Dataset
**ASL Alphabet Dataset** containing 87,000 training images (200x200 pixels) and a separate test set.

## Notebooks

### 1. `asl-project-mobilenetv2 (2).ipynb`
* **Method:** Transfer Learning
* **Model:** **MobileNetV2** (pre-trained on ImageNet)
* **Input Size:** 224x224
* **Highlights:**
    * Uses `tensorflow.keras.applications.MobileNetV2` as the feature extractor.
    * Freezes base layers and trains a custom dense head.
    * Fast convergence and high accuracy suitable for mobile deployment.
    * Includes `EarlyStopping` and `ReduceLROnPlateau` callbacks.

### 2. `asl-project-with-cnn-model.ipynb`
* **Method:** Custom Architecture
* **Model:** 4-Block **Convolutional Neural Network (CNN)**
* **Input Size:** 128x128
* **Highlights:**
    * Built from scratch using `Conv2D`, `MaxPooling2D`, `BatchNormalization`, and `Dropout`.
    * Extensive **Data Augmentation** (Rotation, Shifts, Shear, Zoom) to improve generalization.
    * Good baseline for understanding CNN feature learning.

## Model Comparison

The following table provides a side-by-side comparison of the two methodologies implemented in this project.

| Feature | Notebook 1: MobileNetV2 | Notebook 2: Custom CNN |
| :--- | :--- | :--- |
| **Model Type** | **Transfer Learning** (Pre-trained) | **Custom CNN** (Sequential) |
| **Architecture** | Uses **MobileNetV2** as a base (inverted residuals, linear bottlenecks), removing the top layer and adding custom Dense layers for classification. | A custom stack of **4 Convolutional Blocks** (increasing filters: 32, 64, 128, 256) followed by fully connected Dense layers (512, 256). |
| **Input Image Size** | **224 x 224 pixels** (Standard for MobileNet). | **128 x 128 pixels** (Smaller, faster for custom training). |
| **Preprocessing** | Uses `preprocess_input` specific to MobileNetV2 (scales inputs between -1 and 1). | Uses standard rescaling (`1./255`) to normalize pixel values between 0 and 1. |
| **Data Augmentation**| Basic scaling and preprocessing. | **Extensive Augmentation:** Implements rotation (15°), width/height shifts, shearing, zooming, and brightness adjustments to improve generalization. |
| **Regularization** | Uses Dropout and Batch Normalization in the custom head added to the base model. | Heavily regularized with **Batch Normalization** after every Conv/Dense layer and **Dropout** (0.4 to 0.6) to reduce overfitting. |
| **Training Speed** | Generally converges **faster** because the feature extraction layers are already learned; only the top layers need extensive training initially. | **Slower** to converge as the model must learn edge and shape detection features from scratch. |
| **Performance** | Typically achieves **higher accuracy (>95%)** more quickly due to the robust features learned from ImageNet. | Achieves competitive accuracy (**~90-95%**) but requires more epochs and careful tuning of augmentation to reach the same level as the pre-trained model. |

## Requirements
* opencv-python
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-Learn
* Pillow
* mediapipe
