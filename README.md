# Image Compression and Reconstruction using Custom Neural Networks

##  Overview
This project explores the use of deep learning for image compression and reconstruction. We implement a neural network from scratch using PyTorch (or NumPy) to reconstruct high-quality images from their compressed JPEG versions.


# Project Workflow: Image Compression and Reconstruction Using Neural Networks

This workflow outlines the step-by-step process for implementing the project, ensuring a structured approach from data preprocessing to model evaluation.

## Step 1: Data Exploration and Preprocessing

### Objective: Understand the dataset and prepare it for training.

#### Tasks:
- Download the **LIVE 1 Image Compression Dataset** from Data directory.
- Load and inspect image pairs (**original & compressed JPEG**).
- Convert images to **grayscale** or **RGB tensors**.
- Normalize pixel values to the **[0,1]** or **[-1,1]** range.
- Implement **data augmentation** (optional).
- Save preprocessed data for fast loading.

## Step 2: Neural Network Implementation

### Objective: Build a neural network for image compression and reconstruction.

#### Tasks:
- Start with a **simple autoencoder** architecture:
  - **Encoder:** Convolutional layers to extract features.
  - **Latent Space:** Fully connected (Dense) layers for compression.
  - **Decoder:** Deconvolutional layers for reconstruction.
- Experiment with different configurations:
  - Vary the number of layers (e.g., adding more convolutional layers).
  - Try **Batch Normalization** and **Dropout** to improve generalization.
  - Use activation functions (**ReLU, Sigmoid, etc.**).
- Implement multiple architectures to compare performance.

## Step 3: Training Process

### Objective: Train the neural network using original images as input and compressed JPEG images as targets.

#### Tasks:
- Define a loss function (**Mean Squared Error** or **L1 Loss**).
- Choose an optimizer (**Adam** or **SGD**).
- Train the model on the **training dataset**.
- Adjust learning rate dynamically (**learning rate scheduling**).
- Monitor training loss and validation loss using **TensorBoard** or **Matplotlib**.
- Save the **best model weights** in `image_compression_model.pth` based on validation performance of Deep CNN.

## Step 4: Model Evaluation

### Objective: Measure how well the model reconstructs images.

#### Tasks:
- Use **PSNR (Peak Signal-to-Noise Ratio)** to assess reconstruction quality.
- Compute **SSIM (Structural Similarity Index)** for perceptual quality.
- Compare model performance across different architectures.
- Generate side-by-side visualizations of:
  - **Original images**
  - **JPEG-compressed images**
  - **Reconstructed images**

## Step 5: Analysis & Optimization

### Objective: Understand the trade-offs between model complexity, training time, and reconstruction quality.

#### Tasks:
- Compare **simple vs. complex models** in terms of:
  - **Training time**
  - **PSNR & SSIM scores**
  - **Reconstruction quality**
- Optimize model performance by:
  - **Fine-tuning learning rates**
  - **Increasing or decreasing depth of convolutional layers**
  - **Adding skip connections or attention mechanisms** (if necessary)
- Discuss results and best practices for **neural network-based image compression**.
