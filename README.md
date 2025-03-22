# Computer Vision
# 1. Gaze Detection and Tracking :-
## Overview
This project implements gaze detection and tracking using computer vision techniques. It processes real-time video input to identify eye contours, apply masking, reshape NumPy arrays, and run inference.

## Included Files
- **`Contours.py`**: Extracts contours of the eyes for tracking
- **`EyeMasking.py`**: Applies masking techniques to isolate relevant eye regions
- **`NumPyReshaper.py`**: Handles reshaping of NumPy arrays for processing
- **`run.py`**: Main script to execute gaze detection and tracking

## Requirements
### Libraries
Ensure the following dependencies are installed before running the project:
- `dlib`
- `cv2` (OpenCV)
- `numpy`

You can install the required libraries using:
```bash
pip install opencv-python numpy dlib
```

## Setup and Usage
1. Ensure you have Python installed (recommended version 3.7 or later).
2. Install the necessary libraries as listed above.
3. Run `run.py` to start gaze detection and tracking.
```bash
python run.py
```
4. Follow on-screen instructions or refer to output for further details.

## Precautions for Using dlib
- `dlib` requires **CMake and Boost** installed for compilation if building from source.
- Windows users may need to install Visual Studio Build Tools to resolve dependencies.
- Linux users should ensure `build-essential` and `cmake` are installed.
- If encountering installation issues, consider using pre-compiled binaries where available.

# 2. Generative Adversarial Network (GAN) Project

This repository contains two implementations of a Generative Adversarial Network (GAN) for generating synthetic MNIST digits.

## Overview

This project implements a GAN to generate handwritten digit images using the MNIST dataset.  
- The **PyTorch implementation** is fully functional and well-structured for ease of use.  
- The **TensorFlow implementation** is currently incomplete and contains known bugs.

## Features

✅ PyTorch implementation works flawlessly  
✅ Generates high-quality synthetic MNIST digits  
✅ Modular code for better maintainability  
✅ Saves model checkpoints every 5 epochs  
❌ TensorFlow version is not working due to bugs  

## Installation

Before running the project, install the required dependencies:

```bash
pip install torch torchvision numpy matplotlib
```
For the TensorFlow implementation (if fixed in the future):

```bash
pip install tensorflow numpy matplotlib
```
## Usage
Training the GAN (PyTorch)
To train the model:

```bash
cd GAN_Pytorch
python train.py
```
Checkpoints are saved every 5 epochs in the checkpoints/ directory.

# Generating Images
Once trained, generate images using:

```bash
python generate.py
```
Generated images will be stored in the output/ directory.

## Notes
Use the PyTorch implementation as it is stable and fully functional.
The TensorFlow version currently contains issues and may not work correctly.
If modifying or improving the TensorFlow version, ensure proper debugging before use.



