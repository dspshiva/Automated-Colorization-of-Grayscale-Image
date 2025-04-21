# Automated Colorization of Grayscale Images

This project implements an automated system for colorizing grayscale images using Convolutional Neural Networks (CNNs) and deep learning techniques. The system uses the Caffe framework, OpenCV for image processing, and CNNs for the colorization process. The goal of the project is to transform black-and-white images into their colorized counterparts, enhancing the image's aesthetic and information value.

## Project Overview

The core of the project involves using a deep learning model trained on a dataset of color images to predict and apply color to grayscale images. The colorization process is performed by training a Convolutional Neural Network (CNN) model, which learns to map grayscale pixel values to their corresponding color values.

## Technologies Used
- **Caffe**: Deep learning framework used for training the CNN model.
- **OpenCV**: Library for image processing and manipulation.
- **Python**: Programming language for implementing the model.
- **NumPy**: For numerical operations.
- **Matplotlib**: For visualization and displaying results.

## Features
- Converts grayscale images to realistic color images.
- Utilizes CNNs to predict the correct color for each pixel.
- Image preprocessing and postprocessing using OpenCV for optimal results.
- Ability to process large datasets for training and real-time colorization.

## Screenshots

### Colorization Enabled
![Colorization Enabled](https://github.com/dspshiva/Automated-Colorization-of-Grayscale-Image/blob/main/static/Screenshot%202025-04-21%20202241.png)

### Colorization Removed
![Colorization Removed](https://github.com/dspshiva/Automated-Colorization-of-Grayscale-Image/blob/main/static/Screenshot%202025-04-21%20201838.png)

## Training a Custom Model

If you'd like to train the model on your own dataset, follow the instructions in the `train_model.py` file. You will need a dataset of color images and grayscale versions of those images.

1. Prepare your dataset (color and grayscale images).
2. Configure the model architecture in `train_model.py`.
3. Run the training script:
   
   ```bash
   python train_model.py
