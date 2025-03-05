# Airbus Ship Detection Project

## Overview
This project implements a deep learning model for ship detection using a U-Net-like convolutional neural network architecture. The model is trained on the Airbus Ship Detection dataset from Kaggle and can identify ships in satellite imagery.

## Features
- Custom U-Net model for ship segmentation
- Intersection over Union (IoU) loss function
- Streamlit web application for ship detection
- Data visualization and preprocessing scripts

## Requirements
- Python 3.8+
- TensorFlow
- OpenCV
- Matplotlib
- Streamlit
- NumPy
- Pandas

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

## Model Architecture
The model uses a U-Net-like architecture with:
- Encoder path with convolutional and max-pooling layers
- Decoder path with transposed convolutions and skip connections
- Sigmoid activation for binary segmentation

## Dataset
Trained on the Airbus Ship Detection dataset from Kaggle
[Dataset Link](https://www.kaggle.com/competitions/airbus-ship-detection)

## Metrics
- Intersection over Union (IoU)
- Background IoU

## Usage
Place images in the `ship_images/` directory and run the Streamlit application to detect ships.
