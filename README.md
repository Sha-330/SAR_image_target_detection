# Radar Image Classification using Scattering Transform and Deep Learning

This repository provides a complete pipeline for radar image classification using the [Scattering2D transform](https://www.kymat.io/) for feature extraction and a deep neural network for classification. It includes utilities for training, evaluation, and inference, and supports bulk prediction with visualization and result export.

## Project Structure
- `padded_imgs/`  
  Input directory containing class-wise folders with images
- `radarmodel.h5`  
  Trained Keras model
- `label.pkl`  
  Label encoder for class mapping
- `train_model.py`  
  Script to train and evaluate the model
- `radar_tester.py`  
  Module for loading the model and making predictions
- `README.md`  
  Project documentation


## Requirements

Install dependencies using `pip`:

```bash
pip install numpy opencv-python matplotlib pandas scikit-learn tensorflow joblib kymatio
```
## Data Source 
https://www.kaggle.com/datasets/atreyamajumdar/mstar-dataset-8-classes/data
