# Project 2: Improved Digit Classifier with Optimization

## Objective
Enhance the MNIST digit classifier from Project 1 with regularization and Adam optimization.

## Skills
- Regularization (L2)
- Advanced optimization (Adam)
- Hyperparameter tuning
- PyTorch implementation

## Steps
1. Load and preprocess MNIST dataset with a validation split.
2. Define a 3-layer neural network with L2 regularization.
3. Train with mini-batch gradient descent and Adam optimizer.
4. Tune hyperparameters and evaluate performance.

## Setup
### Option 1: Miniconda
- Create and activate the environment:
  ```bash
  conda env create -f environment.yml
  conda activate deeplearningproject2