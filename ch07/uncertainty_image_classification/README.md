# Uncertainty estimation for image classification

# Uncertainty Analysis in Neural Networks

This project demonstrates uncertainty analysis in neural networks using MNIST and Fashion MNIST datasets.

## Features

- Trains a neural network on MNIST data
- Evaluates uncertainty on in-distribution, out-of-distribution, and ambiguous data
- Calculates and visualizes three types of uncertainty:
  - Total uncertainty
  - Data uncertainty
  - Knowledge uncertainty
- Computes AUC scores for uncertainty-based classification

## Usage

Run the main script with optional parameters:

```commandline
poetry run python ch07/uncertainty_image_classification/main.py [--nb-epochs INT] [--nb-mc-iter INT]
```

- `--nb-epochs`: Number of training epochs (default: 50)
- `--nb-mc-iter`: Number of Monte Carlo iterations for uncertainty estimation (default: 50)

## Output

- Saves the trained model as `model_mnist.keras`
- Generates `uncertainty_types.png` visualization
- Prints AUC scores for different uncertainty types and datasets
