# Self-Training Neural Network with Active Learning

A PyTorch implementation of a self-training neural network that combines active learning techniques with pseudo-labeling for semi-supervised learning. This implementation is demonstrated on the MNIST dataset but can be adapted for other image classification tasks.

## Features

- Semi-supervised learning using self-training
- Uncertainty-weighted loss function
- Confidence-based pseudo-labeling
- Data augmentation with strong and weak transformations
- Class-balanced sampling for labeled data
- Dynamic pseudo-label selection and updating
- Automatic model checkpointing

## Requirements

- PyTorch
- torchvision  
- numpy
- tqdm

Install requirements:
```bash
pip install -r requirements.txt
```

## Architecture
The implementation consists of several key components:

- CustomCNN: A convolutional neural network architecture designed for MNIST
- AugmentationTransforms: Data augmentation pipeline with strong and weak transformations
- CustomActiveLearningLoss: Uncertainty-aware loss function for both labeled and pseudo-labeled data
- SelfTraining: Main training loop with pseudo-label generation and management

## Usage
You can run easily main script like below in terminal:

```bash
python main.py
```

### Key parameters in main():

- batch_size: Batch size for training (default: 1024)
- num_epochs: Number of training epochs (default: 1000)
- learning_rate: Learning rate for optimizer (default: 1e-4)
- confidence_threshold: Threshold for pseudo-label selection (default: 0.95)
- pseudo_weight: Weight for pseudo-label loss (default: 0.2)
- labeled_ratio: Ratio of labeled data to total data (default: 0.01)


## Key Features
### Uncertainty-Weighted Loss
This custom loss function is designed for active learning scenarios, combining two types of losses:

- Supervised Loss: For labeled data
- Pseudo-Label Loss: For unlabeled data with model-generated pseudo-labels
### Key features:

- Uses entropy-based uncertainty scoring to weight samples
- Applies temperature scaling to soften predictions
- Weights pseudo-labels based on prediction confidence
- Combines both losses with adjustable weights
- Includes uncertainty weighting for both labeled and pseudo-labeled samples
- The loss function helps the model learn from both labeled and unlabeled data while accounting for prediction uncertainty and confidence. This is particularly useful in semi-supervised learning scenarios where labeled data is limited.

## Results
The model achieves strong performance(around 85%) with only 1% labeled data(600 images) on MNIST dataset.

## Conclusion
With very low number of labeled images and using self-training, active learning, regularization, and augmentation possible to achieve promising performance.

The detais are explained in medium story which you can see from [link]().