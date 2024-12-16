from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

class AugmentationTransforms:
    """
    A class to define different image transformations for data augmentation.

    This class provides transformations for:
        - Strong augmentation: Used for unlabeled data, with larger random affine transformations (translation and scale).
        - Weak augmentation: Used for labeled data, with smaller random affine transformations (translation and scale).
        - Test transformation: Used for test data, with no augmentation other than converting to a tensor.

    Attributes:
        strong_transform (transforms.Compose): A pipeline of transformations applying stronger augmentation to images.
        weak_transform (transforms.Compose): A pipeline of transformations applying weaker augmentation to images.
        test_transform (transforms.Compose): A pipeline for test data, which only converts images to tensor format.

    Methods:
        __init__(self): Initializes the augmentation transforms with predefined strong, weak, and test transformations.

    """
    def __init__(self):
        self.strong_transform = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
        ])

        self.weak_transform = transforms.Compose([
            transforms.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

def prepare_data(labeled_ratio, batch_size):
    """
    Prepares the MNIST dataset for semi-supervised learning by splitting the training data 
    into labeled and unlabeled subsets based on the specified labeled_ratio.

    Args:
        labeled_ratio (float): Proportion of the dataset to be used for labeled data (0 < labeled_ratio < 1).
        batch_size (int): Batch size to be used for the DataLoader.

    Returns:
        tuple: A tuple containing:
            - labeled_loader (DataLoader): DataLoader for the labeled dataset.
            - unlabeled_loader (DataLoader): DataLoader for the unlabeled dataset.
            - test_loader (DataLoader): DataLoader for the test dataset.
    
    Prints:
        - Class distribution in the labeled dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize for MNIST
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Calculate samples per class for labeled data
    total_samples = len(train_dataset)
    num_classes = 10
    samples_per_class = int((labeled_ratio * total_samples) / num_classes)

    # Create dictionaries to store indices for each class
    class_indices = {i: [] for i in range(num_classes)}
    
    # Group indices by class
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)

    # Randomly select equal number of samples from each class for labeled data
    labeled_indices = []
    unlabeled_indices = []
    
    for class_label in range(num_classes):
        # Shuffle indices for current class
        current_class_indices = class_indices[class_label]
        np.random.shuffle(current_class_indices)
        
        # Select samples for labeled set
        labeled_indices.extend(current_class_indices[:samples_per_class])
        # Remaining samples go to unlabeled set
        unlabeled_indices.extend(current_class_indices[samples_per_class:])

    # Shuffle both sets of indices
    np.random.shuffle(labeled_indices)
    np.random.shuffle(unlabeled_indices)

    # Create subsets
    labeled_dataset = Subset(train_dataset, labeled_indices)
    unlabeled_dataset = Subset(train_dataset, unlabeled_indices)

    # Verify class distribution in labeled set
    labeled_distribution = {i: 0 for i in range(num_classes)}
    for idx in labeled_indices:
        _, label = train_dataset[idx]
        labeled_distribution[label] += 1
    
    print("\nLabeled data class distribution:")
    for class_idx, count in labeled_distribution.items():
        print(f"Class {class_idx}: {count} samples ({count/len(labeled_indices)*100:.2f}%)")

    # Create DataLoaders
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return labeled_loader, unlabeled_loader, test_loader
