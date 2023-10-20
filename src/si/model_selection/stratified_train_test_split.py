

import numpy as np
from typing import Dict, Tuple, Callable, Union
from si.data.dataset import Dataset

import random

def stratified_train_test_split(dataset, test_size=0.20, random_state=None):
    
    # Get unique class labels and their counts
    unique_labels, label_counts = dataset.get_unique_labels()

    # Initialize empty lists for train and test indices
    train_indices = []
    test_indices = []

    # Set random seed for reproducibility
    if random_state is not None:
        random.seed(random_state)

    # Loop through unique labels
    for label in unique_labels:
        # Calculate the number of test samples for the current class
        num_test_samples = int(label_counts[label] * test_size)

        # Shuffle and select indices for the current class
        class_indices = dataset.get_indices_by_label(label)
        random.shuffle(class_indices)
        test_indices.extend(class_indices[:num_test_samples])
        train_indices.extend(class_indices[num_test_samples:])

    # Create training and testing datasets
    train_data = dataset.get_data_by_indices(train_indices)
    test_data = dataset.get_data_by_indices(test_indices)
    train_dataset = Dataset(train_data)
    test_dataset = Dataset(test_data)

    return train_dataset, test_dataset
