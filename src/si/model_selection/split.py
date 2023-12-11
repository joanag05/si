from typing import Tuple

import numpy as np

from typing import Dict, Tuple, Callable, Union

import random
from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.20, random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets.

    Parameters
    ----------
    dataset: Dataset
        The dataset to split.
    test_size: float
        The size of the testing Dataset (e.g., 0.2 for 20%).
    random_state: int
        Seed for generating permutations.

    Returns
    -------
    Tuple[Dataset, Dataset]
        A tuple containing the stratified train and test Dataset objects.
    """
    unique_labels, label_counts = np.unique(dataset.y, return_counts=True)

    train_idx = []
    test_idx = []

    if random_state is not None:
        random.seed(random_state)

    for label in unique_labels:
        num_test_samples = int(label_counts[label] * test_size)

        class_indices = np.where(dataset.y == label)[0]
        random.shuffle(class_indices)

        
        test_idx.extend(class_indices[:num_test_samples])

       
        train_idx.extend(class_indices[num_test_samples:])

    train_data = dataset.X[train_idx]
    train_labels = dataset.y[train_idx]
    train_dataset = Dataset(train_data, train_labels, dataset.features, dataset.label)

    test_data = dataset.X[test_idx]
    test_labels = dataset.y[test_idx]
    test_dataset = Dataset(test_data, test_labels, dataset.features, dataset.label)

    return train_dataset, test_dataset


# Testing the function

if __name__ == '__main__':


    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data # Features
    y = iris.target # Labels

    dataset = Dataset(X, y, label="iris", features=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    
    
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

    print("Train set size:", len(train_set))
    print("Test set size:", len(test_set))


    
    train_set, test_set = stratified_train_test_split(dataset, test_size=0.2, random_state=42)

    print("Train set size:", len(train_set))
    print("Test set size:", len(test_set))

