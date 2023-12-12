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
    # set random state

    np.random.seed(random_state)

    # get unique labels and their counts

    unique_labels, counts = np.unique(dataset.y, return_counts=True)  #finds the unique elements in an array and the indexes of those
    
    # initialize train and test indexes

    train_idx = []
    test_idx = []

    # lopp over the unique labels and their counts

    for label in unique_labels:
        # get the number of samples in the test set
        n_test = int(counts[np.where(unique_labels == label)] * test_size)

        # get the permutations of the samples with the current label
        permutations = np.random.permutation(np.where(dataset.y == label)[0])

        # get the samples in the test set
        test_idx.extend(permutations[:n_test])

        # get the samples in the training set
        train_idx.extend(permutations[n_test:])

    train_dataset = Dataset(dataset.X[train_idx],dataset.y[train_idx],features=dataset.features, label=dataset.label) #changes in X and y, adapted to train and test
    test_dataset = Dataset(dataset.X[test_idx],dataset.y[test_idx],features=dataset.features, label=dataset.label)

    return train_dataset, test_dataset


# Testing the function

if __name__ == '__main__':


    from sklearn import datasets
    from si.io.csv_file import read_csv

    filename = "\\Users\\joana\\OneDrive\\Documentos\\GitHub\\si\\datasets\\iris\\iris.csv"
    iris_data = read_csv(filename= filename, sep = ',', features = True, label = True)
    
    train_set, test_set = train_test_split(iris_data, test_size=0.2, random_state=42)

    print("Train set size:", train_set.shape())
    print("Test set size:", test_set.shape())


    
    train_dataset, test_dataset = stratified_train_test_split(iris_data, test_size=0.2, random_state=42)

    print("Train set size:", train_dataset.shape())
    print("Test set size:", test_dataset.shape())

