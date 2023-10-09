from si.data.dataset import Dataset
from typing import Tuple

import numpy as np

def train_test_selection(dataset : Dataset, test_size: float = 0.2, random_state: int = 50) -> Tuple[Dataset:Dataset]:
    """
    Splits the input dataset into training and testing sets.

    Args:
        dataset (Dataset): The dataset to be split.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Seed used by the random number generator. Defaults to 50.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training and testing datasets.
    """
    
    np.random.seed(random_state)
    samples = dataset.shape()[0]
    test_size = int( samples * test_size) # get number of samples in the test set
    indices = np.random.permutation(samples)
    test_indices = indices[:test_size] #samples in test
    train_indices = indices[test_size:] #samples in train
    


    print(dataset.X)
    print(dataset.y)
    trains = Dataset(dataset.X[train_indices], dataset.y[train_indices], features=dataset.features, label=dataset.label)
    tests = Dataset(dataset.X[test_indices], dataset.y[test_indices], features=dataset.features, label=dataset.label)
    
    return trains, tests
    
def stratified_train_test_split(dataset: Dataset,test_size: float = 0.2,random_state: int = 50, stratify_by: str = None) -> Tuple[Dataset, Dataset]:
    """
    Splits the input dataset into training and testing sets, stratifying by the specified feature.

    Args:
        dataset (Dataset): The dataset to be split.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Seed used by the random number generator. Defaults to 50.
        stratify_by (str, optional): The feature to stratify by. Defaults to None.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training and testing datasets.
    """
    np.random.seed(random_state)
    samples = dataset.shape()[0]
    test_size = int(samples * test_size) # get number of samples in the test set
    indices = np.random.permutation(samples)
    test_indices = indices[:test_size] 
    train_indices = indices[test_size:] 

    if stratify_by is not None:
        stratify_col = dataset.features.index(stratify_by)
        stratify_vals = dataset.X[:, stratify_col]
        stratify_train = stratify_vals[train_indices]
        stratify_test = stratify_vals[test_indices]
        _, train_counts = np.unique(stratify_train, return_counts=True)
        _, test_counts = np.unique(stratify_test, return_counts=True)
        while np.min(train_counts) < 2 or np.min(test_counts) < 2:
            indices = np.random.permutation(samples)
            test_indices = indices[:test_size] 
            train_indices = indices[test_size:] 
            stratify_train = stratify_vals[train_indices]
            stratify_test = stratify_vals[test_indices]
            _, train_counts = np.unique(stratify_train, return_counts=True)
            _, test_counts = np.unique(stratify_test, return_counts=True)

    trains = Dataset(dataset.X[train_indices], dataset.y[train_indices], features=dataset.features, label=dataset.label)
    tests = Dataset(dataset.X[test_indices], dataset.y[test_indices], features=dataset.features, label=dataset.label)

    return trains, tests
    