

from typing import Callable, Union

import numpy as np
import csv
from si.data.dataset import Dataset
from si.metrics.rmse import rmse

from si.statistics.euclidean_distance import euclidean_distance

class KNNRegressor:
    """
    A k-nearest neighbors regressor.

    Parameters:
    -----------
    k : int, optional (default=1)
        The number of neighbors to consider.
    distance : Callable, optional (default=euclidean_distance)
        The distance metric to use when computing distances between instances.

    Attributes:
    -----------
    k : int
        The number of neighbors to consider.
    distance : Callable
        The distance metric to use when computing distances between instances.
    dataset : None
        The training dataset.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        self.k = k
        self.distance = distance
        self._train_dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Fits the KNNRegressor model to the given dataset.

        Args:
            dataset (Dataset): The dataset to fit the model to.

        Returns:
            KNNRegressor: The fitted KNNRegressor model.
        """
        self._train_dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray) -> int:
        """
        Returns the predicted label for a given sample by finding the k nearest neighbors.

        Args:
            sample (np.ndarray): The sample for which to predict the label.

        Returns:
            Union[int, str]: The predicted label for the sample.
        """
        distances = self.distance(sample, self._train_dataset.X)
        k_nearest_neighbors = np.argsort(distances)[:self.k]
        k_nearest_neighbors_labels = self._train_dataset.y[k_nearest_neighbors]

        return np.mean(k_nearest_neighbors_labels)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the labels for a given dataset.

        Args:
            dataset (Dataset): The dataset for which to predict the labels.

        Returns:
            np.ndarray: The predicted labels.
        """
        return np.apply_along_axis(self._get_closest_label, 1, dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        Computes the root mean squared error (RMSE) between the predicted labels and the true labels.

        Args:
            dataset (Dataset): The dataset for which to compute the score.

        Returns:
            float: The RMSE score.
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)
 
if __name__ == "__main__":
    
    from si.model_selection.split import train_test_split
    from si.models.knn_regressor import KNNRegressor
    from si.io.csv_file import read_csv

    cpu = read_csv("\\Users\\joana\\OneDrive\\Documentos\\GitHub\\si\\datasets\\cpu\\cpu.csv", sep = ",", features=True, label=True)

    train, test = train_test_split(cpu, test_size=0.2)

    # initialize the KNN_Classifier

    knn = KNNRegressor(k=3)

    # fit the model

    knn.fit(train)

    # evaluate the model on test data

    score = knn.score(test)
    print("Score RMSE:", score)