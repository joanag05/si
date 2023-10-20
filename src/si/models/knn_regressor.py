

from typing import Callable, Union

import numpy as np

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

        self.dataset = None
    
    
    def fit(self, dataset: Dataset) -> 'KNNRegressor':
            """
            Fits the KNNRegressor model to the given dataset.

            Args:
                dataset (Dataset): The dataset to fit the model to.

            Returns:
                KNNRegressor: The fitted KNNRegressor model.
            """
            self.dataset = dataset
            return self
    
    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
         
        distances = self.distance(sample, self.dataset.X)
        k_nearest_neighbors = np.argsort(distances)[:self.k]
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]
        tmp = np.average(k_nearest_neighbors_labels)

        return tmp
    
    class KNNRegressor:
        def __init__(self, k: int):
            """
            Initializes a KNNRegressor object with a given value of k.

            Args:
            - k: The number of nearest neighbors to consider when making a prediction.
            """
            self.k = k

        def fit(self, dataset: Dataset):
            """
            Fits the KNNRegressor object to a given dataset.

            Args:
            - dataset: The dataset to fit the KNNRegressor object to.
            """
            self.dataset = dataset

        def predict(self, dataset: Dataset) -> np.ndarray:
            """
            Predicts the target values for a given dataset using the KNNRegressor object.

            Args:
            - dataset: The dataset to make predictions for.

            Returns:
            - A numpy array containing the predicted target values for the given dataset.
            """
            return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
    def predict(self, dataset : Dataset) -> np.ndarray:
         
         return np.apply_along_axis(self._get_closest_label, axis = 1, arr=dataset.X)
    
    def score(self, dataset : Dataset) -> float:
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)
    
if __name__ == '__main__':
    
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    import pandas as pd
    from si.io.csv_file import read_csv
    from sklearn import preprocessing

    
    df = read_csv("C:\Users\joana\OneDrive\Documentos\GitHub\si\datasets\cpu.csv")
    print(df.head())
    dataset_ = Dataset.from_dataframe(df, label='perf')

    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.25)


    knn = KNNRegressor(k=5)


    knn.fit(dataset_train)

    score = knn.score(dataset_test)
    print(f'The rmse of the model is: {score}')