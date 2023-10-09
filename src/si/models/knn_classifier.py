
from typing import Callable, Union

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

from si.statistics.euclidean_distance import euclidean_distance
    
class KNNClassifier:
    """
        A k-nearest neighbors classifier.

        Parameters:
        -----------
        k : int, optional (default=3)
            The number of neighbors to consider when making a prediction.
        distance : callable, optional (default=euclidean_distance)
            The distance metric to use when calculating distances between points.

        Attributes:
        -----------
        k : int
            The number of neighbors to consider when making a prediction.
        distance : callable
            The distance metric to use when calculating distances between points.
        dataset : array-like, shape (n_samples, n_features)
            The training dataset.
        """
    def __init__(self, k: int = 1 , distance: Callable = euclidean_distance):

            self.k = k
            self.distance = distance

            self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNClassifier':
            """
            Fits the KNNClassifier model to the dataset

            Parameters
            ----------
            dataset: Dataset
                The dataset to fit the model to
            
            Returns
            -------
            self : object
            """
            self.dataset = dataset
            return self
    
    def get_closest_label(self, samples: np.ndarray) -> Union[int, str]:
        """
        Returns the label that appears the most among the k nearest neighbors

        Parameters
        ----------
        neighbors_labels: np.ndarray
            The labels of the k nearest neighbors

        Returns
        -------
        Union[int, float]
            The label that appears the most among the k nearest neighbors
        """

    
        distances = self.distance(samples, self.dataset.X)
        neighbors_labels = np.argsort(distances)[:self.k]
        nearest_neighbors_labels = self.dataset.y[neighbors_labels]
        unique_labels, counts = np.unique(nearest_neighbors_labels, return_counts=True)
        max_count_idx = np.argmax(counts)

        return unique_labels[max_count_idx]

    class KNNClassifier:
        def __init__(self, k: int):
            """
            Initializes a KNNClassifier object with a given value of k.

            Args:
            - k (int): The number of nearest neighbors to consider when making a prediction.
            """
            self.k = k

        def fit(self, dataset: Dataset):
            """
            Fits the KNNClassifier to a given dataset.

            Args:
            - dataset (Dataset): The dataset to fit the classifier to.
            """
            self.dataset = dataset

        def predict(self, dataset: Dataset) -> np.ndarray:
            """
            Predicts the labels of a given dataset using the KNNClassifier.

            Args:
            - dataset (Dataset): The dataset to predict labels for.
              Returns:
            - np.ndarray: An array of predicted labels for the given dataset.
            """
            return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
        
        def score(self, dataset : Dataset) -> float: 
            predictions = self.predict(dataset)
            return accuracy(dataset.y, predictions)
        
    if __name__ == '__main__':
    
        from si.data.dataset import Dataset
        from si.model_selection.split import train_test_split

    
        dataset_ = Dataset.from_random(600, 100, 2)
        dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

        knn = KNNClassifier(k=5)

    
        knn.fit(dataset_train)

        score = knn.score(dataset_test)
        print(f'The accuracy of the model is: {score}')

