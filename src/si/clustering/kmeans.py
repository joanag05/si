from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance


from typing import Callable
import numpy as np
from si.data import Dataset
from si.statistics import euclidean_distance

class KMeans:
    """
    KMeans clustering algorithm implementation.

    Parameters:
    -----------
    k : int
        The number of clusters to form.
    max_iter : int, optional (default=1000)
        The maximum number of iterations for the algorithm.
    distance : Callable, optional (default=euclidean_distance)
        The distance function to use for calculating distances between points.

    Attributes:
    -----------
    k : int
        The number of clusters to form.
    max_iter : int
        The maximum number of iterations for the algorithm.
    distance : Callable
        The distance function to use for calculating distances between points.
    centroids : None
        The centroids of the clusters.
    clusters : None
        The clusters formed by the algorithm.
    """
    def __init__(self, k: int , max_iter: int = 1000 , distance: Callable = euclidean_distance):
        """
        Initializes the KMeans clustering algorithm.

        Parameters:
        -----------
        k : int
            The number of clusters to form.
        max_iter : int, optional (default=1000)
            The maximum number of iterations for the algorithm.
        distance : Callable, optional (default=euclidean_distance)
            The distance function to use for calculating distances between points.
        """
        self.k = k
        self.max_iter = max_iter
        self.distance = distance
        self.centroids = None
        self.clusters = None

    
    def _init_centroids(self, dataset: Dataset):
        """
        Initializes the centroids of the clusters.

        Parameters:
        -----------
        dataset : Dataset
            The dataset to initialize the centroids from.
        """
        seeds = np.random.permutation(dataset.X.shape[0])[:self.k]
        self.centroids = dataset.X[seeds]

    def get_closes_centroid(self, sample: np.ndarray) -> np.ndarray:
        """
        Returns the index of the closest centroid to a given sample.

        Parameters:
        -----------
        sample : np.ndarray
            The sample to find the closest centroid to.

        Returns:
        --------
        np.ndarray
            The index of the closest centroid.
        """
        distances = self.distance(sample, self.centroids)
        return np.argmin(distances, axis=1)
        
    def fit(self, dataset: Dataset) -> 'KMeans':
        """
        Fits the KMeans model to the dataset.

        Parameters:
        -----------
        dataset : Dataset
            The dataset to fit the model to.

        Returns:
        --------
        KMeans
            The fitted KMeans model.
        """
        self._init_centroids(dataset) # initialize the centroids

        converged = False

        k = 0
        labels = np.zeros(dataset.X.shape[0])
        while not converged and k < self.max_iter:

            new_labels = np.apply_along_axis(self.get_closes_centroid, 1, dataset.X)

            centroids = []

            for i in range(self.k):
                centroids.append(np.mean(dataset.X[new_labels == i], axis=0)) # compute the new centroids

            self.centroids = np.array(centroids)

            converged = np.array_equal(labels, new_labels)

            labels = new_labels

            k += 1
        

        self.labels = labels

        return self
    
    def get_distances(self, sample:np.ndarray) -> np.ndarray:
            """
            Calculates the distances between a given sample and the centroids of the KMeans model.

            Args:
                sample (np.ndarray): The sample to calculate the distances from.

            Returns:
                np.ndarray: An array containing the distances between the sample and the centroids.
            """
            return self.distance(sample, self.centroids)

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset by assigning each sample to the closest cluster.

        Parameters:
        -----------
        dataset : Dataset
            The dataset to transform.

        Returns:
        --------
        Dataset
            The transformed dataset.
        """
        if self.centroids is None:
            raise ValueError("Model not fitted")

        centtroids_distance = np.apply_along_axis(self.get_closes_centroid, 1, dataset.X) # get the closest centroid for each sample

        return centtroids_distance
    
    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits the KMeans model to the dataset and transforms it.

        Parameters:
        -----------
        dataset : Dataset
            The dataset to fit the model to and transform.

        Returns:
        --------
        Dataset
            The transformed dataset.
        """
        return self.fit(dataset).transform(dataset)
    
    def predict (self, dataset: Dataset) -> Dataset:
        """
        Predicts the class of each sample in the dataset.

        Parameters:
        -----------
        dataset : Dataset
            The dataset to predict the classes of.

        Returns:
        --------
        Dataset
            The dataset with the predicted classes.
        """
        if self.centroids is None:
            raise ValueError("Model not fitted")
        
        labels = np.apply_along_axis(self.get_closes_centroid, 1, dataset.X)
        return labels
    
    def fit_predict (self, dataset: Dataset) -> Dataset:
        """
        Fits the KMeans model to the dataset and predicts the class of each sample.

        Parameters:
        -----------
        dataset : Dataset
            The dataset to fit the model to and predict the classes of.

        Returns:
        --------
        Dataset
            The dataset with the predicted classes.
        """
        return self.fit(dataset).predict(dataset)
    

if __name__ == '__main__':
    
    from si.data.dataset import Dataset
    dataset_ = Dataset.from_random(200, 10)

    k_ = 3
    kmeans = KMeans(k_)
    res = kmeans.fit_transform(dataset_)
    predictions = kmeans.predict(dataset_)
    print(res.shape)
    print(predictions.shape)
    print(predictions)
    
    
    
        

              

        
        