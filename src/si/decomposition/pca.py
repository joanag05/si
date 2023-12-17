
from typing import Callable
import numpy as np
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from numpy.linalg import svd

# Exercise 6
class PCA:
    """
    Principal Component Analysis (PCA) implementation.

    Parameters:
    -----------
    n_components : int
        Number of components to keep.
    Attributes:
    -----------
    n_components : int
        Number of components to keep.
    eigenvectors : None
        The eigenvectors of the covariance matrix.
    eigenvalues : None
        The eigenvalues of the covariance matrix.
    """
    def __init__(self, n_components: int):
        """
        Initializes the PCA algorithm.

        Parameters:
        -----------
        n_components : int
            Number of components to keep.
        """
        self.n_components = n_components
        self.mean = None 
        self.components = None # eigenvectors
        self.explained_variance = None  # variance explained by each of the selected components

    def fit(self, dataset: Dataset) -> 'PCA':
        """
        Fits the PCA model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        
        Returns
        -------
        self : object
        """
        self.mean = np.mean(dataset.X, axis=0)

        # substraction the mean from the dataset
        X = dataset.X - self.mean

        # calculate de svd

        U, S, V = svd(X, full_matrices=False)

        self.components = V[:self.n_components]
        
        # EV = S2/(n-1) 

        self.explained_variance = S[:self.n_components] ** 2 / (len(X) - 1)

        return self
    
    def transform (self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset by selecting the k best features

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform
        
        Returns
        -------
        Dataset
        """
        if self.components is None:
            raise ValueError("Model not fitted")
        
        # center the data
        
        X = dataset.X - self.mean
        
        # calculate  the reduced X
        
        X = np.dot(X, np.transpose(self.components))
        return X
    

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits the PCA model to the dataset and transforms it

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to and transform
        
        Returns
        -------
        Dataset
        """
        self.fit(dataset)
        return self.transform(dataset)

# Testing PCA
   
if __name__ == '__main__':
    
    from si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                    [0, 1, 4, 3],
                                    [0, 1, 1, 3]]),
                        y=np.array([0, 1, 0]),
                        features=["f1", "f2", "f3", "f4"],
                        label="y")
        
    pca = PCA(n_components=3)
    pca.fit_transform(dataset)
    print("Variance", pca.explained_variance)
    print()
    print("Componentes", pca.components)
    print()
    print("Media", pca.mean)
    print()
    print(dataset.X)
    print()
    print("Fit_tranform", pca.fit_transform(dataset))



    

