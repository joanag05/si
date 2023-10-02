
from typing import Callable
import numpy as np
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from numpy.linalg import svd

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
        X = dataset.X - self.mean

        U, S, V = svd(X, full_matrices=False)

        self.components = V[:self.n_components]

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
        
        X = dataset.X - self.mean
        X = np.dotnp.dot(X, np.transpose(self.components))
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
        return self.fit(dataset).transform(dataset)
    
if __name__ == '__main__':
    from si.data.dataset import Dataset
    import pandas as pd
    from sklearn import preprocessing
    df = pd.read_csv("C:\Users\joana\OneDrive\Documentos\GitHub\si\datasets\iris\iris.csv")
    print(df.head())
    dataset_ = Dataset.from_dataframe(df, label='class')
    dataset_.X = preprocessing.scale(dataset_.X)

    n = 2
    pca = PCA(n)
    res = pca.fit_transform(dataset_)
    print(res.shape)

    print(pca.explained_variance)
    

