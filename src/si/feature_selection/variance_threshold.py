from pyexpat import features
from statistics import variance
import numpy as np
import pandas as pd

import numpy as np

from si.data.dataset import Dataset


class VarianceThreshold:
    def __init__(self, threshold: float = 0.0) -> None:
        """
        VarianceThreshold removes features with low variance.

        Parameters
        ----------
        threshold: float
            The variance threshold
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
        

        self.threshold = threshold
        self.variance = None # para já está vazio porque nós não temos o dataset aqui


    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Fits the VarianceThreshold model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        
        Returns
        -------
        self : object
        """
        self.variance = np.var(dataset.X, axis=0) # também se podia usar o método get_variance() do dataset
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset by removing features with low variance

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform
        
        Returns
        -------
        Dataset
        """
        if self.variance is None:
            raise ValueError("Model not fitted")

        mask = self.variance > self.threshold
        X = dataset.X[:, mask]
        features = np.array(dataset.features)[mask].tolist() # para ficar com o mesmo tipo de dados que o dataset original


        return Dataset(X, dataset.y, features=features, label=dataset.label)  # o label é o mesmo do dataset original 
    
    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits the VarianceThreshold model to the dataset and transforms it

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to and transform
        
        Returns
        -------
        Dataset
        """
        return self.fit(dataset).transform(dataset)
    
    
