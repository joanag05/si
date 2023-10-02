from typing_extensions import Self
import numpy as np
import pandas as pd

from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest:
    """
    Select features according to the k highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.
    
    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    k: int, default=10
        Number of top features to select.
    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """

    def __init__(self, score_func: Callable = f_classification, k: int = 10):
        """
        SelectKBest selects the k best features according to a scoring function.

        Parameters
        ----------
        score_func: Callable
            The scoring function
        k: int
            The number of features to select
        """
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        Fits the SelectKBest model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        
        Returns
        -------
        self : object
        """
        self.F, self.p = self.score_func(dataset)
        return self
    
    def transform(self, dataset: Dataset) -> Dataset:
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
        if self.F is None or self.p is None:
            raise ValueError("Model not fitted")
        
        index = np.argsort(self.F)[::-1][:self.k]
        features = np.array(dataset.features[index])
        return Dataset(features, dataset.y, dataset.classes, dataset.features_names[index])
    
    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits the SelectKBest model to the dataset and transforms it

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to and transform
        
        Returns
        -------
        Dataset
        """
        return self.fit(dataset).transform(dataset)
