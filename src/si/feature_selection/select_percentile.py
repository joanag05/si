from typing import Callable

import numpy as np
import pandas as pd


import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile:
    """
    SelectPercentile selects the k best features according to a scoring function and a given percentile.

    Parameters
    ----------
    scorefunc: Callable
        The scoring function
    percentile: float
        The percentile of features to select
    """

    def __init__(self, scorefunc: Callable = f_classification, percentile: float = 0.5):
            """
            Initializes a SelectPercentile instance with the given score function and percentile.

            Args:
                scorefunc (Callable): The score function to use for feature selection. Defaults to f_classification.
                percentile (float): The percentile of features to select. Defaults to 0.5.
            """
            self.scorefunc = scorefunc
            self.percentile = percentile
            self.F = None
            self.p = None

    def fit (self, dataset: Dataset) -> 'SelectPercentile':
        """
        Fits the SelectPercentile model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        
        Returns
        -------
        self : object
        """
        self.F, self.p = self.scorefunc(dataset)
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

        if self.F is None:
            raise ValueError("Model not fitted")
        
        total = len(list(dataset.features))
        k = int(self.percentile * total)
        if k == 0 : k = 1
        print(k)
        index = np.argsort(self.F)[-k:]
        features = np.array(dataset.features[index])
        return Dataset(X=dataset.X[:, index], y=dataset.y, features=list(features), label=dataset.label)
    
    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits the SelectPercentile model to the dataset and transforms it

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit and transform
        
        Returns
        -------
        Dataset
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.io.csv_file import read_csv

    df = read_csv(r'C:\Users\joana\OneDrive\Documentos\GitHub\si\datasets\iris\iris.csv', label=True, features=True)

    selector = SelectPercentile(percentile=1.1)
    selector = selector.fit(df)
    dataset = selector.transform(df)
    print(dataset.features)   



         

