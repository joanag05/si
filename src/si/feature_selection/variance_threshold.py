import numpy as np

from si.data.dataset import Dataset


class VarianceThreshold:
    """
    Variance Threshold feature selection.
    Features with a training-set variance lower than this threshold will be removed from the dataset.

    Parameters
    ----------
    threshold: float
        The threshold value to use for feature selection. Features with a
        training-set variance lower than this threshold will be removed.

    Attributes
    ----------
    variance: array-like, shape (n_features,)
        The variance of each feature.
    """

    def __init__(self, threshold: float = 0.0):
        """
        Variance Threshold feature selection.
        Features with a training-set variance lower than this threshold will be removed from the dataset.

        Parameters
        ----------
        threshold: float
            The threshold value to use for feature selection. Features with a
            training-set variance lower than this threshold will be removed.
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        # parameters
        self.threshold = threshold

        # attributes
        self.variance = None

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Fit the VarianceThreshold model according to the given training data.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit.

        Returns
        -------
        self : object
        """
        self.variance = np.var(dataset.X, axis=0)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It removes all features whose variance does not meet the threshold.
        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        dataset: Dataset
        """
        X = dataset.X

        features_mask = self.variance > self.threshold
        X = X[:, features_mask]
        features = np.array(dataset.features)[features_mask]
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit to data, then transform it.
        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        dataset: Dataset
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset = Dataset.from_random(100, 10, 2)
    dataset.X[:, 0] = 0

    selector = VarianceThreshold()
    dataset = selector.fit_transform(dataset)
    print(dataset.features)
