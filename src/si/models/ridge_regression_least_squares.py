import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse

import numpy as np
from si.data import Dataset

class RidgeRegressionLeastSquares:
    """
    Ridge Regression Least Squares model implementation.

    Parameters
    ----------
    l2_penalty : float, default=1.0
        L2 penalty (regularization term) parameter.
    alpha : float, default=0.001
        Learning rate parameter.
    scale : bool, default=True
        Whether to scale the input features.

    Attributes
    ----------
    l2_penalty : float
        L2 penalty (regularization term) parameter.
    alpha : float
        Learning rate parameter.
    scale : bool
        Whether to scale the input features.
    theta : np.array
        Model parameters.
    theta_zero : float
        Intercept term.
    mean : np.array
        Mean of the input features (if scaled).
    std : np.array
        Standard deviation of the input features (if scaled).

    Methods
    -------
    fit(dataset)
        Fit the model to the input data.
    predict(dataset)
        Predict the output for the input data.
    score(dataset)
        Compute the R^2 score for the input data.
    """

    def __init__(self, l2_penalty=1.0, alpha: float = 0.001, scale=True):
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fit the model to the input data.

        Parameters
        ----------
        dataset : Dataset
            Input data.

        Returns
        -------
        self : RidgeRegressionLeastSquares
            Returns self.
        """
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = dataset.shape()
        
        # Add intercept term to X
        X = np.c_[np.ones(m), X]
        
        # Compute the (penalty term l2_penalty * identity matrix)
        penalty_matrix = self.l2_penalty * np.eye(n + 1)
        
        # Change the first position of the penalty matrix to 0
        penalty_matrix[0, 0] = 0
        
        # Compute the model parameters
        A = X.T.dot(X) + penalty_matrix
        b = X.T.dot(dataset.y)
        thetas = np.linalg.inv(A).dot(b)
        
        self.theta_zero = thetas[0]
        self.theta = thetas[1:]

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output for the input data.

        Parameters
        ----------
        dataset : Dataset
            Input data.

        Returns
        -------
        y_pred : np.array
            Predicted output.
        """
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m = dataset.shape()[0]
        X = np.c_[np.ones(m), X]
        y_pred = X.dot(np.r_[self.theta_zero, self.theta])

        return y_pred

    def score(self, dataset: Dataset) -> float:
        """
        Compute the R^2 score for the input data.

        Parameters
        ----------
        dataset : Dataset
            Input data.

        Returns
        -------
        score : float
            R^2 score.
        """
        y_pred = self.predict(dataset)
        ss_res = np.sum((dataset.y - y_pred) ** 2)
        ss_tot = np.sum((dataset.y - np.mean(dataset.y)) ** 2)
        score = 1 - ss_res / ss_tot

        return score

# This is how you can test it against sklearn to check if everything is fine
if __name__ == '__main__':
    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegressionLeastSquares(alpha=2.0)
    model.fit(dataset_)
    print(model.theta)
    print(model.theta_zero)

    # compute the score
    print(model.score(dataset_))

    # compare with sklearn
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=2.0)
    # scale data
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    print(model.coef_) # should be the same as theta
    print(model.intercept_) # should be the same as theta_zero
    print(mse(dataset_.y, model.predict(X)))