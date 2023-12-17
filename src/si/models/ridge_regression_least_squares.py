import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse

import numpy as np
from si.data.dataset import Dataset

# Exercise 8

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

    def __init__(self, l2_penalty: float = 1, scale: bool = True):

        self.l2_penalty = l2_penalty
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
        # scale the input features if needed

        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # add the intercept term to X (add colums of 1s to the FIRST column of X)

        X = np.c_[np.ones(X.shape[0]), X]

        # compute the model parameters (l2_penalty * identity matrix)
        penalty_m = np.eye(X.shape[1]) * self.l2_penalty

        # set the intercept term to 0

        penalty_m[0, 0] = 0

        # compute the model parameters theta and theta_zero

        theta_vector = np.linalg.inv(X.T.dot(X) + penalty_m).dot(X.T).dot(dataset.y)

        # set the intercept term to theta_zero and the rest to theta
        self.theta_zero = theta_vector[0]
        self.theta = theta_vector[1:]

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
        # scale the input features if needed
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # add the intercept term (add colums of 1s to the left of X)
        
        X = np.c_[np.ones(X.shape[0]), X]
        
        # compute the predictions

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
        

        return mse(dataset.y, self.predict(dataset))

# This is how you can test it against sklearn to check if everything is fine

if __name__ == '__main__':
    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegressionLeastSquares()
    model.fit(dataset_)
    print(model.theta)
    print(model.theta_zero)

    # compute the score
    print(model.score(dataset_))

    # compare with sklearn
    from sklearn.linear_model import Ridge
    model = Ridge()
    # scale data
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    print(model.coef_) # should be the same as theta
    print(model.intercept_) # should be the same as theta_zero
    print(mse(dataset_.y, model.predict(X)))