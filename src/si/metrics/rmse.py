import numpy as np

# Exercise 7.1
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the mean squared error for the y_pred variable.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    mse: float
        The mean squared error of the model
    """
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true)) #  raiz quadrada do mse


