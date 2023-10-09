
import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float :
    """
    Calculates the accuracy of the model's predictions.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.

    Returns:
        float: The accuracy of the model's predictions.
    """
    return np.sum(y_true == y_pred) / len(y_true)