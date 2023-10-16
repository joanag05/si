import numpy as np

def mse(y_true : np.ndarray, Y_pred: np.ndarray) -> float:
    """
    Calculates the mean squared error between the true and predicted values.
    
    Args:
    y_true (np.ndarray): Array of true values.
    Y_pred (np.ndarray): Array of predicted values.
    
    Returns:
    float: Mean squared error between the true and predicted values.
    """
    return np.sum((y_true-Y_pred ) ** 2) / len(y_true * 2)