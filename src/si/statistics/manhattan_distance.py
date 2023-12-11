import numpy as np

# Exercise 4

def manhattan_distance (x:np.array, y:np.array):
    """
    Computes the Manhattan distance between a single sample and multiple samples.

    Parameters
    ----------
    x: np.array
        A single sample
    y: np.array
        Multiple samples.

    Returns
    -------
    np.array
        The manhattan distance between x and y
    """
    return np.sum(np.abs(x-y), axis=1)

