import numpy as np


def manhattan_distance (x:np.array, y:np.array):
    """
    Computes the manhattan distance between two arrays

    Parameters
    ----------
    x: np.array
        The first array
    y: np.array
        The second array

    Returns
    -------
    np.array
        The manhattan distance between x and y
    """
    return np.sum(np.abs(x-y), axis=1)

