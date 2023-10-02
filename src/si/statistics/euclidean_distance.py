import numpy as np

def euclidean_distance(x: np.array, y: np.array) -> np.array:
    """
    Computes the euclidean distance between two arrays

    Parameters
    ----------
    x: np.array
        The first array
    y: np.array
        The second array

    Returns
    -------
    np.array
        The euclidean distance between x and y
    """
    return np.sqrt(np.sum((x-y)**2, axis=1))



