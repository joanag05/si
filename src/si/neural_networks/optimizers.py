from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.
        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.
        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.
        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.
        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.
        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient

#Exercise 15 
class Adam(Optimizer):
    """
    Adam optimizer implementation.

    Args:
        learning_rate (float): The learning rate for the optimizer.
        beta_1 (float, optional): The exponential decay rate for the first moment estimates. Defaults to 0.9.
        beta_2 (float, optional): The exponential decay rate for the second moment estimates. Defaults to 0.999.
        epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-8.
    """

    def __init__(self, learning_rate: float, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        
        # arguments
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # estimated parameters
        self.m = None
        self.v = None
        self.t = 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Perform an update step using the Adam optimizer.

        Args:
            w (np.ndarray): The weights to be updated.
            grad_loss_w (np.ndarray): The gradient of the loss function with respect to the weights.

        Returns:
            np.ndarray: The updated weights.
        """
        # Initialize m and v if they are None

        if self.m is None:
            self.m = np.zeros(np.shape(w))
        if self.v is None:
            self.v = np.zeros(np.shape(w))

        # Update m, v and t

        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)

        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
#test Adam class

if __name__ == '__main__':
    import numpy as np
    #test the optimizer
    adam = Adam(learning_rate=0.001)
    w = np.array([1, 2, 3])
    grad_loss_w = np.array([1, 2, 3])
    print(adam.update(w, grad_loss_w))









    