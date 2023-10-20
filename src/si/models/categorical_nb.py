
import numpy as np

from si.data.dataset import Dataset
class CategoricalNB:
    """
    A Naive Bayes classifier for categorical data.

    Parameters:
    -----------
    smoothing : float, default=1.0
        The smoothing parameter to apply to the probabilities.

    Attributes:
    -----------
    class_prior : array-like of shape (n_classes,)
        The prior probabilities of the classes.
    feature_probs : array-like of shape (n_classes, n_features)
        The conditional probabilities of each feature given each class.
    """
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.class_prior = None
        self.feature_probs = None

    def fit(self, dataset):
        """
        Fit the model to the given dataset.

        Parameters:
        -----------
        dataset : Dataset
            The dataset to fit the model to.
        """
        X, y = dataset.X, dataset.y
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        class_counts = np.zeros(n_classes)
        feature_counts = np.zeros((n_classes, n_features))
        class_prior = np.zeros(n_classes)

        # Calculate class counts
        for i in range(n_classes):
            class_counts[i] = np.sum(y == i)

        # Calculate feature counts
        for i in range(n_classes):
            for j in range(n_features):
                feature_counts[i, j] = np.sum(X[y == i, j])

        # Calculate class prior probabilities
        class_prior = class_counts / n_samples

        # Apply Laplace smoothing to feature_counts and class_counts
        feature_counts += self.smoothing
        class_counts += self.smoothing * n_features

        # Calculate feature probabilities
        feature_probs = feature_counts / class_counts[:, np.newaxis]

        self.class_prior = class_prior
        self.feature_probs = feature_probs

        return self
    
    def predict(self, dataset):
        """
        Predict the class labels for a given set of samples.

        Parameters:
        -----------
        dataset : Dataset
            The dataset to make predictions on.

        Returns:
        --------
        predictions : array-like of shape (n_samples,)
            An array of predicted class labels.
        """
        X = dataset.X
        n_samples, n_features = X.shape
        n_classes, _ = self.feature_probs.shape

        predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            sample = X[i]
            class_probs = np.zeros(n_classes)

            for c in range(n_classes):
                class_probs[c] = np.prod(sample * self.feature_probs[c] + (1 - sample) * (1 - self.feature_probs[c])) * self.class_prior[c]

            predictions[i] = np.argmax(class_probs)

        return predictions

    def score(self, dataset):
        """
        Calculate the accuracy between actual values and predictions.

        Parameters:
        -----------
        dataset : Dataset
            The dataset to evaluate the model.

        Returns:
        --------
        error : float
            The error between the actual values and predictions (1.0 - accuracy).
        """
        y_true = dataset.y
        y_pred = self.predict(dataset)

        accuracy = np.mean(y_true == y_pred)
        error = 1.0 - accuracy

        return error

import numpy as np



X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]])
y = np.array([0, 1, 0, 1])

from si.data.dataset import Dataset  
dataset = Dataset(X, y)


model = CategoricalNB(smoothing=1)

model.fit(dataset)


predictions = model.predict(dataset)

error = model.score(dataset)


print("Predicted classes:", predictions)
print("Classification error:", error)