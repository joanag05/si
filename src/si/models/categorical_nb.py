import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class CategoricalNB:
    def __init__(self, smoothing=1.0):
        """
        Custom implementation of Categorical Naive Bayes for discrete/categorical data.

        Parameters:
        -----------
        smoothing : float, default=1.0
            Laplace smoothing to avoid zero probabilities.
        """
        self.smoothing = smoothing
        self.class_prior = None
        self.feature_probs = None
        self.num_classes = 0

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

        for i in range(n_classes):
            class_counts[i] = np.sum(y == i)

        for i in range(n_classes):
            for j in range(n_features):
                feature_counts[i, j] = np.sum(X[y == i, j])

        class_prior = class_counts / n_samples

        feature_counts += self.smoothing
        class_counts += self.smoothing * n_features

        feature_probs = feature_counts / class_counts[:, np.newaxis]

        self.class_prior = class_prior
        self.feature_probs = feature_probs
        self.num_classes = n_classes

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
        n_classes = self.num_classes

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
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)
    
    


if __name__ == "__main__":

    
    X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]])
    y = np.array([0, 1, 0, 1])
  
    dataset = Dataset(X, y)

    model = CategoricalNB(smoothing=1)
    model.fit(dataset)

    predictions = model.predict(dataset)
    error = model.score(dataset)

    print("Predicted classes:", predictions)
    print("Classification error:", model.score(dataset))

    # test the model on a new sample
    new_sample = np.array([[1, 0, 0]])
    new_dataset = Dataset(new_sample)
    new_predictions = model.predict(new_dataset)

    print("New sample:", new_sample)
    print("Predicted class:", new_predictions[0])
    print("Classification error:", model.score(dataset))

    #show me using sklearn
    print("Using sklearn:")
    print("--------------")
    from sklearn.naive_bayes import CategoricalNB as skCategoricalNB

    sk_model = skCategoricalNB(alpha=1)
    sk_model.fit(X, y)

    sk_predictions = sk_model.predict(X)
    sk_error = sk_model.score(X, y)

    print("Predicted classes:", sk_predictions)
    print("Classification error:", sk_error)





