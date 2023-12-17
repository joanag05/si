import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class CategoricalNB:
    """
    Categorical Naive Bayes classifier.

    Parameters:
    - smoothing (float): The smoothing parameter for Laplace smoothing. Default is 1.0.

    Attributes:
    - smoothing (float): The smoothing parameter for Laplace smoothing.
    - class_prior (ndarray): The prior probabilities of each class.
    - feature_probs (ndarray): The conditional probabilities of each feature given each class.
    """

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_prior = None
        self.feature_probs = None

    def fit(self, dataset: Dataset) -> 'CategoricalNB':
        """
        Fit the CategoricalNB model to the given dataset.

        Parameters:
        - dataset (Dataset): The training dataset.

        Returns:
        - self (CategoricalNB): The fitted CategoricalNB model.
        """
        n_samples, n_features = dataset.shape()
        n_classes = len(dataset.get_classes())

        self.class_prior = np.zeros(n_classes)
        self.feature_probs = np.zeros((n_classes, n_features))

        class_counts = np.zeros(n_classes)
        feature_counts = np.zeros((n_classes, n_features))

        # apllying laplace smoothing to avoid zero probabilities

        for i in range(n_classes):
            class_samples = dataset.y == i
            class_counts[i] = np.sum(class_samples)
            feature_counts[i] = np.sum(dataset.X[class_samples], axis=0)

        # compute the class prior and feature probabilities
        self.class_prior = class_counts / n_samples
        self.feature_probs = (feature_counts + self.smoothing) / (class_counts.reshape(-1, 1) + self.smoothing * n_features)

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the class labels for the given dataset.

        Parameters:
        - dataset (Dataset): The dataset to make predictions on.

        Returns:
        - predictions (ndarray): The predicted class labels.
        """
        predictions = []

        # probability of each class for each sample
        
        for sample in dataset.X:
            class_probs = []
            for i in range(len(self.class_prior)):
                class_probs.append(
                    np.prod(
                        sample * self.feature_probs[i] + (1 - sample) * (1 - self.feature_probs[i])
                    ) * self.class_prior[i]
                )
            # class with the highest probability as the prediction class
            predictions.append(np.argmax(class_probs))
        return np.array(predictions)

    def score(self, dataset: Dataset) -> float:
        """
        Calculate the accuracy score of the model on the given dataset.

        Parameters:
        - dataset (Dataset): The dataset to evaluate the model on.

        Returns:
        - score (float): The accuracy score.
        """
        predictions = self.predict(dataset)
        true_labels = dataset.y
        return accuracy(true_labels, predictions)

    


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





