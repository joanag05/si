import numpy as np
from typing import List, Tuple
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class RandomForestClassifier:
    """
    Random Forest Classifier.

    Parameters:
    - n_estimators: int, optional (default=100)
        The number of trees in the forest.
    - max_features: int, optional (default=None)
        The number of features to consider when looking for the best split.
        If None, then max_features=sqrt(n_features).
    - min_sample_split: int, optional (default=2)
        The minimum number of samples required to split an internal node.
    - max_depth: int, optional (default=10)
        The maximum depth of the tree.
    - mode: str, optional (default='gini')
        The function to measure the quality of a split.
        Supported criteria are 'gini' for the Gini impurity and 'entropy' for the information gain.
    - seed: int, optional (default=42)
        The random seed for reproducible results.

    Attributes:
    - trees: list
        List of tuples containing the selected features and the corresponding decision tree.

    Methods:
    - fit(dataset: Dataset) -> RandomForestClassifier:
        Fit the random forest classifier to the given dataset.
        Returns the fitted RandomForestClassifier object.
    - predict(dataset: Dataset) -> np.ndarray:
        Predict the class labels for the given dataset.
        Returns an array of predicted class labels.
    - score(dataset: Dataset) -> float:
        Calculate the accuracy score of the classifier on the given dataset.
        Returns the accuracy score.

    """

    def __init__(self, n_estimators: int, max_features: int = None, min_sample_split: int = 2, max_depth: int = 10, mode: str = 'gini', seed: int = 42) -> None:
        #parameters
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        #estimated parameters
        self.trees = []

    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Fit the random forest classifier to the given dataset.

        Parameters:
        - dataset: Dataset
            The training dataset.

        Returns:
        - RandomForestClassifier
            The fitted RandomForestClassifier object.
        """

        # set random seed
        if self.seed is not None:
            np.random.seed(self.seed)   

        # get number of samples and features
        n_samples, n_features = dataset.shape()

        # set max_features if not set
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))


        for i in range(self.n_estimators):
            # bootstrap dataset
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            features = np.random.choice(n_features, self.max_features, replace=False)
            bootstrap_dataset = Dataset(dataset.X[bootstrap_indices, :][:, features], dataset.y[bootstrap_indices])
            # create tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_sample_split=self.min_sample_split, mode=self.mode)
            # fit tree
            tree.fit(bootstrap_dataset)
            # save features and tree as tuple
            self.trees.append((features, tree))
        return self
    
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the class labels for the given dataset.

        Parameters:
        - dataset: Dataset
            The dataset to predict on.

        Returns:
        - np.ndarray
            An array of predicted class labels.
        """
        #  predict for each tree
        
        y_pred = [None] * self.n_estimators
        for i, (features_idx, tree) in enumerate(self.trees):
            y_pred[i] = tree.predict(Dataset(dataset.X[:, features_idx], dataset.y))

        # most common prediction for each sample
        
        most_common = []
        for z in zip(*y_pred):
            most_common.append(max(set(z), key=z.count))

        return np.array(most_common)

    def score(self, dataset: Dataset) -> float:
        """
        Calculate the accuracy score of the classifier on the given dataset.

        Parameters:
        - dataset: Dataset
            The dataset to calculate the accuracy score on.

        Returns:
        - float
            The accuracy score.
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)



if __name__ == '__main__':

    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split
    from si.metrics.accuracy import accuracy
    
    # read iris dataset
    data = read_csv("\\Users\\joana\\OneDrive\\Documentos\\GitHub\\si\\datasets\\iris\\iris.csv", sep=',', features=True, label=True)

    # split dataset into train and test set
    train_set, test_set = train_test_split(data, test_size=0.33, random_state=42)

    # create random forest classifier
    clf = RandomForestClassifier(n_estimators=10, max_depth=5, min_sample_split=2, mode='gini')

    # fit classifier to train set
    clf.fit(train_set)

    # predict on test set
    y_pred = clf.predict(test_set)

    # calculate accuracy score
    score = accuracy(test_set.y, y_pred)
    print('Accuracy score: ', score)

    # compare to sklearn

    from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
    
    clf1 = skRandomForestClassifier(n_estimators=10, max_depth=5, min_samples_split=2)
    clf1.fit(train_set.X, train_set.y)
    y_pred = clf1.predict(test_set.X)
    score = accuracy(test_set.y, clf1.predict(test_set.X))
    print('Accuracy score sklearn: ', score)






