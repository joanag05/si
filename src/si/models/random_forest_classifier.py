from typing import List, Tuple
import numpy as np
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier:
    """
    Random Forest Classifier implementation.

    Parameters:
    - n_estimators: int, optional (default=100)
        The number of trees in the forest.
    - max_features: int, optional (default=None)
        The number of features to consider when looking for the best split.
    - min_sample_split: int, optional (default=2)
        The minimum number of samples required to split an internal node.
    - max_depth: int, optional (default=10)
        The maximum depth of the tree.
    - mode: str, optional (default='gini')
        The criterion used to evaluate the quality of a split.
    - seed: int, optional (default=42)
        The random seed for reproducibility.

    Attributes:
    - n_estimators: int
        The number of trees in the forest.
    - max_features: int
        The number of features to consider when looking for the best split.
    - min_sample_split: int
        The minimum number of samples required to split an internal node.
    - max_depth: int
        The maximum depth of the tree.
    - mode: str
        The criterion used to evaluate the quality of a split.
    - seed: int
        The random seed for reproducibility.
    - trees: list
        The list of decision trees in the random forest.
    """

    def __init__(self, n_estimators: int, max_features: int = None, min_sample_split: int = 2, max_depth: int = 10, mode: str = 'gini', seed: int = 42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Fit the random forest classifier to the given dataset.

        Parameters:
        - dataset: Dataset
            The training dataset.

        Returns:
        - RandomForestClassifier
            The fitted random forest classifier.
        """
        

        if self.seed is not None: # if seed is not None, set the random seed
            np.random.seed(self.seed) # set the random seed for reproducibility


        n_samples, n_features = dataset.shape()

        self.max_features = int(np.sqrt(n_features)) if self.max_features is None else self.max_features

        # train each tree in the forest

        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)
            tree_dataset = Dataset(dataset.X[bootstrap_indices, :][:, feature_indices], dataset.y[bootstrap_indices])
            
            tree = DecisionTreeClassifier(min_sample_split=self.min_sample_split, max_depth=self.max_depth,
                                          mode=self.mode)
            tree.fit(tree_dataset)
            
            self.trees.append((feature_indices, tree))

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the class labels for the given dataset.

        Parameters:
        - dataset: Dataset
            The dataset to make predictions on.

        Returns:
        - np.ndarray
            The predicted class labels.
        """
        predictions = []
        
        for features, tree in self.trees:
            tree_dataset = Dataset(dataset.X[:, features], dataset.features, label=dataset.label)
            tree_pred = tree.predict(tree_dataset)
            predictions.append(tree_pred)

        predictions = np.array(predictions).T
        final_predictions = np.array([np.argmax(np.bincount(pred)) for pred in predictions])
        return final_predictions

    def score(self, dataset: Dataset) -> float:
        """
        Calculate the accuracy score of the random forest classifier on the given dataset.

        Parameters:
        - dataset: Dataset
            The dataset to calculate the accuracy score on.

        Returns:
        - float
            The accuracy score.
        """
        y_pred = self.predict(dataset)
        accuracy = np.mean(y_pred == dataset.y)
        return accuracy


if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import stratified_train_test_split

    data =  "C:\\Users\\joana\\OneDrive\\Documentos\\GitHub\\si\\datasets\\iris\\iris.csv"
    data = read_csv(data, sep=",",features=True,label=True)
   
    train_data, test_data = stratified_train_test_split(data, test_size=0.2, random_state=42)


    random_forest = RandomForestClassifier(n_estimators=10, max_features=None, min_sample_split=2,
                                       max_depth=None, mode='gini', seed=42)
    random_forest.fit(train_data)

    score = random_forest.score(test_data)
    print(f"Accuracy on test set: {score}")
    score = random_forest.score(train_data)
    print(f"Accuracy on train set: {score}")