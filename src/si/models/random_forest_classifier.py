import numpy as np
import numpy as np
from collections import Counter
from si.model_selection.split import train_test_split
from models.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:

    def __init__(self, n_estimators, max_features, min_sample_split,max_depth, mode, seed):

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []
    
    def fit(self, X, y):
        if self.seed:
            np.random.seed(self.seed)

        n_samples, n_features = X.shape
        self.max_features = int(np.sqrt(n_features)) if self.max_features is None else self.max_features

        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            features = np.random.choice(n_features, self.max_features, replace=False)

            X_bootstrap = X[idx][:, features]
            y_bootstrap = y[idx]

            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, criterion=self.mode)
            tree.fit(X_bootstrap, y_bootstrap)

            self.trees.append((features, tree))

        return self
        

    def predict(self, X):
        predictions = []
        for features, tree in self.trees:
            X_subset = X[:, features]
            tree_pred = tree.predict(X_subset)
            predictions.append(tree_pred)

        predictions = np.array(predictions)
        ensemble_preds = np.array([Counter(sample).most_common(1)[0][0] for sample in predictions.T])
        return ensemble_preds

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)
        return accuracy
    
if __name__ == '__main__':
    from si.io.csv_file import read_csv

    df = read_csv(r'C:\Users\joana\OneDrive\Documentos\GitHub\si\datasets\iris\iris.csv', label=True, features=True)
    
X_train, X_test, y_train, y_test = train_test_split(df, iris.target, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_features=None, min_samples_split=2, max_depth=float('inf'), mode='gini', seed=42)
rf.fit(X_train, y_train)

test_score = rf.score(X_test, y_test)
print(f"Test set accuracy: {test_score:.4f}")



        