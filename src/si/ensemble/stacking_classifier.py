import numpy as np
from typing import List
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

#Exercise 10

class StackingClassifier:
    def __init__(self, models: list, final_model):
        """
        Initialize the StackingClassifier with a list of models and a final model.
        :param models: List of models to be stacked.
        :param final_model: Final model to make predictions.
        """
        # parameters
        self.models = models
        self.final_model = final_model
    

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the models to the dataset.
        :param dataset: Dataset object to fit the models to.
        :return: self: StackingClassifier
        """
        # trains the models
        for model in self.models:
            model.fit(dataset)

        # gets the models predictions
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # trains the final model
        self.final_model.fit(Dataset(np.array(predictions).T, dataset.y))

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Computes the prevision of all the models and returns the final model prediction.
        :param dataset: Dataset object to predict the labels of.
        :return: the final model prediction
        """
        # gets the intial model predictions
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # gets the final model prediction
        y_pred = self.final_model.predict(Dataset(np.array(predictions).T, dataset.y))

        return y_pred

    def score(self, dataset: Dataset) -> float:
        """
        Calculates the accuracy of the model.
        :return: Accuracy of the model.
        """
        y_pred = self.predict(dataset)
        score = accuracy(dataset.y, y_pred)

        return score
    
if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import stratified_train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    from si.models.decision_tree_classifier import DecisionTreeClassifier

    data = "C:\\Users\\joana\\OneDrive\\Documentos\\GitHub\\si\\datasets\\breast_bin\\breast-bin.csv"  
    breast=read_csv(data, sep=",",features=True,label=True)
    train_data, test_data = stratified_train_test_split(breast, test_size=0.20, random_state=42)

    #knnregressor
    knn = KNNClassifier(k=3)
    
    #logistic regression
    LG=LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    #decisiontreee
    DT=DecisionTreeClassifier(min_sample_split=3, max_depth=3, mode='gini')

    # Final Model (Choose a different model as the final model)
    final_model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    # Initialize StackingClassifier
    models = [knn, LG, DT]
    exercise = StackingClassifier(models, final_model)
    exercise.fit(train_data)
    print(exercise.score(test_data))

    # compare with sklearn
    from sklearn.ensemble import StackingClassifier as SklearnStackingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)

    # Logistic Regression
    LG = LogisticRegression(penalty='l2', C=1, solver='lbfgs', max_iter=1000)

    # Decision Tree
    DT = DecisionTreeClassifier(min_samples_split=3, max_depth=3, criterion='gini')

    # Final Model (Choose a different model as the final model)
    final_model = LogisticRegression(penalty='l2', C=1, solver='lbfgs', max_iter=1000)

    # Initialize StackingClassifier
    models = [('knn', knn), ('LG', LG), ('DT', DT)]
    sklearn = SklearnStackingClassifier(estimators=models, final_estimator=final_model)
    sklearn.fit(train_data.X, train_data.y)
    y_pred = sklearn.predict(test_data.X)
    print(accuracy_score(test_data.y, y_pred))

    