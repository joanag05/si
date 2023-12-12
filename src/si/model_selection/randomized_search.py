from typing import Dict, Tuple, Callable, Union

import numpy as np

from si.data.dataset import Dataset
from si.model_selection.cross_validation import k_fold_cross_validation


# Exercise 11


import numpy as np

def randomized_search_cv(model, dataset, hyperparameter_grid, scoring=None, cv=5, n_iter=None, random_state=None):
    """
    Implements a parameter optimization strategy with cross validation using a number of random combinations selected from a distribution possible hyperparameters.
    More efficient and useful in large datasets.
    Makes n random combinations with the hyperparameters and may not give the optimal solution, but it gives a good combination in less time.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    hyperparameter_grid: Dict[str, Tuple]
        The hyperparameter grid to use.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.
    n_iter:int
        Number of hyperparameter random combinations to test.
    random_state: int, optional
        Seed for random state to ensure reproducibility.

    Returns
    -------
    results: Dict[str, Any]
        The results of the grid search cross validation. Includes the scores, hyperparameters,
        best hyperparameters, and best score.
    """
    for parameter in hyperparameter_grid:
            if not hasattr(model, parameter):
                raise AttributeError(f"Model {model} does not have parameter {parameter}.") 

    results = {'scores': [], 'hyperparameters': []}

    for _ in range(n_iter):
        parameters = {}
        for key, values in hyperparameter_grid.items():
            # Choose a different random value for each hyperparameter
            parameters[key] = np.random.choice(values)

        # Set the hyperparameters in the model
        for key, value in parameters.items():
            setattr(model, key, value)

        # Cross validate the model
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # Record the score and hyperparameters for this iteration
        results['scores'].append(np.mean(score))
        results['hyperparameters'].append(parameters)

    # Identify the best score and best hyperparameters
    best_index = np.argmax(results['scores'])
    results['best_hyperparameters'] = results['hyperparameters'][best_index]
    results['best_score'] = np.max(results['scores'])

    return results

if __name__ == '__main__':
    # import dataset
    from si.models.logistic_regression import LogisticRegression

    num_samples = 600
    num_features = 100
    num_classes = 2

    # random data
    X = np.random.rand(num_samples, num_features)  
    y = np.random.randint(0, num_classes, size=num_samples)  # classe aleatórios

    dataset_ = Dataset(X=X, y=y)

    #  features and class name
    dataset_.features = ["feature_" + str(i) for i in range(num_features)]
    dataset_.label = "class_label"

    # initialize the Logistic Regression model
    knn = LogisticRegression()

    # parameter grid
    parameter_grid_ = {
        'l2_penalty': (1, 10),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 2000)
    }

    # cross validate the model
    results_ = randomized_search_cv(knn,
                              dataset_,
                              hyperparameter_grid=parameter_grid_,
                              cv=3,
                              n_iter=8)

    # print the results
    print(results_)

    # get the best hyperparameters
    best_hyperparameters = results_['best_hyperparameters']
    print(f"Best hyperparameters: {best_hyperparameters}")

    # get the best score
    best_score = results_['best_score']
    print(f"Best score: {best_score}")