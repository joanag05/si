
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
import numpy as np

class CategoricalNB:
    def __init__(self, smoothing=1.0):

        self.smoothing = smoothing # Parameter to avoid probabilities of 0


        #Attributes

        self.class_prior = None
        self.feature_probs = None


    def fit(self, dataset):
        n_samples, n_features = dataset.X.shape
        self.class_prior = np.zeros(len(np.unique(dataset.y)))
        self.feature_probs = np.zeros((len(np.unique(dataset.y)), n_features))

        for c in np.unique(dataset.y):
            class_samples = dataset.X[dataset.y == c]
            class_count = len(class_samples)
            self.class_prior[c] = class_count / n_samples

            feature_counts = np.sum(class_samples, axis=0)
            self.feature_probs[c] = (feature_counts + self.smoothing) / (class_count + self.smoothing * 2)
            
    def predict(self, dataset):
        predictions = []
        for sample in dataset.X:
            class_probs = []
            for c in range(len(self.class_prior)):
                class_prob = np.prod(sample * self.feature_probs[c] + (1 - sample) * (1 - self.feature_probs[c])) * self.class_prior[c]
                class_probs.append(class_prob)
            predictions.append(np.argmax(class_probs))
        return np.array(predictions)

    def score(self, dataset):
        predictions = self.predict(dataset)
        accuracy = np.mean(predictions == dataset.y)
        return accuracy        

if __name__ == '__main__':

    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    
    nb = CategoricalNB(smoothing=1.0)

  
    nb.fit(dataset_train)

    prediction = nb.predict(dataset_test)

   
    score = nb.score(dataset_test)
    print(f'The accuracy of the model is: {score}')