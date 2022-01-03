import numpy as np
from ..util import l2_distance, accuracy_score
from .model import Model


class KNN(Model):
    def __init__(self, num_neighbors, classification=True):
        super(KNN, self).__init__()
        self.num_neighbors = num_neighbors
        self.classification = classification

    def fit(self, dataset): #returns self
        self.dataset = dataset
        self.is_fitted = True

    def get_neighbors(self, x):
        distances = l2_distance(x, self.dataset.X) #Euclidean distance: calcula a distância entre um valor do dataset e um valor estipulado x
        sorted_index = np.argsort(distances) #Returns the indices that would sort an array
        return sorted_index[:self.num_neighbors] #Pick the first K entries from the sorted collection

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        neighbors = self.get_neighbors(x)
        values = self.dataset.y[neighbors].tolist()
        if self.classification: #return the mode of the K labels
            prediction = max(set(values), key=values.count)
        else: #regression
            prediction = sum(values)/len(values) #return the average of the K labels
        return prediction

    def cost(self, X=None, y=None):
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=X.T) #Apply a function (self.predict) to 1-D slices along the given axis.
        return accuracy_score(y, y_pred)