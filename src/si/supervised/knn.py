import numpy as np
from ..util import l2_distance, accuracy_score
from .model import Model


class KNN(Model):
    def __init__(self, num_neighbors, classification=True):
        super(KNN, self).__init__()
        self.num_neighbors = num_neighbors
        self.classification = classification

    def fit(self, dataset):
        self.dataset = dataset
        self.is_fitted = True

    def get_neighbors(self, x):
        distances = l2_distance(x, self.dataset.X)
        sorted_index = np.argsort(distances)
        return sorted_index[:self.num_neighbors]

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        neighbors = self.get_neighbors(x)
        values = self.dataset.y[neighbors].tolist()
        if self.classification:
            prediction = max(set(values), key=values.count)
        else:
            prediction = sum(values)/len(values)
        return prediction

    def cost(self, X=None, y=None):
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        y_pred = np.ma.apply_along_axis(self.predict,
                                        axis=0, arr=X.T)
        return accuracy_score(y, y_pred)
