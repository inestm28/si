import numpy as np
from ..util import l2_distance, accuracy_score
from .model import Model


class KNN(Model):
    '''
    It calculates the distance of a new data point to all other training data points.
    The distance can be of any type e.g Euclidean or Manhattan etc.
    It then selects the K-nearest data points, where K can be any integer.
    Finally, it assigns the data point to the class to which the majority of the K data points belong.
    '''
    def __init__(self, num_neighbors, classification=True):
        super(KNN, self).__init__()
        self.num_neighbors = num_neighbors
        self.classification = classification

    def fit(self, dataset): #returns self
        self.dataset = dataset
        self.is_fitted = True

    def get_neighbors(self, x): #selects the K-nearest data points to x
        distances = l2_distance(x, self.dataset.X) #Euclidean distance: calcula a distância entre um valor do dataset e um valor estipulado x
        sorted_index = np.argsort(distances) #Returns the indices that would sort ascending an array (lowest distance = nearest); lista com índices dos valores das distâncias
        return sorted_index[:self.num_neighbors] #Pick the first K entries from the sorted collection

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        neighbors = self.get_neighbors(x) #lista de índices dos valores das distâncias
        values = self.dataset.y[neighbors].tolist() #lista com os k valores de y (labels) correspondentes
        if self.classification: #if task is classification -> return the mode of the K labels (label that appears the most often)
            prediction = max(set(values), key=values.count) #set is unordered collection which does not allow duplicates.
        else: #regression
            prediction = sum(values)/len(values) #if regression -> return the average of the K labels
        return prediction #label mais frequente

    def cost(self, X=None, y=None):
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y

        y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=X.T) #se dataset.X tiver mais de que uma coluna -> Apply a function (self.predict) to 1-D slices along the given axis (axis=0 -> linhas, como é transposto faz ao longo das colunas).
        return accuracy_score(y, y_pred)