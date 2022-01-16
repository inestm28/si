from .model import Model
import numpy as np
 #fvote -> formas de combinar o output
def majority(values):
    '''
    voting approach -> classification
    uses the predicted labels and a majority rules system
    '''
    return max(set(values), key=values.count)

def average(values):
    '''
    Parallel methods aim to reduce the error rate by training many models in parallel and averaging the results together.
    Regression
    '''
    return sum(values)/len(values)

class Ensemble(Model):
    '''
    joining together the predictions of multiple classifiers (combines different algorithms together)
    it can correct for errors made by any individual classifier
    by either improving prediction accuracy or reducing aspects like bias and variance,
    leading to better accuracy overall
    improve the performance of a predictive model
    '''
    def __init__(self, models, fvote, score):
        super().__init__()
        self.models=models
        self.fvote=fvote #como combino o output de cada um dos modelos -> def majority
        self.score=score #def accuracy_score do "metrics.py" na pasta util

    def fit(self, dataset):
        self.dataset=dataset
        for model in self.models:
            model.fit(dataset)
        self.is_fitted=True

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        preds = (model.predict(x) for model in self.models)
        vote=self.fvote(preds)
        return vote #valor do melhor modelo

    def cost(self, X=None, y=None):
        X=X if X is not None else self.dataset.X
        y=y if y is not None else self.dataset.y

        y_pred=np.ma.apply_along_axis(self.predict, axis=0, arr=X.T)
        return self.score(y, y_pred)