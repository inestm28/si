import numpy as np
from .model import Model
from ..util import add_intersect, sigmoid

class LogisticRegression(Model):
    #no regularization
    #alpha is the learning rate
    def __init__(self, epochs=1000, alph=0.3):
        super(LogisticRegression, self).__init__()
        self.epochs=epochs
        self.alph=alph
        self.theta=None #theta are randomly initialized values

    def fit(self, dataset): #using x_train and y_train to train the model
        X,y = dataset.getXy()
        X = add_intersect(X) #vai pôr X e uma coluna de uns com o mesmo no de linhas lado a lado
        self.X=X
        self.y=y
        self.train_gd(X,y)
        self.is_fitted=True

    def train_gd(self, X, y): #gradient descendent: update theta values until cost function reaches its minimum
        n = X.shape[1] #no de colunas
        self.history={}
        self.theta=np.zeros(n)
        for epoch in range(self.epochs):
            z = np.dot(self.theta, X.T)
            h = sigmoid(z) #predicted value
            gradiente = np.dot(X.T, (h-y)) / y.size
            self.theta -= self.alph * gradiente
            self.history[epoch] = [self.theta[:], self.cost()]

    def predict(self, X):
        assert self.is_fitted, 'model must be fitted before predicting'
        _x = np.hstack(([1],X))
        z = np.dot(self.theta, _x)
        h = sigmoid(z) #predicted value
        if h <0.5:
            return 0
        else:
            return 1

    def cost(self, X=None, y=None, theta=None):
        X=add_intersect(X) if X is not None else self.X
        y=y if y is not None else self.y
        theta=theta if theta is not None else self.theta

        z = np.dot(self.theta, self.X.T)
        y1 = sigmoid(z) #predicted value
        return -(1/len(self.X)) * np.sum(self.y*np.log(y1) + (1-self.y)*np.log(1-y1))


class LogisticRegressionReg(LogisticRegression):
    #with L2 regularization, aka, Ridge Regression
    #solves overfitting by penalizing the cost function
    #it adds a penalty term in the cost function
    #lambda is the regularization parameter, which controls the trade-off between fitting the training data well vs keeping the params small to avoid overfitting.
    def __init__(self,epochs = 1000, alph=0.3,lambd = 1):
        super(LogisticRegressionReg, self).__init__(epochs=epochs, alph=alph)
        self.lambd = lambd

    def fit(self, dataset): #using x_train and y_train to train the model
        X,y = dataset.getXy()
        X = add_intersect(X) #vai pôr X e uma coluna de uns com o mesmo no de linhas lado a lado
        self.X=X
        self.y=y
        self.train_gd(X,y)
        self.is_fitted=True

    def train_gd(self, X, y):
        m = X.shape[0] #no de linhas
        n = X.shape[1] #no de colunas
        self.history ={}
        self.theta = np.zeros(n)
        lambdas = np.full(m, self.lambd)
        lambdas[0] = 0
        for epoch in range(self.epochs):
            z = np.dot(X, self.theta)
            h = sigmoid(z) #predicted value
            gradiente = np.dot(X.T, (h-y)) / y.size
            gradiente[1:] = gradiente[1:] + (self.lambd/m) * self.theta[1:]
            self.theta -= self.alph * gradiente
            self.history[epoch] = [self.theta[:], self.cost()]

    def predict(self, X):
        assert self.is_fitted, 'model must be fitted before predicting'
        _x = np.hstack(([1],X))
        z = np.dot(self.theta, _x)
        h = sigmoid(z) #predicted value
        if h <0.5:
            return 0
        else:
            return 1

    def cost(self, X=None, y=None, theta=None):
        # it adds a penalty term in the cost function
        X=add_intersect(X) if X is not None else self.X
        y=y if y is not None else self.y
        theta=theta if theta is not None else self.theta

        z = np.dot(self.theta, self.X.T)
        y1 = sigmoid(z) #predicted value
        penalizacao = np.dot(self.theta[1:],self.theta[1:]) * self.lambd / (2*len(self.X))
        cost = -(1/len(self.X)) * np.sum(self.y*np.log(y1) + (1-self.y)*np.log(1-y1)) + penalizacao
        return cost