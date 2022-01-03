from .model import Model
from ..util import mse, add_intersect
import numpy as np

class LinearRegression(Model):
    def __init__(self, gd=False, epochs=1000, alph=0.001):
        super(LinearRegression, self).__init__()
        self.gd=gd
        self.theta=None #theta are randomly initialized values
        self.epochs=epochs
        self.alph=alph #tx de aprendizagem

    def fit(self, dataset):
        X,y =dataset.getXy()
        X=np.hstack((np.ones((X.shape[0],1)),X))
        self.X=X
        self.y=y
        #closed form or GD
        self.train_gd(X,y) if self.gd else self.train_closed(X,y)
        self.is_fitted=True

    def train_closed(self, X, y):
        self.theta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def train_gd(self, X, y): #gradient descendent
        m=X.shape[0] #no de linhas
        n=X.shape[1] #no de colunas
        self.history={}
        self.theta=np.zeros(n)
        for epoch in range(self.epochs):
            grad=1/m * (X.dot(self.theta)-y).dot(X)
            self.theta -= self.alph*grad
            self.history[epoch]=[self.theta[:], self.cost()]

    def predict(self, X):
        assert self.is_fitted, 'Model must be fit before predicting'
        _x=np.hstack(([1], X))
        return np.dot(self.theta, _x)

    def cost(self, X=None, y=None, theta=None):
        X=add_intersect(X) if X is not None else self.X
        y=y if y is not None else self.y
        theta=theta if theta is not None else self.theta

        y_pred=np.dot(self.X, self.theta)
        return mse(self.y, y_pred)/2

class LinearRegressionReg(LinearRegression):
    #with L2 regularization, aka, Ridge Regression
    #solves overfitting by penalizing the cost function
    #it adds a penalty term in the cost function
    #lambda is the regularization parameter, which controls the trade-off between fitting the training data well vs keeping the params small to avoid overfitting.
    def __init__(self, gd=False, epochs=1000, alph=0.001, lambd=1):
        super(LinearRegressionReg, self).__init__(gd=gd, epochs=epochs, alph=alph)
        self.lambd=lambd  #(parâmetro de regularização)

    def train_closed(self, X, y): #uses closed form to fit the model
        n=X.shape[1]
        identity=np.eye(n)    #cria matriz identidade e muda-se 1o valor da diag para zero (para teta zero ser 0)
        identity[0,0]=0
        self.theta=np.linalg.inv(X.T.dot(X) + self.lambd * identity).dot(X.T).dot(y)
        self.is_fitted=True

    def train_gd(self, X, y):
        m=X.shape[0] #no de linhas
        n=X.shape[1] #no de colunas
        self.history={}
        self.theta=np.zeros(n)
        lambdas=np.full(m, self.lambd) #preenche um vector de m linhas com o valor do lambda (parâmetro de regularização): vector [lambda lambda lambda lambda]
        lambdas[0]=0     #vector [0 lambda lambda lambda lambda]
        for epoch in range(self.epochs):
            grad=1/m * (X.dot(self.theta)-y).dot(X)
            self.theta -= (self.alph/m)*(self.theta*lambdas+grad)  #self.theta -= self.alph*grad
            self.history[epoch] = [self.theta[:], self.cost()]

    def predict(self, X):
        assert self.is_fitted
        _x = np.hstack(([1], X))
        return np.dot(self.theta, _x)

    def cost(self, X=None, y=None, theta=None):
        X=add_intersect(X) if X is not None else self.X
        y=y if y is not None else self.y
        theta=theta if theta is not None else self.theta

        y_pred = np.dot(self.X, self.theta)
        return mse(self.y, y_pred) / 2