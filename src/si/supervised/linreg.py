from . import Model
from ..util import mse
import numpy as np

class LinearRegression(Model):
    def __init__(self, gd=False, epochs=1000, lr=0.001):
        super(LinearRegression, self).__init__()
        self.gd=gd
        self.theta=None
        self.epochs=epochs
        self.lr=lr

    def fit(self, dataset):
        X,y =dataset.getXy()
        X=np.hstack((np.ones((X.shape[0],1)),X))
        # self.X=X
        # self.y=y
        #closed form or GD
        self.train_gd(X,y) if self.gd else self.train_closed(X,y)
        self.is_fitted=True

    def train_closed(self, X, y):
        self.theta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def train_gd(self, X, y): #gradient descendent
        m=X.shape[0]
        n=X.shape[1]
        self.history={}
        self.theta=np.zeros(n)
        for epoch in range(self.epochs):
            grad=1/m * (X.dot(self.theta)-y).dot(X)
            self.theta-=self.lr*grad
            self.history[epoch]=[self.theta[:], self.cost()]

    def predict(self, x):
        assert self.is_fitted, 'Model must be fit before predicting'
        _x=np.hstack(([1], x))
        return np.dot(self.theta, _x)

    def cost(self):
        y_pred=np.dot(self.X, self.theta)
        return mse(self.y, y_pred)/2

    # def cost(selfself, X=None, y=None, theta=None):
    #     X=add_intersect(X) if X is not None else self.X
    #     y=y if y is not None else self.y
    #     theta=theta if theta is not None else self.theta
    #     y_pred=np.dot(X, theta)
    #     return mse(y, y_pred)/2

class LinearRegressionReg(LinearRegression):
    def __init__(self, gd=False, epochs=1000, lr=0.001, lbd=1): #linear regress. model with L2 regularization
        super(LinearRegressionReg, self).__init__(gd=gd, epochs=epochs,lr=lr)
        self.lbd=lbd

    def train_closed(self, X, y): #uses closed form to fit the model
        n=X.shape[1]
        identity=np.eye(n)    #cria matriz identidade e muda-se 1o valor da diag para zero (para teta zero ser 0)
        identity[0,0]=0
        self.theta=np.linalg.inv(X.T.dot(X)+self.lbd*identity).dot(X.T).dot(y)
        self.is_fitted=True

    def train_gd(self, X, y):
        m=X.shape[0]
        n=X.shape[1]
        self.history={}
        self.theta=np.zeros(n)
        lbds=np.full(m, self.lbd) #vector [0 lambda lambda lambda lambda)
        lbds[0]=0
        for epoch in range(self.epochs):
            grad=(X.dot(self.theta)-y).dot(X)
            self.theta -= (self.lr/m)*(lbds+grad)
            self.history[epoch] = [self.theta[:], self.cost()]