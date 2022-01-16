from .model import Model
from ..util import mse, add_intersect
import numpy as np

class LinearRegression(Model):
    '''
    fits multiple lines on the data points and
    returns the optimal value for the intercept and the slope
    of the line that results in the least error (minimiza a discrepancia)
    '''

    def __init__(self, gd=False, epochs=1000, alph=0.001):
        super(LinearRegression, self).__init__()
        self.gd=gd #gradient descendent
        self.theta=None #theta são valores iniciados aleatoriamente
        self.epochs=epochs
        self.alph=alph #tx de aprendizagem

    def fit(self, dataset): #treinar o algoritmo com training data, ou seja, finds the best value for the intercept and slope
        X,y =dataset.getXy()
        X=np.hstack((np.ones((X.shape[0],1)),X)) #vai pôr X e uma coluna de 1's com o mesmo no de linhas lado a lado
        self.X=X
        self.y=y
        #if gd=false -> closed form, if gd=true -> gradient descendent
        self.train_gd(X,y) if self.gd else self.train_closed(X,y)
        self.is_fitted=True

    def train_closed(self, X, y):
        self.theta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) #solves system of linear scalar equations

    def train_gd(self, X, y): #gradient descendent: update theta values until cost function reaches its minimum
        m=X.shape[0] #no de linhas
        n=X.shape[1] #no de colunas
        self.history={}
        self.theta=np.zeros(n) #coluna com n linhas (no col de X) preenchida a zeros
        for epoch in range(self.epochs):
            grad=1/m * (X.dot(self.theta)-y).dot(X)
            self.theta -= self.alph*grad #update theta
            self.history[epoch]=[self.theta[:], self.cost()]

    def predict(self, X):
        assert self.is_fitted, 'Model must be fit before predicting'
        _x=np.hstack(([1], X))
        return np.dot(self.theta, _x) #produto escalar

    def cost(self, X=None, y=None, theta=None): #dá a medida de quão longe o valor previsto está do output original
        X=add_intersect(X) if X is not None else self.X
        y=y if y is not None else self.y
        theta=theta if theta is not None else self.theta

        y_pred=np.dot(self.X, self.theta) #predicted value é o produto escalar entre input variable (self.X) e theta
        return mse(self.y, y_pred)/2 #erro quadrático médio ("metrics.py" do util)

class LinearRegressionReg(LinearRegression):
    #with L2 regularization, aka, Ridge Regression
    #solves overfitting by penalizing the cost function
    #it adds a penalty term in the cost function
    #lambda is the regularization parameter, which controls the trade-off between fitting the training data well vs keeping the params small to avoid overfitting.
    def __init__(self, gd=False, epochs=1000, alph=0.001, lambd=1):
        super(LinearRegressionReg, self).__init__(gd=gd, epochs=epochs, alph=alph)
        self.lambd=lambd  #parâmetro de regularização

    def train_closed(self, X, y): #closed form to fit the model
        n=X.shape[1]
        identity=np.eye(n)    #cria matriz identidade e muda-se 1o valor da diag para zero (para teta zero ser 0)
        identity[0,0]=0
        self.theta=np.linalg.inv(X.T.dot(X) + self.lambd * identity).dot(X.T).dot(y) #adiciona parâmetro de regularização e multiplica por matriz identidade
        self.is_fitted=True

    def train_gd(self, X, y): #gradient descendent: update theta values until cost function reaches its minimum
        m=X.shape[0] #no de linhas
        n=X.shape[1] #no de colunas
        self.history={}
        self.theta=np.zeros(n)
        lambdas=np.full(m, self.lambd) #preenche um vector de m linhas com o valor do lambda (parâmetro de regularização): vector [lambda lambda lambda lambda]
        lambdas[0]=0     #vector [0 lambda lambda lambda lambda]
        for epoch in range(self.epochs):
            grad=1/m * (X.dot(self.theta)-y).dot(X)
            self.theta -= (self.alph/m)*(self.theta*lambdas+grad)  #sem regularização -> self.theta -= self.alph*grad
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