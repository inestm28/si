from .model import Model
from src.si.util.metrics import accuracy_score
from ..util import add_intersect
import numpy as np


class CategoricalNB(Model):
    '''
    Naive Bayes Classifier
    predict the conditional probability of a class
    1o -> determine the probability that the data belongs to a certain distribution of classes P(Data|Class).
    2o -> multiply that by P(Class). To calculate the P(Class), count the number of samples (rows) for a specific class and
          divide that by the total number of samples.
    '''
    def __init__(self):
        super(CategoricalNB, self).__init__()

    def fit(self, X, y):
        '''
        Compute summary and prior statistics for each class (y feature) in the (training) dataset.
        '''
        # X,y =dataset.getXy()
        self.X=X
        self.y=y
        # get number of samples (rows) and features (columns)
        self.n_samples=X.shape[0] #no de linhas
        self.n_features=X.shape[1] #no de colunas

        # get number of uniques classes
        self.n_classes = len(np.unique(y))

        # create three zero-matrices to store summary stats & prior
        self.mean = np.zeros((self.n_classes, self.n_features)) #(no linhas=no classes, no col=no features)
        self.variance = np.zeros((self.n_classes, self.n_features))
        self.priors = np.zeros(self.n_classes) #array com no linhas igual ao no classes

        #iterate over all the classes, compute the statistics and update the zero matrices.
        for c in range(self.n_classes): #se forem 3 classes, c=0,1,2
            # create a subset of data for the specific class 'c'
            X_c = self.X[self.y == c] #dataset X mas só com as amostras que pertencem à classe "c"

            # calculate statistics and update zero-matrices
            self.mean[c, :] = np.mean(X_c, axis=0) #calcular a média de cada feature (coluna do dataset X) por cada classe
            self.variance[c, :] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / self.n_samples #frequência/probabilidade de uma classe -> no amostras que pertencem a uma classe / no total de amostras

    #TO MAKE THE PREDICTION:
    #obtain the probability that the data belong to a certain class or, more specifically, come from the same distribution.
    #assume distribution of the data is Gaussian. create new method that returns the probability of a new sample.

    def gaussian_density(self, x, mean, var):
        '''
        implementation of gaussian density function

        receives a single sample and calculates the probability.
        provide the mean and variance.
        '''
        const = 1 / np.sqrt(var * 2 * np.pi)
        proba = np.exp(-0.5 * ((x - mean) ** 2 / var))

        return const * proba

    #é necessário calcular a probabilidade condicional de uma classe para apenas uma amostra

    def get_class_probability(self, x):
        '''
        itera sobre todas as classes
        recolhe as estatísticas anteriores
        calcula a probabilidade condicional de uma classe para apenas uma amostra
        '''
        # store new posteriors for each class in a single list
        posteriors = []
        for c in range(self.n_classes):
            # get summary stats & prior
            mean = self.mean[c]
            variance = self.variance[c]
            prior = np.log(self.priors[c]) #frequência/probabilidade anterior/ calculada da classe c
            # calculate new posterior & append to list
            posterior = np.sum(np.log(self.gaussian_density(x, mean, variance))) #calcula a probabilidade da classe c para uma det amostra x, dado media e var
            posterior = prior + posterior
            posteriors.append(posterior)

        #posteriors=[0->20%, 1->, 2->]
        # retorna o índice da classe com maior probabilidade
        return np.argmax(posteriors)

    def predict(self, X):
        # for each sample x in the dataset X
        y_hat = [self.get_class_probability(x) for x in X]
        #[amostra1->indice da classe com maior probabilidade, amostra 2->..., ...]
        return np.array(y_hat)

    def get_accuracy(self, y_t, y_hat):
        return np.sum(y_t == y_hat) / len(y_t)