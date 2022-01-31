from abc import ABC, abstractmethod
#from typing import MutableSequence
import numpy as np

# from numpy.core.fromnumeric import size, transpose
# from scipy.signal.ltisys import LinearTimeInvariant

from .model import Model
#from scipy import signal

from si.util.metrics import mse, mse_prime
# from si.util.im2col import pad2D, im2col, col2im

__all__ = ['Dense', 'Activation', 'NN']

class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_error, lr):
        #lr is the learning rate (alpha)
        raise NotImplementedError

class Dense(Layer):
    #Fully Connected layer
    '''
    The dense layer’s neuron receives output from every neuron of its preceding layer,
    where neurons of the dense layer perform matrix-vector multiplication,
    where the row vector of the output from the preceding layers is equal to the column vector of the dense layer.
    Values under the matrix are the trained parameters of the preceding layers and also can be updated by the backpropagation.
    '''
    def __init__(self, input_size, output_size):
        self.weights= np.random.rand(input_size, output_size) -0.5 #matriz com shape (no linhas=no elementos matriz input, no colunas=no elementos matriz output)
        self.bias= np.zeros((1, output_size)) #array com shape(1 linha, no colunas=no elementos output)

    def setWeights(self, weights, bias):
        #Sets the weight and the bias for the NN
        if (weights.shape != self.weights.shape):
            raise ValueError(f'shapes mismatch {weights.shape} and {self.weights.shape}") ')
        if (bias.shape != self.bias.shape):
            raise ValueError(f"Shapes missmatch{bias.shape}and {self.bias.shape}")
        self.weights = weights
        self.bias = bias

    def forward(self, input):
        '''
        predictions are made based on the values in the input nodes and the weights.
        '''
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias #produto escalar entre inputs and weights e no final soma-se o bias
        return self.output

    def backward(self, output_error, lr):
        '''
        compare the predicted output with the actual output.
        Next, fine-tune weights and the bias in such a manner that our predicted output becomes closer to the actual output,
        known as "training the neural network".
        1o -> Calculating the cost/loss -> difference between the predicted output and the actual output.
        2o -> minimize cost function

        'computes dE/dW, dE/dB for a given output_error=dE/dY'
        '''
        # compute the weights error dE/dW = X.T*dE/dY
        weights_error = np.dot(self.input.T, output_error)
        # compute the bias error dE/dB
        bias_error = np.sum(output_error, axis=0)
        input_error = np.dot(output_error, self.weights.T)
        # update parameters
        self.weights -= lr * weights_error
        self.bias -= lr * bias_error
        return input_error

class Activation(Layer):
    def __init__(self, activation):
        self.activation = activation

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, lr):
        # learning rate is not used because there is no "learnable" parameters
        # only passes the error do the previous layer
        return np.multiply(self.activation.prime(self.input), output_error)


class NN(Model):
    def __init__(self, epochs=1000, lr=0.001, verbose=True):
        #neural network model
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, dataset):
        X, y = dataset.getXy()
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            output = X
            # forward propagation
            for layer in self.layers:
                output = layer.forward(output)

            # backward propagation
            error = self.loss_prime(y, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)

            # calculate average error on all samples
            err = self.loss(y, output)
            self.history[epoch] = err
            if self.verbose:
                print(f'epoch {epoch+1}/{self.epochs} error = {err}')
        if not self.verbose:
            print(f"error={err}")
        self.is_fitted = True

    def predict(self, input):
        assert self.is_fitted, 'Model must be fit before'
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def cost(self, X=None, y=None):
        assert self.is_fitted, 'Model must be fit'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        output = self.predict(X)
        return self.loss(y, output)

# im2col.py para pasta util
# class Pooling2D
# class MaxPooling2D(Pooling2D)
#     def pool????
#         raise NotImplementedError
#     def dpool????
#         raise NotImplementedError
#
#     def forward????
#     self.X_shape = input.shape
#     n,h,w,d = input.shape
#     h_out = (h-self.size) / self.stride + 1
#     w_out = (w-self.size) / self.stride + 1
#     if not w_out.is_integer() or not h_out.is_integer():
#         raise Exception('Invalid output dimension!')
#     h_out,w_out = int(h_out), int(w_out)
#     x_reshaped=input.reshape(n*d,h,w,1)
#     self.X_col = im2col(X_reshaped, self.size, self.size, padding=0, stride=self.stride)
#     out,self.max_idx = self.pool(self.X_col)
#     out = out.reshape(h_out, w_out,n,d)
#     out = out.transpose(3,2,0,1)
#     return out
#
#     def backward(self, output_error, learning_rate):
#         n, w, h, d = self.X_shape
#         dX_col = np.zeros_like(self.X_col)
#         dout_col = output_error.transpose(2,3,0,1).ravel()
#         dX = self.dpool(dX_col, dout_col, self.max_idx)
#         dX = col2im(dX, (n+d,1,h,w),self.size, self.size, padding=0, stride=self.stride)
#         dX=dX.reshape(self.X_shape)
#         return dX