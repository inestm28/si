from copy import copy
import numpy as np

# Y is reserved to idenfify dependent variables
ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'


def label_gen(n):
    import itertools
    """ Generates a list of n distinct labels similar to Excel"""
    def _iter_all_strings():
        size = 1
        while True:
            for s in itertools.product(ALPHA, repeat=size):
                yield "".join(s)
            size += 1

    generator = _iter_all_strings()

    def gen():
        for s in generator:
            return s

    return [gen() for _ in range(n)]


def l1_distance(x, y):
    """Computes the manhatan distance of a point (x) to a set of
    points y.
    x.shape=(n,) and y.shape=(m,n)
    """
    import numpy as np
    dist = (np.absolute(x - y)).sum(axis=1)
    return dist


def l2_distance(x, y):
    """Computes the euclidean distance of a point (x) to a set of
    points y.
    x.shape=(n,) and y.shape=(m,n)
    """
    dist = ((x - y) ** 2).sum(axis=1)
    return dist


def train_test_split(dataset, split=0.8):
    from ..data import Dataset
    n = dataset.X.shape[0]
    m = int(split*n)
    arr = np.arange(n)
    np.random.shuffle(arr)
    train_mask = arr[:m]
    test_mask = arr[m:]

    train = Dataset(dataset.X[train_mask], dataset.y[train_mask], dataset._xnames, dataset._yname)
    test = Dataset(dataset.X[test_mask], dataset.y[test_mask], dataset._xnames, dataset._yname)
    return train, test


def add_intersect(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def to_categorical(y, num_classes=None, dtype='float32'):
    '''
    One-hot encoded vector (1 meaning TRUE, 0 meaning FALSE)
    a vector which has integers that represent different categories,
    can be converted into a matrix which has binary values
    number of rows equal to the length of the input vector and
    number of columns equal to the number of classes/categories.
    each integer value is represented as a binary vector that is
    all zero values except the index of the integer, which is marked with a 1. ([1, 0, 0] = class 1
                                                                                [0, 1, 0] = class 2
                                                                                [0, 0, 1] = class 3)
    '''
    y = np.array(y, dtype='int')
    input_shape = y.shape     #(no linhas, no colunas). shape[-1] -> no colunas. len(shape) -> no elementos da dimensão (2)
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1: #se y for 1 coluna (no linhas, 1)
        input_shape = tuple(input_shape[:-1]) #dá um tuplo apenas com o no de linhas
    y = y.ravel() #se tiver mais de uma coluna, flatten array (fica apenas uma coluna)
    if not num_classes: #se num_classes=None
        num_classes = np.max(y) + 1 #valor máximo de y + 1
    n = y.shape[0] #no de linhas
    categorical = np.zeros((n, num_classes), dtype=dtype) #array com n linhas e no de colunas igual ao no de categorias, preenchido a zeros
    categorical[np.arange(n), y] = 1 #np.arange->array com valores até n-1. vai pôr os 1's
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape) #numpy.reshape(array, newshape)
    return categorical