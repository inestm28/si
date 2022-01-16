from .util import train_test_split
import numpy as np
import itertools

# MODEL SELECTION

class Cross_Validation:
    #avaliar a performance de um modelo
    def __init__(self, model, dataset,score=None, **kwargs):
        self.model=model #modelo que se quer avaliar
        self.dataset=dataset
        self.cv=kwargs.get('cv',3) #.get returns 3. number of folds (K-fold)
        self.split=kwargs.get('split', 0.8)
        self.train_scores=None
        self.test_scores=None
        self.ds=None
        self.score=score

    def run(self):
        train_scores = []
        test_scores = []
        ds=[] #lista com tuplos de conjuntos de treino e de teste
        for _ in range(self.cv): # 3 folds. underscore pq não vamos precisar do valor da variável
            train, test = train_test_split(self.dataset, self.split)
            ds.append((train, test))
            self.model.fit(train)
            if not self.score: #if self.score diferente de None então corre o ciclo
                train_scores.append(self.model.cost()) #cost -> dá a medida de quão longe o valor previsto está do output original
                test_scores.append(self.model.cost(test.X, test.y))
            else: #if self.score = None
                y_train=np.ma.apply_along_axis(self.model.predict, axis=0, arr=train.X.T)
                train_scores.append(self.score(train.y, y_train))
                y_test=np.ma.apply_along_axis(self.model.predict, axis=0, arr=test.X.T)
                test_scores.append(self.score(test.y, y_test))
        self.train_scores=train_scores
        self.test_scores=test_scores
        self.ds=ds
        return train_scores, test_scores #accuracies de cada fold

    def toDataframe(self):
        import pandas as pd
        assert self.train_scores and self.test_scores, 'Need to run code first'
        return pd.DataFrame({'Train Scores': self.train_scores, 'Test scores': self.test_scores})

class Grid_Search:
    #automatically selecting the best hyper parameteres for a particular model
    def __init__(self, model, dataset, parameters, **kwargs):
        self.model=model #modelo a ser avaliado
        self.dataset=dataset
        hasparam=[hasattr(self.model, param) for param in parameters] #hasattr() returns true if an object has the given named attribute, hasattr(object, name of attribute)
        if np.all(hasparam): #Test whether all array elements along a given axis evaluate to True.
            self.parameters=parameters #dictionary of all the parameters and their corresponding list of values that you want to test for best performance
        else:
            index=hasparam.index(False)
            keys=list(parameters.keys())
            raise ValueError(f"wrong parameters: {keys[index]}")
        self.kwargs=kwargs
        self.results=None

    def run(self):
        self.results=[]
        attrs=list(self.parameters.keys()) #nome dos parametros
        values=list(self.parameters.values()) #valores dos parametros
        for conf in itertools.product(*values): #itertools.product -> cartesian product of all the iterable provided as the argument.
            for i in range(len(attrs)):
                setattr(self.model, attrs[i], conf[i])
            scores=Cross_Validation(self.model, self.dataset, **self.kwargs).run() #faz CROSS VALIDATION
            self.results.append((conf, scores)) #para cada valor de parametro, dá as accuracies do modelo
        return self.results

    def toDataframe(self):
        import pandas as pd
        assert self.results, 'The grid search needs to be ran.'
        data=dict()
        for i, k in enumerate(self.parameters.keys()):
            v=[]
            for r in self.results:
                v.append(r[0][i])
            data[k]=v
        for i in range(len(self.results[0][1][0])):
            treino, teste = [], []
            for r in self.results:
                treino.append(r[1][0][i])
                teste.append(r[1][1][i])
            data['Train ' + str(i + 1)] = treino
            data['Test ' + str(i + 1)] = teste
        return pd.DataFrame(data)