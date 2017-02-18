import numpy as np
from abc import ABCMeta, abstractmethod

from .optimization import gd


class BaseLinearModel(metaclass=ABCMeta):
    """ Base linear model """

    def __init__(self, alpha=0.01, n_iters=1000, debug=False):
        self._coef = None
        self._alpha = alpha
        self._n_iters = n_iters
        self._debug = debug

    def fit(self, X, y):
        # add intercept column
        X_copy = np.insert(X, 0, [1], axis=1)
        print(X_copy)

        self._coef, cost = gd(
            X_copy,
            y,
            gradient_f=self._gradient_f,
            alpha=self._alpha,
            n_iters=self._n_iters,
            cost_f=self._cost_f,
            debug=self._debug
        )

        if self._debug:
            return cost
        else:
            return None

    def predict(self, X):
        if self._coef is None:
            raise Exception('Model isn\'t fitted')
        X_copy = np.insert(X, 0, [1], axis=1)
        return self._decision_function(X_copy, self._coef)

    @staticmethod
    def _decision_function(X, coef):
        return np.dot(X, coef)

    @abstractmethod
    def _loss(self, X, y, coef):
        pass

    @abstractmethod
    def _cost_f(self, X, y, coef):
        pass

    @abstractmethod
    def _gradient_f(self, X, y, coef):
        pass


class LinearRegression(BaseLinearModel):
    """ Linear regression """

    def _loss(self, X, y, coef):
        h = self._decision_function(X, coef)
        return h - y

    def _cost_f(self, X, y, coef):
        loss = self._loss(X, y, coef)
        m = X.shape[0]

        return np.sum(loss ** 2) / (2 * m)

    def _gradient_f(self, X, y, coef):
        loss = self._loss(X, y, coef)
        m = X.shape[0]

        return np.dot(X.T, loss) / m
