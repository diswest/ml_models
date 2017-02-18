import numpy as np
from abc import ABCMeta, abstractmethod

from .optimization import gd


class BaseLinearModel(metaclass=ABCMeta):
    """ Base linear model """

    def __init__(self, alpha=0.01, n_iters=1000, tol=0.0001, debug=False):
        self._coef = None
        self._alpha = alpha
        self._n_iters = n_iters
        self._debug = debug
        self._tol = tol

    def fit(self, X, y):
        # add intercept column
        X_copy = np.insert(X, 0, [1], axis=1)

        self._coef, cost = gd(
            X_copy,
            y,
            gradient_f=self._gradient_f,
            cost_f=self._cost_f,
            alpha=self._alpha,
            n_iters=self._n_iters,
            tol=self._tol,
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


class RidgeRegression(LinearRegression):
    """ Ridge regression (Linear regression with L2 regularization) """

    def __init__(self, alpha=0.01, lambda_coef=0.1, n_iters=1000, tol=0.0001, debug=False):
        self._lambda_coef = lambda_coef
        super(RidgeRegression, self).__init__(
            alpha=alpha,
            n_iters=n_iters,
            tol=tol,
            debug=debug
        )

    def _loss(self, X, y, coef):
        h = self._decision_function(X, coef)
        return h - y

    def _cost_f(self, X, y, coef):
        loss = self._loss(X, y, coef)
        m = X.shape[0]

        # noinspection PyTypeChecker
        penalty = self._lambda_coef * np.sum(coef ** 2)

        return (np.sum(loss ** 2) + penalty) / (2 * m)

    def _gradient_f(self, X, y, coef):
        loss = self._loss(X, y, coef)
        m = X.shape[0]

        penalty = self._lambda_coef * np.sum(coef)

        return (np.dot(X.T, loss) + penalty) / m
