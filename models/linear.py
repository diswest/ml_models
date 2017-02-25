import numpy as np
from abc import ABCMeta, abstractmethod

from .optimization import gd, cd


class BaseLinearModel(metaclass=ABCMeta):
    """ Base linear model """

    def __init__(self, n_iters=1000, tol=.0001, debug=False):
        self._coef = None
        self._norm = None
        self._n_iters = n_iters
        self._debug = debug
        self._tol = tol

    def fit(self, X, y):
        # add intercept column
        X_copy = np.insert(X, 0, [1], axis=1)
        X_copy, self._norm = self._normalize(X_copy)

        self._coef, cost = self._solver(X_copy, y)
        self._coef = self._coef / self._norm.reshape(self._coef.shape)

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

    @staticmethod
    def _normalize(X):
        norm = np.sqrt(np.sum(X**2, axis=0))
        return X / norm, norm

    @abstractmethod
    def _loss(self, X, y, coef):
        pass

    @abstractmethod
    def _solver(self, X, y):
        pass


class LinearRegression(BaseLinearModel):
    """ Linear regression """

    def __init__(self, alpha=.01, n_iters=1000, tol=.0001, debug=False):
        self._alpha = alpha
        super(LinearRegression, self).__init__(
            n_iters=n_iters,
            tol=tol,
            debug=debug
        )

    def _solver(self, X, y):
        return gd(
            X,
            y,
            gradient_f=self._gradient_f,
            cost_f=self._cost_f,
            alpha=self._alpha,
            n_iters=self._n_iters,
            tol=self._tol,
            debug=self._debug
        )

    def _loss(self, X, y, coef):
        return y - self._decision_function(X, coef)

    def _cost_f(self, X, y, coef):
        loss = self._loss(X, y, coef)
        m = X.shape[0]

        return np.sum(loss ** 2) / (2 * m)

    def _gradient_f(self, X, y, coef):
        loss = self._loss(X, y, coef)
        m = X.shape[0]

        return -np.dot(X.T, loss) / m


class RidgeRegression(LinearRegression):
    """ Ridge regression (Linear regression with L2 regularization) """

    def __init__(self, alpha=.01, l2_penalty=.1, n_iters=1000, tol=.0001, debug=False):
        self._l2_penalty = l2_penalty
        super(RidgeRegression, self).__init__(
            alpha=alpha,
            n_iters=n_iters,
            tol=tol,
            debug=debug
        )

    def _cost_f(self, X, y, coef):
        loss = self._loss(X, y, coef)
        m = X.shape[0]
        penalty = self._l2_penalty * np.sum(np.dot(coef[1:].T, coef[1:]))

        return (np.dot(loss.T, loss).flatten() + penalty) / (2 * m)

    def _gradient_f(self, X, y, coef):
        loss = self._loss(X, y, coef)
        m = X.shape[0]
        penalty = self._l2_penalty * np.sum(coef[1:])

        gradient = -np.dot(X.T, loss) + penalty
        gradient[0] -= penalty

        return gradient / m


class Lasso(BaseLinearModel):
    """ LASSO (Least Absolute Shrinkage and Selection Operator, linear regression with L1 regularization) """

    def __init__(self, l1_penalty=.1, n_iters=1000, tol=.0001, debug=False):
        self._l1_penalty = l1_penalty

        super(Lasso, self).__init__(
            n_iters=n_iters,
            tol=tol,
            debug=debug
        )

    def _solver(self, X, y):
        return cd(
            X,
            y.flatten(),
            optimize_f=self._optimize_f,
            n_iters=self._n_iters,
            tol=self._tol,
            debug=self._debug
        )

    def _loss(self, X, y, coef):
        return y - self._decision_function(X, coef)

    def _optimize_f(self, X, y, coef, j):
        loss = self._loss(X, y, coef)

        # ro is greek ρ
        ro = np.sum(np.dot(X[:, j].T, loss))
        return ro if j == 0 else self._soft_threshold(ro)

    def _soft_threshold(self, ro):
        if ro < -self._l1_penalty/2:
            return ro + self._l1_penalty/2
        elif -self._l1_penalty/2 <= ro <= self._l1_penalty/2:
            return 0
        elif ro > self._l1_penalty/2:
            return ro - self._l1_penalty/2
