import numpy as np
import sys


def gd(X, y, gradient_f, cost_f=None, alpha=.01, n_iters=1000, tol=.0001, debug=False):
    """ Full batch gradient descent """

    coef = np.ones((X.shape[1], 1))
    costs = []
    cost = cost_f(X, y, coef)
    step = 0
    while step < n_iters and cost > tol:
        gradient = gradient_f(X, y, coef)
        coef -= alpha * gradient
        cost = cost_f(X, y, coef)
        step += 1
        if debug:
            costs.append(cost)

    return coef, costs


def cgd():
    """ Conjugate gradient descent """
    pass


def mgd():
    """ Mini-batch gradient descent """
    pass


def sgd():
    """ Stochastic gradient descent """

    # Momentum
    # Averaging (sag?)
    # AdaGrad
    # Adadelta
    # RMSProp
    # Adam
    # Adamax
    # Nadam
    # kSGD

    pass


def pgd():
    """ Proximal gradient descent """
    pass


def sag():
    """ Stochastic average gradient """
    pass


def cd(X, y, optimize_f=None, n_iters=1000, tol=.001, debug=False):
    """ Coordinate descent """
    coef = np.zeros(X.shape[1])
    costs = []
    step = 0
    cost = sys.float_info.max
    while step < n_iters and cost > tol:
        for j in range(len(coef)):
            old_coef_j = coef[j].copy()
            coef_copy = coef.copy()
            coef_copy[j] = 0
            coef[j] = optimize_f(X, y, coef_copy, j)
            cost = abs(coef[j] - old_coef_j)
            if debug:
                costs.append(cost)
        step += 1

    return coef, costs


def newton():
    """ Newton's method """
    pass


def newton_cg():
    """ Newton conjugate gradient """
    pass
