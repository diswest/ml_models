import numpy as np


def gd(X, y, gradient_f, alpha=0.01, n_iters=1000, cost_f=None, debug=False):
    """ Full batch gradient descent """

    coef = np.ones((X.shape[1], 1))
    cost = []
    for _ in range(n_iters):
        gradient = gradient_f(X, y, coef)
        coef -= alpha * gradient
        if debug:
            cost.append(cost_f(X, y, coef))

    return coef, cost
