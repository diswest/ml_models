import numpy as np


def gd(X, y, gradient_f, cost_f=None, alpha=.01, n_iters=1000, tol=.001, debug=False):
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
