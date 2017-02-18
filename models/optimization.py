import numpy as np


def gd(X, y, gradient_f, cost_f=None, alpha=0.01, n_iters=1000, tol=0.0001, debug=False):
    """ Full batch gradient descent """

    coef = np.ones((X.shape[1], 1))
    costs = []
    for _ in range(n_iters):
        gradient = gradient_f(X, y, coef)
        coef -= alpha * gradient
        cost = cost_f(X, y, coef)
        if debug:
            costs.append(cost)

        # check tolerance
        if cost <= tol:
            break

    return coef, costs
