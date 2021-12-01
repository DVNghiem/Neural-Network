import numpy as np


def sigmoid(x, grad=False):
    if grad:
        return sigmoid(x)*(1-sigmoid(x))
    return 1.0/(1.0 + np.exp(-x))


def relu(x, grad=False):
    if grad:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    return np.maximum(x, 0)


def softmax(x, grad=False):
    if grad:
        J = - x[..., None] * x[:, None, :]
        iy, ix = np.diag_indices_from(J[0])
        J[:, iy, ix] = x * (1. - x)
        return J.sum(axis=1)

    exps = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return exps*1.0 / np.sum(exps, axis=1)[:, None]
