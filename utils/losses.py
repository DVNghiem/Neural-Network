import numpy as np


def cross_entropy(y_true, y_pred, grad=False):
    n = y_true.shape[0]
    if grad:
        return y_pred - y_true

    loss = -np.sum(y_true * np.log(y_pred))/n
    return loss


def MSE(y_true, y_pred, grad=False):
    n = y_pred.shape[0]
    if grad:
        return 2*(y_true-y_pred)/n

    return np.sum((y_true-y_pred)**2)/n


def binary_crossentropy(y_true, y_pred, grad=False):
    n = y_pred.shape[0]
    if grad:
        return -(y_true/y_pred-(1-y_true)/(1-y_pred))

    loss = -np.sum(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))/n
    return loss
