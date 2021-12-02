from typing import Any
import numpy as np
from abc import abstractmethod


class Loss:
    def __init__(self, eps=1e-15) -> None:
        self.eps = eps

    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def grad(self, y_true, y_pred):
        raise NotImplementedError

    def normalize(self, y):
        y = np.clip(y, self.eps, 1.0 - self.eps)
        return y

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.loss(*args)

    def __str__(self) -> str:
        return self.__class__.__name__


class BinaryCrossentropy(Loss):
    def __init__(self, eps=1e-15) -> None:
        super().__init__(eps=eps)

    def loss(self, y_true, y_pred):
        n = y_pred.shape[0]
        y_pred = self.normalize(y_pred)
        return -np.sum(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))/n

    def grad(self, y_true, y_pred):
        y_pred = self.normalize(y_pred)
        return -(y_true/y_pred-(1-y_true)/(1-y_pred))


class CategoricalCrossentropy(Loss):
    def __init__(self, eps=1e-15) -> None:
        super().__init__(eps=eps)

    def loss(self, y_true, y_pred):
        n = y_pred.shape[0]
        y_pred = self.normalize(y_pred)
        return -np.sum(y_true * np.log(y_pred))/n

    def grad(self, y_true, y_pred):
        return y_pred - y_true


class MeanSquareError(Loss):
    def __init__(self, eps=1e-15) -> None:
        super().__init__(eps=eps)

    def loss(self, y_true, y_pred) -> float:
        n = y_pred.shape[0]
        return np.sum((y_true-y_pred)**2)/n

    def grad(self, y_true, y_pred) -> Any:
        n = y_pred.shape[0]
        return 2*(y_true-y_pred)/n
