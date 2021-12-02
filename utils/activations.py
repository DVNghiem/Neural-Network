from typing import Any
import numpy as np
from abc import abstractmethod


class Activation:
    def __init__(self) -> None:
        pass

    def __call__(self, x) -> Any:
        return self.feed_forward(x)

    @abstractmethod
    def feed_forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def backward(self, x):
        raise NotADirectoryError


class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__()

    def feed_forward(self, x):
        return 1.0/(1.0+np.exp(-x))

    def backward(self, x):
        fn = self.feed_forward(x)
        return fn*(1-fn)


class Relu(Activation):
    def __init__(self) -> None:
        super().__init__()

    def feed_forward(self, x):
        return np.maximum(x, 0)

    def backward(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


class Softmax(Activation):
    def __init__(self) -> None:
        super().__init__()

    def feed_forward(self, x):
        e_X = np.exp(x - np.max(x, axis=self.dim, keepdims=True))
        return e_X / e_X.sum(axis=self.dim, keepdims=True)

    def backward(self, x):
        J = - x[..., None] * x[:, None, :]
        iy, ix = np.diag_indices_from(J[0])
        J[:, iy, ix] = x * (1. - x)
        return J.sum(axis=1)
