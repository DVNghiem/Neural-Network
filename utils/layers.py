from abc import abstractmethod
from typing import Any
import numpy as np
from .activations import Activation


class Layer:
    def __init__(self, input_shape=None) -> None:
        self.input_shape = input_shape

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.feed_forward(*args)

    @abstractmethod
    def initialParam(self) -> None: ...

    @abstractmethod
    def feed_forward(self, x) -> Any: ...


class Dense(Layer):
    def __init__(self, unit: int, activation: Activation,  input_shape=None) -> None:
        super().__init__(input_shape=input_shape)
        self.unit = unit
        self.activation = activation

    def initialParam(self) -> None:
        self.weight = np.random.normal(
            loc=0, scale=0.01, size=(self.input_shape, self.unit))
        self.bias = np.random.randn(1, self.unit)

    def feed_forward(self, x) -> Any:
        z = x@self.weight+self.bias
        a = self.activation(z)
        return z, a

    def __str__(self) -> str:
        return 'dense'
