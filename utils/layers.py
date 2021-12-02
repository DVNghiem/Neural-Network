from abc import abstractmethod
import numpy as np


class Layer:
    def __init__(self, input_shape=None) -> None:
        self.input_shape = input_shape

    @abstractmethod
    def initialParam(self):
        pass

    @abstractmethod
    def feed_forward(self, x):
        pass


class Dense(Layer):
    def __init__(self, unit, activation,  input_shape=None) -> None:
        super().__init__(input_shape=input_shape)
        self.unit = unit
        self.activation = activation

    def initialParam(self):
        self.weight = np.random.normal(
            loc=0, scale=0.01, size=(self.input_shape, self.unit))

        self.bias = np.random.randn(1, self.unit)

    def feed_forward(self, x):
        z = x@self.weight+self.bias
        a = self.activation(z)
        return z, a

    def __str__(self) -> str:
        return 'dense'
