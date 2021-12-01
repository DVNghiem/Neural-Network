from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.001) -> None:
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w, b, delta_w, delta_b):
        pass

    def __str__(self) -> str:
        return self.__name__


class SGD(Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.0) -> None:
        super().__init__(learning_rate=learning_rate)
        self.momentum = momentum
        self.v_w = None
        self.v_b = None

    def update(self, w, b, delta_w, delta_b):
        if self.v_w is None:
            self.v_w = np.zeros_like(w)
            self.v_b = np.zeros_like(b)
        self.v_w = self.momentum*self.v_w + self.learning_rate*delta_w
        self.v_b = self.momentum*self.v_b + self.learning_rate*delta_b
        w -= self.v_w
        b -= self.v_b
        return w, b


class Adagrad(Optimizer):
    def __init__(self, learning_rate=0.001, epsilon=1e-6) -> None:
        super().__init__(learning_rate=learning_rate)
        self.epsilon = epsilon
        self.G_w = None
        self.G_b = None

    def update(self, w, b, delta_w, delta_b):
        if self.G_w is None:
            self.G_w = np.zeros_like(delta_w)
            self.G_b = np.zeros_like(delta_b)
        self.G_w += delta_w**2
        self.G_b += delta_b**2

        w -= self.learning_rate*delta_w/(np.sqrt(self.G_w)+self.epsilon)
        b -= self.learning_rate*delta_b/(np.sqrt(self.G_b)+self.epsilon)

        return w, b


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, decay=0.9, epsilon=1e-07) -> None:
        super().__init__(learning_rate=learning_rate)
        self.decay = decay
        self.epsilon = epsilon
        self.G_w = None
        self.G_b = None

    def update(self, w, b, delta_w, delta_b):
        if self.G_w is None:
            self.G_w = np.zeros_like(delta_w)
            self.G_b = np.zeros_like(delta_b)

        self.G_w = self.decay*self.G_w+(1-self.decay)*delta_w**2
        self.G_b = self.decay*self.G_b+(1-self.decay)*delta_b**2

        w -= self.learning_rate*delta_w / (np.sqrt(self.G_w)+self.epsilon)
        b -= self.learning_rate*delta_b / (np.sqrt(self.G_b)+self.epsilon)

        return w, b


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8) -> None:
        super().__init__(learning_rate=learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None

    def update(self, w, b, delta_w, delta_b):
        if self.m_w is None:
            self.m_w = np.zeros_like(delta_w)
            self.m_b = np.zeros_like(delta_b)

            self.v_w = np.zeros_like(delta_w)
            self.v_b = np.zeros_like(delta_b)
        self.t += 1
        self.m_w = self.beta1*self.m_w+(1-self.beta1)*delta_w
        self.m_b = self.beta1*self.m_b+(1-self.beta1)*delta_b
        self.m_w_bar = self.m_w/(1-self.beta1**self.t)
        self.m_b_bar = self.m_b/(1-self.beta1**self.t)

        self.v_w = self.beta2*self.v_w + (1-self.beta2)*delta_w**2
        self.v_b = self.beta2*self.v_b + (1-self.beta2)*delta_b**2
        self.v_w_bar = self.v_w/(1-self.beta2**self.t)
        self.v_b_bar = self.v_b/(1-self.beta2**self.t)

        w -= self.learning_rate*self.m_w_bar / \
            (np.sqrt(self.v_w_bar)+self.epsilon)
        b -= self.learning_rate*self.m_b_bar / \
            (np.sqrt(self.v_b_bar)+self.epsilon)

        return w, b
