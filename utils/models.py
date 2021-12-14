import pickle
from typing import Any, Dict
import numpy as np
from .layers import Layer
from .optimizers import Optimizer
from .losses import Loss
from .utils import Progbar, MetricMean
from copy import copy, deepcopy


class Sequential(object):
    def __init__(self) -> None:
        self.layers = []

    def add(self, layer: Layer) -> None:
        self.input_shape = None
        if layer.input_shape is not None:
            self.input_shape = layer.input_shape
            layer.input_shape = self.input_shape
        else:
            layer.input_shape = self.layers[-1].unit
        layer.initialParam()
        self.layers.append(layer)

    def summary(self) -> None:
        list_param = []
        text = "%-20s %-20s %20s\n" % ("Layer", "Output shape", "Params")
        text += "===============================================================\n"
        text += "%-20s %-20s %20d\n" % ("Input",
                                        str(f'None, {self.layers[0].input_shape}'), 0)
        for i, layer in enumerate(self.layers):
            list_param.append(np.prod(layer.weight.shape)+layer.bias.shape[0])
            text += "______________________________________________________________\n"
            text += "%-20s %-20s %20d\n" % (layer.__str__()+'_%d' % i,
                                            str(f'None, {layer.unit}'), list_param[i])
        text += "______________________________________________________________\n"
        text += f"Total Params: {np.sum(list_param)}"
        print(text)

    def compile(self, optimizer: Optimizer, loss: Loss) -> None:
        self.optimizer = [copy(optimizer) for _ in self.layers]
        self.loss = loss
        self.accuracy = None
        if loss.__str__() == 'BinaryCrossentropy':
            self.accuracy = self.binary_accuracy
        if loss.__str__() == 'CategoricalCrossentropy':
            self.accuracy = self.softmax_accuracy

    def feed_forward(self, x):
        self.params = [(None, x)]
        final_result = None
        for layer in self.layers:
            z, x = layer(x)
            self.params.append((z, x))
            final_result = x
        return final_result

    def back_forward(self, y_true, y_pred):
        delta = []
        m = y_true.shape[0]
        error = self.loss.grad(y_true, y_pred)
        error = error / \
            m*self.layers[-1].activation.backward(self.params[-1][0])
        delta.append((self.params[-2][1].T @ error, np.mean(error)))
        for i in range(len(self.layers)-2, -1, -1):
            error = error @ self.layers[i+1].weight.T * \
                self.layers[i].activation.backward(
                    self.params[i+1][0])
            d = self.params[i][1].T @ error
            delta.append((d, np.mean(error)))
        num_layer = len(self.layers)
        for i in range(num_layer-1, -1, -1):
            self.layers[i].weight, self.layers[i].bias = \
                self.optimizer[i].update(
                    self.layers[i].weight,
                    self.layers[i].bias,
                    delta[num_layer-(i+1)][0],
                    delta[num_layer-(i+1)][1]
            )

    def fit(self,
            x: Any,
            y: Any,
            epochs=10,
            batch_size=32,
            validation_data=None,
            callback=None) -> Dict:

        if callback is not None:
            self.callback = callback
        hist = {'train_loss': [], 'val_loss': [],
                'train_acc': [], 'val_acc': []}
        num_batch = x.shape[0]//batch_size
        train_acc = MetricMean()
        train_loss = MetricMean()
        stateful_metrics = ['acc', 'loss']
        if validation_data is not None:
            val_x, val_y = validation_data
            stateful_metrics.extend(['val_acc', 'val_loss'])
            val_acc = MetricMean()
            val_loss = MetricMean()
        for i in range(epochs):
            print("\nEpoch {}/{}".format(i+1, epochs))
            prog = Progbar(num_batch, width=20,
                           stateful_metrics=stateful_metrics)
            train_loss.reset_state()
            train_acc.reset_state()
            if validation_data is not None:
                val_acc.reset_state()
                val_loss.reset_state()
            for j in range(num_batch):
                x_batch = x[j*batch_size:min(len(x), (j+1)*batch_size), :]
                y_batch = y[j*batch_size:min(len(x), (j+1)*batch_size), :]
                loss, acc = self.train_step(x_batch, y_batch)
                train_loss.update_state(loss)
                train_acc.update_state(acc)

                values = [('acc', train_acc.result()),
                          ('loss', train_loss.result())]

                if validation_data is not None:
                    y_pred = self.feed_forward(val_x)
                    loss = self.loss(val_y, y_pred)
                    acc = self.accuracy(y_pred, val_y)
                    val_acc.update_state(acc)
                    val_loss.update_state(loss)
                    values.extend([
                        ('val_acc', val_acc.result()),
                        ('val_loss', val_loss.result())
                    ])

                prog.add(1, values=values)
            hist['train_loss'].append(train_loss.result())
            hist['train_acc'].append(train_acc.result())
            if validation_data is not None:
                hist['val_loss'].append(val_loss.result())
                hist['val_acc'].append(val_acc.result())
            if callback:
                self.optimizer = callback.update(self, i)
        self.params = None
        return hist

    def softmax_accuracy(self, y_pred, y_true) -> float:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        y = y_pred-y_true
        num_false = np.count_nonzero(y)
        return 1-num_false/len(y)

    def binary_accuracy(self, y_pred, y_true) -> float:
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y = y_pred-y_true
        num_false = np.count_nonzero(y)
        return 1-num_false/len(y)

    def train_step(self, x, y) -> Any:
        y_pred = self.feed_forward(x)
        self.back_forward(y, y_pred)
        loss = self.loss(y, y_pred)
        acc = self.accuracy(y_pred, y)
        return loss, acc

    def predict(self, x):
        pred = self.feed_forward(x)
        return pred

    def save(self, file_name: str, save_optimizer=True) -> None:
        with open(file_name, 'wb') as f:
            if save_optimizer:
                pickle.dump(self, f)
            else:
                obj = deepcopy(self)
                del obj.optimizer
                pickle.dump(obj, f)

    @staticmethod
    def load_model(file_name: str) -> object:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
