import matplotlib.pyplot as plt
from utils.models import Sequential
from utils.layers import Dense
from utils.activations import Relu, Sigmoid
from utils.losses import BinaryCrossentropy
from utils.optimizers import SGD, Adagrad, RMSProp, Adam, LearningRateScheduler

import numpy as np
import pickle
import random
import cv2
from sklearn.model_selection import train_test_split

with open('./data.pkl', 'rb') as f:
    train_x, train_y = pickle.load(f)
order = [i for i in range(len(train_x))]
random.shuffle(order)
train_x = train_x[order]
train_y = train_y[order]

train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.1)


def lr_schedule(epoch, lr):
    if epoch < 5:
        return lr
    return lr * np.exp(-0.1)


callback = LearningRateScheduler(lr_schedule)

model = Sequential()
model.add(Dense(128, activation=Relu(), input_shape=128*128))
model.add(Dense(64, activation=Relu()))
model.add(Dense(32, activation=Relu()))
model.add(Dense(32, activation=Relu()))
model.add(Dense(1, activation=Sigmoid()))
model.summary()
model.compile(optimizer=Adam(), loss=BinaryCrossentropy())
hist = model.fit(train_x, train_y, epochs=10, batch_size=16,
                 validation_data=(val_x, val_y), callback=callback)

model.save('model.pkl', save_optimizer=False)

fig, axs = plt.subplots(2)
axs[0].plot(hist['train_acc'])
axs[0].plot(hist['val_acc'])
axs[0].set_title('model accuracy')
axs[0].set_ylabel('accuracy')
axs[0].set_xlabel('epoch')
axs[0].legend(['train', 'val'], loc='upper left')

axs[1].plot(hist['train_loss'])
axs[1].plot(hist['val_loss'])
axs[1].set_title('model loss')
axs[1].set_ylabel('loss')
axs[1].set_xlabel('epoch')
axs[1].legend(['train', 'val'], loc='upper left')
plt.show()

model = Sequential.load_model('./model.pkl')

img = cv2.imread('./2.jpg')
test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
test = cv2.resize(test, (128, 128))
test = test.reshape(1, 128*128)
p = model.predict(test)[0, 0]
label = 'cycle bike' if p > 0.5 else 'car'
test = test.reshape(128, 128)
img = cv2.putText(img, label, (50, 50),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
plt.show()
