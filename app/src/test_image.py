# 240,000 +- 24,000 trainable params

from architectures import createNetwork
from keras.datasets import fashion_mnist, mnist
import numpy as np
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

num_classes = max(np.max(y_test), np.max(y_train)) - min(np.min(y_test), np.min(y_train)) + 1

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)


#model_MLP = createNetwork('Image', 0, X_train.shape[1:], num_classes)
#model_CNN = createNetwork('Image', 1, X_train.shape[1:], num_classes)
model_RNN = createNetwork('Image', 2, X_train.shape[1:], num_classes)
#model_CNNx = createNetwork('Image', 3, X_train.shape[1:], num_classes)

#history_MLP = model_MLP.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))
#history_CNN = model_CNN.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))
history_RNN = model_RNN.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))
#history_CNNx = model_CNNx.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))


