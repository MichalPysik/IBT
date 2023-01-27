from architectures import createNetwork
from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_train.astype('float32') / 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

X_train_MLP = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test_MLP = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)



model_MLP = createNetwork('Image', 0, X_train_MLP.shape[1:], 10)
model_CNN = createNetwork('Image', 1, X_train.shape[1:], 10)

history_MLP = model_MLP.fit(X_train_MLP, y_train, batch_size=32, epochs=30, validation_data=(X_test_MLP, y_test))
history_CNN = model_CNN.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_test, y_test))

