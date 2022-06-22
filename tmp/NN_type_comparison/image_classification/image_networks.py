import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, SimpleRNN
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical

def datasetCommon():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train.astype("float32")/255
    X_test = X_test.astype("float32")/255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (X_train, y_train), (X_test, y_test)

# my own architecture
def buildMLP():
    model = Sequential()
    model.add(Dense(12, input_dim=784, activation='sigmoid'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    return model

# 784-feature vector
def datasetMLP():
    (X_train, y_train), (X_test, y_test) = datasetCommon()
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)
    return (X_train, y_train), (X_test, y_test)

# for mnist from https://keras.io/examples/vision/mnist_convnet/
def buildCNN():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    ''' ALTERNATIVE for fashion_mnist from https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-fashion-mnist-clothing-classification/
    model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])'''
    return model

# 28*28 pixel grayscale images
def datasetCNN():
    (X_train, y_train), (X_test, y_test) = datasetCommon()
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    return (X_train, y_train), (X_test, y_test)

# https://medium.com/machine-learning-algorithms/mnist-using-recurrent-neural-network-2d070a5915a2
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter1-keras-quick-tour/rnn-mnist-1.5.1.py
def buildRNN():
    model = Sequential()
    model.add(SimpleRNN(256, dropout=0.2, input_shape=(28, 28)))
    model.add(Dense(10, activation='softmax'))
    return model

 # 28 timesteps (columns) of 28 features (rows)
def datasetRNN():
    return datasetCommon()




(X_train, y_train), (X_test, y_test) = datasetCNN()

model = buildCNN()

#tf.keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, show_layer_names=False)
#model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

model.evaluate(X_test, y_test, batch_size=32)





