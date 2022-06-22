import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, SimpleRNN
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical

def datasetCommon():
    df = pd.read_csv('seeds_dataset.csv')
    X = df.drop(['id' ,'seedtype'], axis=1) # instances
    y = to_categorical(df['seedtype'])[:, 1:]  # classes (indexed 1..3 instead of 0..2, fixed easily)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return (X_train, y_train), (X_test, y_test)

def buildMLP():
    model = Sequential()
    model.add(Dense(12, input_dim=7, activation='sigmoid'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    return model

# standard csv set of features as each sample
def datasetMLP():
    return datasetCommon()

def buildCNN():
    model = Sequential()
    model.add(Conv1D(32, kernel_size=7, activation='relu', input_shape=(7, 1)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model

# single convolution over every sample (set of features)
def datasetCNN():
    (X_train, y_train), (X_test, y_test) = datasetCommon()
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    return (X_train, y_train), (X_test, y_test)

def buildRNN():
    model = Sequential()
    model.add(SimpleRNN(64, dropout=0.2, input_shape=(7, 1)))
    model.add(Dense(3, activation='softmax'))
    return model

# each sample -> 1 feature with 7 timesteps - shape (7, 1)
def datasetRNN():
    (X_train, y_train), (X_test, y_test) = datasetCommon()
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    return (X_train, y_train), (X_test, y_test)


(X_train, y_train), (X_test, y_test) = datasetMLP()

model = buildMLP()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=1)

model.evaluate(X_test, y_test, batch_size=1)







