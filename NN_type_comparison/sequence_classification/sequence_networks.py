#dataset bibtex: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalAveragePooling1D, BatchNormalization, Flatten, Dropout, SimpleRNN, Input, ReLU
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical

def datasetCommon():
    df_train = pd.read_csv('./FordA/FordA_TRAIN.tsv', sep='\t', header=None)
    X_train = df_train.iloc[:, 1:]
    y_train = df_train.iloc[:, 0].replace(-1, 0)
    df_test = pd.read_csv('./FordA/FordA_TEST.tsv', sep='\t', header=None)
    X_test = df_test.iloc[:, 1:]
    y_test = df_test.iloc[:, 0].replace(-1, 0)
    return (X_train, y_train), (X_test, y_test)

def buildMLP():
    model = Sequential()
    model.add(Dense(64, activation='sigmoid', input_dim=500))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# 500 timesteps as 500 separate features
def datasetMLP():
    return datasetCommon()

# inspired by https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
def buildCNN():
    input_layer = Input(shape=(500, 1))
    conv1 = Conv1D(64, kernel_size=3, padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv2 = Conv1D(64, kernel_size=3, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    conv3 = Conv1D(64, kernel_size=3, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    gap = GlobalAveragePooling1D()(conv3)
    output_layer = Dense(1, activation='sigmoid')(gap)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model

# 1D convolution over timeseries
def datasetCNN():
    (X_train, y_train), (X_test, y_test) = datasetCommon()
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    return (X_train, y_train), (X_test, y_test)

def buildRNN():
    model = Sequential()
    model.add(SimpleRNN(256, dropout=0.2, input_shape=(500, 1)))
    model.add(Dense(1, activation='sigmoid'))
    return model

# single feature with 500 timesteps
def datasetRNN():
    (X_train, y_train), (X_test, y_test) = datasetCommon()
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    return (X_train, y_train), (X_test, y_test)


(X_train, y_train), (X_test, y_test) = datasetRNN()

model = buildRNN()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.1)

model.evaluate(X_test, y_test, batch_size=8)







