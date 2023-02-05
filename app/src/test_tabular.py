# 7,500 +- 750 trainable params
# https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification

from architectures import createNetwork
import numpy as np
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split

#df_train = pd.read_csv('datasets/mobile_train.csv')
#df_test = pd.read_csv('datasets/mobile_train.csv')
#
#X_train = df_train.iloc[:, 0:-1].values
#y_train = df_train.iloc[:, -1].values
#
#X_test = df_test.iloc[:, 0:-1].values
#y_test = df_test.iloc[:, -1].values

df = pd.read_csv('datasets/MiniBooNE_particle.csv')

X = df.iloc[:, 0:-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.15)

num_classes = max(np.max(y_test), np.max(y_train)) - min(np.min(y_test), np.min(y_train)) + 1

model_MLP = createNetwork('Tabular', 0, X_train.shape[1:], num_classes)
model_MLPx = createNetwork('Tabular', 3, X_train.shape[1:], num_classes)
model_CNN = createNetwork('Tabular', 1, X_train.shape[1:], num_classes)
model_RNN = createNetwork('Tabular', 2, X_train.shape[1:], num_classes)

history_MLP = model_MLP.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))
history_MLPx = model_MLPx.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))
history_CNN = model_CNN.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))
history_RNN = model_RNN.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))