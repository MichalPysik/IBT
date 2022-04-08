import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

dataset = pd.read_csv('Car_transformed.csv')

X = dataset.drop(['choice'], axis=1)
y = dataset['choice']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = to_categorical(y_train, 6)
y_test = to_categorical(y_test, 6)

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2)