from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras import regularizers
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')

X = dataset.iloc[:,:-1].values  # instances
y = dataset.iloc[:,-1].values   # classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=4222)


model = Sequential()
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01), activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=4, validation_split=0.2)

_, accuracy = model.evaluate(X_test, y_test)
print('Test accurracy percentage: %.2f' % (accuracy*100))