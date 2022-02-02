from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

dataset = pd.read_csv('weatherAUS_transformed_drop.csv')

X = dataset.iloc[:,:-1].values  # instances
y = dataset.iloc[:,-1].values   # classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=794328)


model = Sequential()
model.add(Dense(24, input_dim=X.shape[1], activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=100, batch_size=32)

_, accuracy = model.evaluate(X_test, y_test)
print('Test accurracy percentage: %.2f' % (accuracy*100))