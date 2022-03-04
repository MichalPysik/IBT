import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

dataset = pd.read_csv('WineQT.csv').drop(['Id'], axis=1)

X = dataset.drop(['quality'], axis=1)
y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

y_train = to_categorical(y_train, 11)
y_test = to_categorical(y_test, 11)

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(11,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(11, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split=0.2)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


_, accuracy = model.evaluate(X_test, y_test)
print('Test accurracy percentage: %.2f' % (accuracy*100))







'''
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
'''