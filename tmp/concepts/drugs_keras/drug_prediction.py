from keras.layers import Dense, Dropout
from keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

dataset = pd.read_csv('drug_consumption.csv')

X = dataset.loc[:, :'Sensation seeking']
y = dataset['Cannabis consumption']

print(y.value_counts)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X.shape[1]))
model.add(Dense(y.nunique(), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=500, batch_size=16, validation_data=(X_test, y_test))