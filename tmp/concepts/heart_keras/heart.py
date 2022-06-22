from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')

X = dataset.drop(['DEATH_EVENT'], axis=1)  # instances
y = dataset['DEATH_EVENT']   # classes

col_names = list(X.columns)
s_scaler = preprocessing.StandardScaler()
X = s_scaler.fit_transform(X)
X = pd.DataFrame(X, columns=col_names)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=7)


model = Sequential()
model.add(Dense(16, kernel_initializer='uniform', activation='relu', input_dim=12))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

val_accuracy = np.mean(history.history['val_accuracy'])
print("\n%s: %.2f%%" % ('val_accuracy', val_accuracy*100))


history_df = pd.DataFrame(history.history)

plt.plot(history_df.loc[:, ['loss']], "#6daa9f", label='Training loss')
plt.plot(history_df.loc[:, ['val_loss']],"#774571", label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="best")

plt.show()


predictions = model.predict(X_test)
for i in range(len(predictions)):
    if predictions[i] > 0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0


cm = confusion_matrix(y_test, predictions)
print(cm)