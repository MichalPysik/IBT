from random import random
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    
    def __init__(self, name):
        if name == 'heart':
            self.filename = 'heart_failure_clinical_records_dataset.csv'
            self.rows = 12
        else:
            return

        ds = pd.read_csv('datasets/' + self.filename)
        X = ds.iloc[:, -1].values
        y = ds.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.15)
        print(self.filename, ' is set')
        



            





class NeuralNet:

    def __init__(self):
        self.model = None
        self.dataset = None

    def setDataset(self, name):
        self.dataset = Dataset(name)

    def createModel(self, layers, neurons):
        self.model = Sequential()
        for i in range(layers):
            if i == 1:
                self.model.add(Dense(neurons[i], input_dim=self.dataset.rows, activation="relu"))
            else:
                self.model.add(Dense(neurons[i], activation="sigmoid"))
        self.model.compile(loss="binary crossentropy", optimizer="adam", metrics=["accuracy"])
        print("created network with ", layers, "layers consisting of ", neurons)






