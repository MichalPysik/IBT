from random import random
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, name):
        self.name = name
        if name == 'Heart failure':
            self.filename = 'heart_failure_clinical_records_dataset.csv'
            self.rows = 12
        elif name == 'Rain in Australia':
            self.filename = 'weatherAUS_transformed_drop.csv'
            self.rows = 92
        else:
            return

        ds = pd.read_csv('datasets/' + self.filename)
        X = ds.iloc[:, :-1].values
        y = ds.iloc[:, -1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.15)
        print(self.filename, 'has been selected as active dataset')
        



class Callback(keras.callbacks.Callback): 
    def __init__(self, epochs):
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        print('Epoch: ' + str(epoch+1) + '/' + str(self.epochs) + '  loss: ' + str(logs['loss']) + '  accuracy: ' + str(logs['accuracy']))

            


class NeuralNet:
    def __init__(self):
        self.model = None
        self.dataset = None


    def setDataset(self, name):
        self.dataset = Dataset(name)


    def createModel(self, layers, neurons):
        self.model = Sequential()
        for i in range(layers):
            if i == 0:
                self.model.add(Dense(neurons[i], input_dim=self.dataset.rows, activation="relu")) # input layer
            elif i == layers - 1:
                self.model.add(Dense(neurons[i], activation="sigmoid")) # output layer
            else:
                self.model.add(Dense(neurons[i], activation="relu")) # hidden layer
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        print("Created neural network with", layers, "layers consisting of", neurons, "neurons")


    def trainModel(self, epochs, batch_size):
        if self.model is None:
            print("To train your model, please create a model first")
            return
        print("Training the model on", self.dataset.name, "dataset...")
        try:
            self.model.fit(self.dataset.X_train, self.dataset.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0, callbacks=[Callback(epochs)])
        except:
            print("Error: Model training failed, the current model was created for different dataset!")
            return
        print("Model training complete!")


    def saveModel(self, path):
        if self.model is None:
            print('Error: You have to create a model first before saving it to a file!')
            return
        self.model.save(path, save_format='h5')


    def loadModel(self, path):
        try:
            self.model = keras.models.load_model(path)
        except:
            print("Error: Cannot load model, invalid file selected!")
        





