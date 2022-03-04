from random import random
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class Dataset:
    def __init__(self, name):
        self.name = name
        if name == 'Heart failure':
            self.filename = 'heart_failure_clinical_records_dataset.csv'
            self.rows = 12
        elif name == 'Rain in Australia':
            self.filename = 'weatherAUS_transformed_drop.csv'
            self.rows = 92
        elif name == 'Wine quality':
            self.filename = 'WineQT.csv'
            self.rows = 11
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
        self.history = None


    def setDataset(self, name):
        self.dataset = Dataset(name)


    def createModel(self, layers, neurons):
        self.model = Sequential()
        for i in range(layers):
            if i == 0:
                self.model.add(Dense(neurons[i], input_dim=self.dataset.rows, activation="relu")) # input layer
            elif i == layers - 1:
                if self.dataset.name == 'Wine quality':
                    self.model.add(Dense(11, activation='softmax')) # output layer
                else:
                    self.model.add(Dense(1, activation="sigmoid")) # output layer
            else:
                self.model.add(Dense(neurons[i], activation="relu")) # hidden layer
        if self.dataset.name == 'Wine quality':
            self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        else:
            self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        print("Created neural network with", layers, "layers consisting of", neurons, "neurons")


    def trainModel(self, epochs, batch_size):
        if self.model is None:
            print("To train your model, please create a model first")
            return
        print("Training the model on", self.dataset.name, "dataset...")
        try:
            self.history = self.model.fit(self.dataset.X_train, self.dataset.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0, callbacks=[Callback(epochs)])
        except:
            print("Error: Model training failed, the current model was created for different dataset!")
            return
        print("Model training complete!")


    def showTrainLossGraph(self):
        if not self.history:
            pass
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.gcf().canvas.set_window_title('Training loss history graph')
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


    def showTrainAccGraph(self):
        if not self.history:
            pass
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        epochs = range(1, len(acc) + 1)
        plt.gcf().canvas.set_window_title('Training accuracy history graph')
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


    def saveModel(self, path):
        if self.model is None:
            print('Error: You have to create a model first before saving it to a file!')
            return
        self.model.save(path, save_format='h5')
        print(path, "saved succesfully!")


    def loadModel(self, path):
        try:
            self.model = keras.models.load_model(path)
        except:
            print("Error: Cannot load model, invalid file selected!")
            raise Exception("Invalid model file")
        print(path, "loaded succesfully!")
        





