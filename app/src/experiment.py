import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
#import NeuralNet


class Dataset:
    def __init__(self, data_type):
        if data_type == 'Tabular':
            self.name = 'Types of seeds'
            ds = pd.read_csv('datasets/seeds_dataset.csv')
            X = ds.iloc[:, :-1].values
            y = ds.iloc[:, -1].values
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.15)

        elif data_type == 'Image':
            self.name = 'Fashion MNIST'
            (self.X_train, self.y_train), (self.X_test, self.y_test) = fashion_mnist.load_data()

        else: # data_type == Sequence
            self.name = 'FordA engine sound'
            ds_train = pd.read_csv('datasets/FordA_TRAIN.tsv')
            ds_test = pd.read_csv('datasets/FordA_TEST.tsv')
            self.X_train = ds_train.iloc[:, 1:].values
            self.y_train = ds_train.iloc[:, 0].values
            self.X_test = ds_test.iloc[:, 1:].values
            self.y_test = ds_test.iloc[:, 0].values

        print(self.name, 'has been set as the active dataset.')


class Experiment:
    def __init__(self, data_type):
        self.data = Dataset(data_type)
        self.NNs = None

        