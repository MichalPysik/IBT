import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist, imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from utils import vectorize_sequences


class Dataset:
    def __init__(self, data_type):
        if data_type == "Tabular":
            self.name = "MiniBooNE particle identification"
            df = pd.read_csv("datasets/MiniBooNE_particle.csv")
            X = df.iloc[:, 0:-1].values
            y = df.iloc[:, -1].values
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, shuffle=True, test_size=0.15
            )
            self.num_classes = (
                max(np.max(self.y_test), np.max(self.y_train))
                - min(np.min(self.y_test), np.min(self.y_train))
                + 1
            )
            self.sample_shape = self.X_train.shape[1:]

        elif data_type == "Image":
            self.name = "Fashion MNIST"
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
            self.X_train = X_train.astype("float32") / 255
            self.X_test = X_test.astype("float32") / 255
            self.num_classes = (
                max(np.max(y_test), np.max(y_train))
                - min(np.min(y_test), np.min(y_train))
                + 1
            )
            self.y_train = to_categorical(y_train, num_classes=self.num_classes)
            self.y_test = to_categorical(y_test, num_classes=self.num_classes)
            self.sample_shape = self.X_train.shape[1:]

        else:  # data_type == Sequential
            self.name = "IMDB movie review sentiment classification"
            self.top_words = 10000
            self.max_review_len = 500
            (X_train, self.y_train), (X_test, self.y_test) = imdb.load_data(
                num_words=self.top_words
            )
            self.num_classes = (
                max(np.max(self.y_test), np.max(self.y_train))
                - min(np.min(self.y_test), np.min(self.y_train))
                + 1
            )
            self.X_train_vectorized = vectorize_sequences(
                X_train, dimension=self.top_words
            )
            self.X_test_vectorized = vectorize_sequences(
                X_test, dimension=self.top_words
            )
            self.vectorized_sample_shape = self.X_train_vectorized.shape[1:]
            self.X_train_padded = pad_sequences(X_train, maxlen=self.max_review_len)
            self.X_test_padded = pad_sequences(X_test, maxlen=self.max_review_len)
            self.padded_sample_shape = self.X_train_padded.shape[1:]

        print(self.name, "has been set as the active dataset.")
