# 370,000 +- 37,000 trainable params

# MLP https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/keras-imdb-mlp.ipynb
# CNN https://bagheri365.github.io/blog/Sentiment-Analysis-of-IMDB-Movie-Reviews-using-Convolutional-Neural-Network-(CNN)-with-Hyperparameters-Tuning/
# RNN https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

from architectures import createNetwork
import numpy as np
from keras.utils import to_categorical
import pandas as pd
from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences

top_words = 10000
max_review_len = 500

def vectorize_sequences(sequences, dimension=top_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

num_classes = max(np.max(y_test), np.max(y_train)) - min(np.min(y_test), np.min(y_train)) + 1

#X_train_vectorized = vectorize_sequences(X_train)
#X_test_vectorized = vectorize_sequences(X_test)

X_train_padded = pad_sequences(X_train, maxlen=max_review_len)
X_test_padded = pad_sequences(X_test, maxlen=max_review_len)

#model_MLP = createNetwork('Sequence', 0, X_train_vectorized.shape[1:], num_classes, optimizer='rmsprop')
#model_CNN = createNetwork('Sequence', 1, X_train_padded.shape[1:], num_classes, optimizer='rmsprop')
#model_RNN = createNetwork('Sequence', 2, X_train.shape[1:], num_classes, optimizer='rmsprop')
model_RNNx = createNetwork('Sequence', 3, X_train.shape[1:], num_classes, optimizer='rmsprop')

#history_MLP = model_MLP.fit(X_train_vectorized, y_train, batch_size=128, epochs=50, validation_data=(X_test_vectorized, y_test))
#history_CNN = model_CNN.fit(X_train_padded, y_train, batch_size=128, epochs=50, validation_data=(X_test_padded, y_test))
#history_RNN = model_RNN.fit(X_train_padded, y_train, batch_size=128, epochs=50, validation_data=(X_test_padded, y_test))
history_RNNx = model_RNNx.fit(X_train_padded, y_train, batch_size=32, epochs=50, validation_data=(X_test_padded, y_test))


