import keras
from keras import Sequential
import keras.layers as layers
import numpy as np

# for each data type, networks with ids 0, 1, 2 are MLP, CNN, RNN
# network with id 3 is always additional network based on the current data type (e.g. another CNN for image data)
def createNetwork(data_type, network_id, input_shape, num_classes, optimizer='adam'):
    model = Sequential()

    if data_type == 'Tabular':
        if network_id == 0: # MLP single hidden
            model.add(layers.Flatten(input_shape=input_shape))
            model.add(layers.Dense(150, activation='relu'))
            model.add(layers.Dropout(0.1))

        elif network_id == 1: # 1D CNN
            model.add(layers.Reshape((input_shape[0], 1), input_shape=input_shape))
            model.add(layers.Conv1D(32, kernel_size=3))
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.Dropout(0.1))
            model.add(layers.Flatten())
            model.add(layers.Dense(10, activation='relu'))
            model.add(layers.Dropout(0.1))

        elif network_id == 2: # LSTM
            model.add(layers.Reshape((input_shape[0], 1), input_shape=input_shape))
            model.add(layers.LSTM(42, dropout=0.1))
            model.add(layers.Dropout(0.1))

        elif network_id == 3: # MLP three hidden
            model.add(layers.Flatten(input_shape=input_shape))
            model.add(layers.Dense(50, activation='relu'))
            model.add(layers.Dropout(0.4))
            model.add(layers.Dense(50, activation='relu'))
            model.add(layers.Dropout(0.4))
            model.add(layers.Dense(50, activation='relu'))
            model.add(layers.Dropout(0.4))
            

    elif data_type == 'Image':
        if network_id == 0: # MLP
            model.add(layers.Flatten(input_shape=input_shape))
            model.add(layers.Dense(224, activation='relu'))
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(224, activation='relu'))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(112, activation='relu'))
            model.add(layers.Dropout(0.3))

        # https://www.kaggle.com/code/gpreda/cnn-with-tensorflow-keras-for-fashion-mnist
        elif network_id == 1: # CNN
            model.add(layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape))
            model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Dropout(0.25))
            model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Dropout(0.25))
            model.add(layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
            model.add(layers.Dropout(0.4))
            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(0.3))

        elif network_id == 2: # RNN
            model.add(layers.LSTM(180, input_shape=input_shape, dropout=0.3, return_sequences=True))
            model.add(layers.LSTM(90, dropout=0.25))
            model.add(layers.Dropout(0.25))
            model.add(layers.Dense(90, activation='relu'))
            model.add(layers.Dropout(0.3))

        else: # CNN 2 (unregularized)
            model.add(layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape))
            model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation='relu'))


    elif data_type == 'Sequence':
        # https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/keras-imdb-mlp.ipynb
        if network_id == 0: # MLP
            model.add(layers.Flatten(input_shape=input_shape))
            model.add(layers.Dropout(0.4))
            model.add(layers.Dense(36, activation='relu'))
            model.add(layers.Dropout(0.4))
            model.add(layers.Dense(36, activation='relu'))
            model.add(layers.Dropout(0.4))
        
        # https://bagheri365.github.io/blog/Sentiment-Analysis-of-IMDB-Movie-Reviews-using-Convolutional-Neural-Network-(CNN)-with-Hyperparameters-Tuning/
        elif network_id == 1: # CNN
            model.add(layers.Embedding(10000, 32, input_length=500))
            model.add(layers.Dropout(0.4))
            model.add(layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'))
            model.add(layers.GlobalMaxPooling1D())
            model.add(layers.Dense(256, activation='relu'))
            model.add(layers.Dropout(0.4))

        elif network_id == 2: # RNN LSTM
            embed_vec_len = 32
            model.add(layers.Embedding(10000, embed_vec_len, input_length=500))
            model.add(layers.LSTM(100, dropout=0.3))
            model.add(layers.Dropout(0.3))

        else: # RNN Deep LSTM
            embed_vec_len = 32
            model.add(layers.Embedding(10000, embed_vec_len, input_length=500))
            model.add(layers.LSTM(80, dropout=0.3, return_sequences=True))
            model.add(layers.LSTM(40, dropout=0.2))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(40, activation='relu'))
            model.add(layers.Dropout(0.3))



    if num_classes > 2:
        model.add(layers.Dense(num_classes))
        model.add(layers.Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    else:
        model.add(layers.Dense(1))
        model.add(layers.Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()
    return model