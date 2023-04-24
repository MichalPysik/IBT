# Project: Classification with Use of Neural Networks in the Keras Environment
# Application: Experimental application for neural network comparison with use of Keras
# Author: Michal Pyšík
# File: architectures.py

from tensorflow.keras import Sequential, layers

# Dictionary containing the names of the NN architectures
architecture_names = {
    "Tabular": ["Tabular_MLP", "Tabular_CNN", "Tabular_RNN", "Tabular_MLPx"],
    "Image": ["Image_MLP", "Image_CNN", "Image_RNN", "Image_CNNx"],
    "Sequential": [
        "Sequential_MLP",
        "Sequential_CNN",
        "Sequential_RNN",
        "Sequential_RNNx",
    ],
}

# Function that creates a given model and based on the given data type returns it
# For each data type, networks with ids 0, 1, 2, 3 are MLP, CNN, RNN, {MLP, CNN, RNN}x respectively
# show_summary dictates whether to print the Keras model summary to stdout
def create_network(
    data_type,
    network_id,
    input_shape,
    num_classes,
    show_summary=True,
    top_words=10000,
    max_review_len=500,
):
    model = Sequential()

    if data_type == "Tabular":
        if network_id == 0:  # Tabular_MLP
            model._name = "Tabular_MLP"
            model.add(layers.Flatten(input_shape=input_shape))
            model.add(layers.Dense(150, activation="relu"))
            model.add(layers.Dropout(0.1))

        elif network_id == 1:  # Tabular_CNN
            model._name = "Tabular_CNN"
            model.add(layers.Reshape((input_shape[0], 1), input_shape=input_shape))
            model.add(layers.Conv1D(32, kernel_size=3))
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.Dropout(0.1))
            model.add(layers.Flatten())
            model.add(layers.Dense(10, activation="relu"))
            model.add(layers.Dropout(0.1))

        elif network_id == 2:  # Tabular_RNN
            model._name = "Tabular_RNN"
            model.add(layers.Reshape((input_shape[0], 1), input_shape=input_shape))
            model.add(layers.LSTM(42, dropout=0.1))
            model.add(layers.Dropout(0.1))

        elif network_id == 3:  # Tabular_MLPx
            model._name = "Tabular_MLPx"
            model.add(layers.Flatten(input_shape=input_shape))
            model.add(layers.Dense(50, activation="relu"))
            model.add(layers.Dropout(0.4))
            model.add(layers.Dense(50, activation="relu"))
            model.add(layers.Dropout(0.4))
            model.add(layers.Dense(50, activation="relu"))
            model.add(layers.Dropout(0.4))

    elif data_type == "Image":
        if network_id == 0:  # Image_MLP
            model._name = "Image_MLP"
            model.add(layers.Flatten(input_shape=input_shape))
            model.add(layers.Dense(224, activation="relu"))
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(224, activation="relu"))
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(112, activation="relu"))
            model.add(layers.Dropout(0.3))

        # https://www.kaggle.com/code/gpreda/cnn-with-tensorflow-keras-for-fashion-mnist
        elif network_id == 1:  # Image_CNN
            model._name = "Image_CNN"
            model.add(
                layers.Reshape(
                    (input_shape[0], input_shape[1], 1), input_shape=input_shape
                )
            )
            model.add(
                layers.Conv2D(
                    32,
                    kernel_size=(3, 3),
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Dropout(0.25))
            model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Dropout(0.25))
            model.add(layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
            model.add(layers.Dropout(0.4))
            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation="relu"))
            model.add(layers.Dropout(0.3))

        elif network_id == 2:  # Image_RNN
            model._name = "Image_RNN"
            model.add(
                layers.LSTM(
                    180, input_shape=input_shape, dropout=0.3, return_sequences=True
                )
            )
            model.add(layers.LSTM(90, dropout=0.25))
            model.add(layers.Dropout(0.25))
            model.add(layers.Dense(90, activation="relu"))
            model.add(layers.Dropout(0.3))

        else:  # Image_CNNx
            model._name = "Image_CNNx"
            model.add(
                layers.Reshape(
                    (input_shape[0], input_shape[1], 1), input_shape=input_shape
                )
            )
            model.add(
                layers.Conv2D(
                    32,
                    kernel_size=(3, 3),
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Conv2D(128, kernel_size=(3, 3), activation="relu"))
            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation="relu"))

    else:  # data_type == "Sequential"
        if network_id == 0:  # Sequential_MLP
            model._name = "Sequential_MLP"
            model.add(layers.Flatten(input_shape=input_shape))
            model.add(layers.Dropout(0.35))
            model.add(layers.Dense(36, activation="relu"))
            model.add(layers.Dropout(0.35))
            model.add(layers.Dense(36, activation="relu"))
            model.add(layers.Dropout(0.35))

        elif network_id == 1:  # Sequential_CNN
            model._name = "Sequential_CNN"
            model.add(layers.Embedding(top_words, 32, input_length=max_review_len))
            model.add(layers.Dropout(0.5))
            model.add(
                layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")
            )
            model.add(layers.GlobalMaxPooling1D())
            model.add(layers.Dense(256, activation="relu"))
            model.add(layers.Dropout(0.5))

        elif network_id == 2:  # Sequential_RNN
            model._name = "Sequential_RNN"
            embed_vec_len = 32
            model.add(
                layers.Embedding(top_words, embed_vec_len, input_length=max_review_len)
            )
            model.add(layers.LSTM(100, dropout=0.75))
            model.add(layers.Dropout(0.75))

        else:  # Sequential_RNNx
            model._name = "Sequential_RNNx"
            embed_vec_len = 32
            model.add(
                layers.Embedding(top_words, embed_vec_len, input_length=max_review_len)
            )
            model.add(layers.LSTM(80, dropout=0.75, return_sequences=True))
            model.add(layers.LSTM(40, dropout=0.75))
            model.add(layers.Dropout(0.75))
            model.add(layers.Dense(40, activation="relu"))
            model.add(layers.Dropout(0.75))

    if num_classes > 2:
        model.add(layers.Dense(num_classes))
        model.add(layers.Activation("softmax"))
    else:
        model.add(layers.Dense(1))
        model.add(layers.Activation("sigmoid"))

    if show_summary:
        model.summary()
        print(" ")

    return model
