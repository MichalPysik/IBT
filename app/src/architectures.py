import keras
from keras import Sequential
import keras.layers as layers

# for each data type, networks with ids 0, 1, 2 are MLP, CNN, RNN
# network with id 3 is always additional network based on the current data type (e.g. another CNN for image data)
def createNetwork(data_type, network_id, input_shape, num_classes):
    model = Sequential()

    if data_type == 'Tabular':
        if network_id == 0:
            pass
            


    elif data_type == 'Image':
        if network_id == 0: # MLP
            model.add(layers.Dense(256, activation='relu', input_shape=input_shape))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.5))

        # https://keras.io/examples/vision/mnist_convnet/
        elif network_id == 1: # CNN
            model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
            model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(layers.Flatten())
            model.add(layers.Dropout(0.5))



    elif data_type == 'Sequence':
        pass


    model.add(layers.Dense(num_classes))
    if num_classes > 2:
        model.add(layers.Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(layers.Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model