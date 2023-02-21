import numpy as np
from keras import callbacks

# Global variables and some utility functions

window_width = 1400
window_height = 900

saved_weights_path = ("./saved_weights")

ask_save_text = "Do you really want to save weights of selected models to \'" + saved_weights_path + "\'?\nThis action will rewrite corresponding currently stored savefiles."
ask_load_text = "Do you really want to load weights of selected models from \'" + saved_weights_path + "\'?\nThe current weights of selected models will be lost."


def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


class TrainProgressCallback(callbacks.Callback):
    def __init__(self, epochs):
        self.epochs = epochs

    def on_train_begin(self, logs=None):
        print("\nStarting training...\n")

    def on_epoch_begin(self, epoch, logs=None):
        print('Starting epoch ' + str(epoch+1) + '/' + str(self.epochs) + "...")

    def on_epoch_end(self, epoch, logs=None):
        print('Epoch: ' + str(epoch+1) + '/' + str(self.epochs) + '  loss: ' + str(logs['loss']) + '  accuracy: ' + str(logs['accuracy']) + '\n')

    def on_train_end(self, logs=None):
        print("Training is finished.\n")
