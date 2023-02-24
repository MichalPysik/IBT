import numpy as np
import tensorflow as tf
from keras import callbacks, metrics

# Global variables and some utility functions

window_width = 1600
window_height = 900

saved_weights_path = "saved_weights"

ask_change_experiment_text = (
    "Do you really want to change the experiment?\nAll unsaved model weights will be lost.\n"
)
ask_save_text = (
    "Do you really want to save weights of selected models to '"
    + saved_weights_path
    + "'?\nThis action will rewrite corresponding currently stored savefiles.\n"
)
ask_load_text = (
    "Do you really want to load weights of selected models from '"
    + saved_weights_path
    + "'?\nThe current weights of selected models will be lost.\n"
)
clear_screen_text = "Do you really want to clear the screen?\nAll current text will be lost.\n"
help_text = ("This is a simple application for comparing neural networks.\n")

def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

class TrainProgressCallback(callbacks.Callback):
    def __init__(self, epochs, name, valid):
        self.epochs = epochs
        self.valid = valid
        self.name = name

    def on_train_begin(self, logs=None):
        print("\nStarting training of " + self.name + "...\n")

    def on_epoch_begin(self, epoch, logs=None):
        print("Beginning epoch " + str(epoch + 1) + " of training " + self.name + "...")

    def on_epoch_end(self, epoch, logs):
        print(
            "Epoch: "
            + str(epoch + 1)
            + "/"
            + str(self.epochs)
            + "  loss: "
            + str(logs["loss"])
            + "  val_loss: "
            + str(logs["val_loss"])
        )
        metrics = logs
        metrics.pop("loss", "val_loss")
        for metric in metrics.keys():
            if metric.startswith(("val_", "f1_")):
                continue
            print(metric + ": " + str(logs[metric]), end="  ")
            if self.valid:
                print("val_" + metric + ": " + str(logs["val_" + metric]))
        try:
            f1_score = 2 * (logs["f1_precision"] * logs["f1_recall"]) / (logs["f1_precision"] + logs["f1_recall"] + tf.keras.backend.epsilon())
            print("f1_score: " + str(f1_score), end="  ")
            if self.valid:
                val_f1_score = 2 * (logs["val_f1_precision"] * logs["val_f1_recall"]) / (logs["val_f1_precision"] + logs["val_f1_recall"] + tf.keras.backend.epsilon())
                print("val_f1_score: " + str(val_f1_score))
        except KeyError:
            pass
        print("" if self.valid else "\n")
        
    def on_train_end(self, logs=None):
        print("Training of " + self.name + " has finished.\n")

class TestProgressCallback(callbacks.Callback):
    def __init__(self, name):
        self.name = name

    def on_test_begin(self, logs=None):
        print("\nStarting testing of " + self.name + "...")

    def on_test_end(self, logs):
        metrics = logs
        for metric in metrics.keys():
            if metric.startswith("f1_"):
                continue
            print(metric + ": " + str(logs[metric]))
        try:
            f1_score = 2 * (logs["f1_precision"] * logs["f1_recall"]) / (logs["f1_precision"] + logs["f1_recall"] + tf.keras.backend.epsilon())
            print("f1_score: " + str(f1_score))
        except KeyError:
            pass
        print("") # New line
