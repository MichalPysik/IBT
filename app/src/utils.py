# Project: Classification with Use of Neural Networks in the Keras Environment
# Application: Experimental application for neural network comparison with use of Keras
# Author: Michal Pyšík
# File: utils.py

import numpy as np
import tensorflow as tf
import time
import itertools
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from architectures import architecture_names

# Config variables, Keras callbacks, and utility functions

window_width = 1600
window_height = 900

saved_weights_path = "saved_weights"
plots_path = "plots"

plot_colors = ["red", "blue", "green", "#ff8c00"]

ask_change_experiment_text = "Do you really want to change the experiment?\nAll unsaved model weights will be lost.\n"
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
clear_screen_text = (
    "Do you really want to clear the screen?\nAll current text will be lost.\n"
)
help_text = open("instructions.txt", "r").read()
fashion_mnist_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
imdb_review_labels = ["Negative", "Positive"]

# Vectorizes the given sequences of samples (one-hot encoding)
# Only used by Sequential_MLP by default
def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


# Custom Keras callback for reporting training progress
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
            + "\nloss: "
            + str(logs["loss"])
            + "  val_loss: "
            + str(logs["val_loss"])
        )
        for metric in logs.keys():
            if metric.startswith(("val_", "f1_")) or metric == "loss":
                continue
            print(metric + ": " + str(logs[metric]), end="  ")
            if self.valid:
                print("val_" + metric + ": " + str(logs["val_" + metric]))
        try:
            f1_score = (
                2
                * (logs["f1_precision"] * logs["f1_recall"])
                / (
                    logs["f1_precision"]
                    + logs["f1_recall"]
                    + tf.keras.backend.epsilon()
                )
            )
            print("f1_score: " + str(f1_score), end="  ")
            if self.valid:
                val_f1_score = (
                    2
                    * (logs["val_f1_precision"] * logs["val_f1_recall"])
                    / (
                        logs["val_f1_precision"]
                        + logs["val_f1_recall"]
                        + tf.keras.backend.epsilon()
                    )
                )
                print("val_f1_score: " + str(val_f1_score))
        except KeyError:
            pass

        print("" if self.valid or len(logs.keys()) <= 2 else "\n")

    def on_train_end(self, logs=None):
        print("Training of " + self.name + " has finished.\n")


# Custom Keras callback for reporting testing progress
class TestCallback(callbacks.Callback):
    def __init__(self, name):
        self.name = name

    def on_test_begin(self, logs=None):
        print("\nStarting testing of " + self.name + "...")

    def on_test_end(self, logs):
        for metric in logs.keys():
            if metric.startswith("f1_"):
                continue
            print(metric + ": " + str(logs[metric]))

        try:
            f1_score = (
                2
                * (logs["f1_precision"] * logs["f1_recall"])
                / (
                    logs["f1_precision"]
                    + logs["f1_recall"]
                    + tf.keras.backend.epsilon()
                )
            )
            print("f1_score: " + str(f1_score))
        except KeyError:
            pass

        print("")  # New line


# Calculates the f1 score from its temporary metrics
def transform_f1(history):
    for prefix in ["", "val_"]:
        precision = np.array(history.history[prefix + "f1_precision"])
        recall = np.array(history.history[prefix + "f1_recall"])
        f1_score = (
            2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        ).tolist()
        del history.history[prefix + "f1_precision"]
        del history.history[prefix + "f1_recall"]
        history.history[prefix + "f1_score"] = f1_score

    return history


# Plots the selected training graphs after training
def plot_graphs(histories, selected_plots, data_type):
    plt.rcParams.update({"font.size": 25})
    first = histories.index(
        next(filter(lambda x: x is not None, histories))
    )  # index of first non-empty history

    if "f1_precision" in histories[first].history.keys():
        for i in range(len(histories)):
            if histories[i] is not None:
                histories[i] = transform_f1(histories[i])

    for metric in histories[first].history.keys():
        if metric == "loss" and not selected_plots["loss"]:
            continue
        elif metric == "val_loss" and not selected_plots["val_loss"]:
            continue
        elif (
            metric.startswith("val_")
            and metric != "val_loss"
            and not selected_plots["val_metrics"]
        ):
            continue
        elif (
            not selected_plots["metrics"]
            and metric != "loss"
            and not metric.startswith("val_")
        ):
            continue

        plt.figure(figsize=(16, 12))
        plt.title(data_type + "_" + metric)
        plt.xlabel("epochs")
        plt.ylabel(metric)

        network_id = 0
        for i in range(len(histories)):
            if histories[i] == None:
                network_id += 1
                continue
            plt.plot(
                histories[i].epoch,
                histories[i].history[metric],
                color=plot_colors[network_id],
                label=architecture_names[data_type][network_id],
            )
            network_id += 1

        plt.legend()
        plt.grid(axis="both")
        plt.savefig(
            plots_path
            + "/"
            + data_type
            + "/"
            + metric
            + "_"
            + time.strftime("%Y-%m-%d_%H:%M:%S")
            + ".png",
            bbox_inches="tight",
        )
        plt.close()


# Plots the confusion matrix after testing
def plot_conf_matrix(y_true, y_pred, data_type, num_classes, network_id):
    plt.rcParams.update({"font.size": 10})
    labels = [str(i) for i in range(num_classes)]
    if num_classes == 2:
        y_pred = (y_pred > 0.5).astype(int)
        if data_type == "Sequential":
            labels = imdb_review_labels
    else:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        if data_type == "Image":
            labels = fashion_mnist_labels

    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("confusion_matrix_" + architecture_names[data_type][network_id])
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    rotation = 45 if num_classes > 2 and labels is not None else 0
    plt.xticks(tick_marks, labels, rotation=rotation)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(
        plots_path
        + "/"
        + data_type
        + "/"
        + "confusion_matrix_"
        + architecture_names[data_type][network_id]
        + "_"
        + time.strftime("%Y-%m-%d_%H:%M:%S")
        + ".png",
        bbox_inches="tight",
    )
    plt.close()
