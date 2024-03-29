# Project: Classification with Use of Neural Networks in the Keras Environment
# Application: Experimental application for neural network comparison with use of Keras
# Author: Michal Pyšík
# File: view.py

import tkinter as tk
from tkinter import RIGHT, ttk, messagebox, filedialog
from tkinter.constants import HORIZONTAL, UNITS, VERTICAL
import utils as ut
from experiment import Experiment
import sys
from architectures import create_network, architecture_names
from tensorflow.keras import metrics


# Redirects a stream (stdout) to a given text widget, handles the widget for writing
# https://stackoverflow.com/questions/18517084/how-to-redirect-stdout-to-a-tkinter-text-widget
class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_space = text_widget

    def write(self, string):
        self.text_space.config(state="normal")
        self.text_space.insert("end", string)
        self.text_space.see("end")
        self.text_space.config(state="disabled")
        self.text_space.update_idletasks()

    def flush(self):
        pass


# Class encapsulating the graphical user interface
class View:
    # Initiates the GUI, including the initial context
    def __init__(self, parent):
        self.container = parent
        self.experiment = None
        self.selected_networks = []
        self.metrics = []
        self.val_metrics = False
        self.selected_plots = {
            "loss": False,
            "val_loss": False,
            "metrics": False,
            "val_metrics": False,
        }

    # Creates and sets up all the widgets used in the GUI
    def setup(self):
        self.create_widgets()
        self.setup_layout()

    # Updates the array of selected networks based on the checkboxes
    def update_selected_networks(self):
        self.selected_networks = [
            self.network0Var.get(),
            self.network1Var.get(),
            self.network2Var.get(),
            self.network3Var.get(),
        ]

    # Updates the array of selected metrics based on the checkboxes
    def update_selected_metrics(self):
        self.metrics = []
        if self.accuracyVar.get():
            if self.experiment.dataset.num_classes == 2:
                self.metrics.append(metrics.BinaryAccuracy(name="accuracy"))
            else:
                self.metrics.append(metrics.CategoricalAccuracy(name="accuracy"))
        if self.precisionVar.get():
            self.metrics.append(metrics.Precision(name="precission"))
        if self.recallVar.get():
            self.metrics.append(metrics.Recall(name="recall"))
        if self.aucPRVar.get():
            self.metrics.append(metrics.AUC(curve="PR", name="auc_pr"))
        if self.aucROCVar.get():
            self.metrics.append(metrics.AUC(curve="ROC", name="auc_roc"))
        self.val_metrics = self.metricsValidationVar.get()
        if self.f1scoreVar.get():
            self.metrics.append(metrics.Precision(name="f1_precision"))
            self.metrics.append(metrics.Recall(name="f1_recall"))

    # Updates the dictionary of selected plots based on the checkboxes
    def update_selected_plots(self):
        self.selected_plots["loss"] = self.plotLossVar.get()
        self.selected_plots["val_loss"] = self.plotValidationLossVar.get()
        self.selected_plots["metrics"] = self.plotMetricsVar.get()
        self.selected_plots["val_metrics"] = self.plotValidationMetricsVar.get()

    # Creates the widgets used in the GUI
    def create_widgets(self):
        # Top bar containing options for saving/loading weights,
        # handling the screen, and printing help
        self.menubar = tk.Menu(self.container)
        self.modelmenu = tk.Menu(self.menubar, tearoff=0)
        self.modelmenu.add_command(
            label="Save selected", command=self.save_models_callback
        )
        self.modelmenu.add_command(
            label="Load selected", command=self.load_models_callback
        )
        self.menubar.add_cascade(label="Models", menu=self.modelmenu)
        self.screenbar = tk.Menu(self.menubar, tearoff=0)
        self.screenbar.add_command(
            label="Save as...", command=self.save_screen_callback
        )
        self.screenbar.add_command(label="Clear", command=self.clear_screen_callback)
        self.menubar.add_cascade(label="Screen", menu=self.screenbar)
        self.menubar.add_command(label="Help", command=self.help_callback)

        # Keyboard shortcuts binding
        self.container.bind("<Control-o>", lambda event: self.load_models_callback())
        self.container.bind("<Control-O>", lambda event: self.load_models_callback())
        self.container.bind("<Control-s>", lambda event: self.save_models_callback())
        self.container.bind("<Control-S>", lambda event: self.save_models_callback())
        self.container.bind("<Control-h>", lambda event: self.help_callback())
        self.container.bind("<Control-H>", lambda event: self.help_callback())
        self.container.bind("<Control-l>", lambda event: self.clear_screen_callback())
        self.container.bind("<Control-L>", lambda event: self.clear_screen_callback())
        self.container.config(menu=self.menubar)

        # The left frame for most of the user input and the main text window
        self.leftFrame = tk.Frame(self.container)
        self.scrollbar = tk.Scrollbar(self.container, orient="vertical")
        self.textWindow = tk.Text(
            self.container,
            width=3840,
            height=2160,
            state="disabled",
            yscrollcommand=self.scrollbar.set,
            bg="#121212",
            fg="#ffffff",
        )
        sys.stdout = StdoutRedirector(self.textWindow)

        # ComboBox for selecting the current experiment
        self.dsCBLabel = ttk.Label(self.leftFrame, text="Selected experiment")
        self.dsSelected = tk.StringVar(value="Click here to select")
        self.dsComboBox = ttk.Combobox(
            self.leftFrame,
            state="readonly",
            justify="center",
            textvariable=self.dsSelected,
        )
        self.dsComboBox["values"] = ("Tabular", "Image", "Sequential")
        self.dsComboBox.bind("<<ComboboxSelected>>", self.dsComboBox_callback)

        # Checkboxes for selecting the active networks/models
        self.selectNetworksLabel = ttk.Label(self.leftFrame, text="Selected models")
        self.selectNetworksFrame = tk.Frame(self.leftFrame)
        self.network0Var = tk.BooleanVar(value=True)
        self.network1Var = tk.BooleanVar(value=True)
        self.network2Var = tk.BooleanVar(value=True)
        self.network3Var = tk.BooleanVar(value=True)
        self.network0Checkbox = tk.Checkbutton(
            self.selectNetworksFrame, variable=self.network0Var, text="MLP"
        )
        self.network1Checkbox = tk.Checkbutton(
            self.selectNetworksFrame, variable=self.network1Var, text="CNN"
        )
        self.network2Checkbox = tk.Checkbutton(
            self.selectNetworksFrame, variable=self.network2Var, text="RNN"
        )
        self.network3Checkbox = tk.Checkbutton(
            self.selectNetworksFrame, variable=self.network3Var, text="Extra"
        )

        # The frame containing checkboxes for selecting the active training metrics
        self.metricsFrame = tk.Frame(self.leftFrame, borderwidth=2, relief="sunken")
        self.metricsLabel = ttk.Label(self.metricsFrame, text="Metrics")
        self.accuracyVar = tk.BooleanVar(value=True)
        self.accuracyCheckbox = tk.Checkbutton(
            self.metricsFrame, variable=self.accuracyVar, text="Accuracy", anchor="w"
        )
        self.precisionVar = tk.BooleanVar(value=False)
        self.precisionCheckbox = tk.Checkbutton(
            self.metricsFrame, variable=self.precisionVar, text="Precission"
        )
        self.recallVar = tk.BooleanVar(value=False)
        self.recallCheckbox = tk.Checkbutton(
            self.metricsFrame, variable=self.recallVar, text="Recall"
        )
        self.aucPRVar = tk.BooleanVar(value=False)
        self.aucPRCheckbox = tk.Checkbutton(
            self.metricsFrame, variable=self.aucPRVar, text="AUC PR"
        )
        self.aucROCVar = tk.BooleanVar(value=False)
        self.aucROCCheckbox = tk.Checkbutton(
            self.metricsFrame, variable=self.aucROCVar, text="AUC ROC"
        )
        self.metricsValidationVar = tk.BooleanVar(value=True)
        self.metricsValidationCheckbox = tk.Checkbutton(
            self.metricsFrame, variable=self.metricsValidationVar, text="validation"
        )
        self.f1scoreVar = tk.BooleanVar(value=False)
        self.f1scoreCheckbox = tk.Checkbutton(
            self.metricsFrame, variable=self.f1scoreVar, text="F1 score"
        )

        # The frame containing checkboxes for selecting the training plots
        self.plotsFrame = tk.Frame(self.leftFrame, borderwidth=2, relief="raised")
        self.plotsLabel = ttk.Label(self.plotsFrame, text="Plots")
        self.plotLossVar = tk.BooleanVar(value=False)
        self.plotLossCheckbox = tk.Checkbutton(
            self.plotsFrame, variable=self.plotLossVar, text="Loss"
        )
        self.plotValidationLossVar = tk.BooleanVar(value=False)
        self.plotValidationLossCheckbox = tk.Checkbutton(
            self.plotsFrame, variable=self.plotValidationLossVar, text="valid."
        )
        self.plotMetricsVar = tk.BooleanVar(value=False)
        self.plotMetricsCheckbox = tk.Checkbutton(
            self.plotsFrame, variable=self.plotMetricsVar, text="Metrics"
        )
        self.plotValidationMetricsVar = tk.BooleanVar(value=False)
        self.plotValidationMetricsCheckbox = tk.Checkbutton(
            self.plotsFrame, variable=self.plotValidationMetricsVar, text="valid."
        )

        # Spinboxes for selecting the number of epochs and batch size
        self.epochsBatchFrame = tk.Frame(self.leftFrame)
        self.epochsLabel = tk.Label(
            self.epochsBatchFrame, text="Epochs:", anchor="e", width=11
        )
        self.epochsVar = tk.StringVar(value="10")
        self.epochsSpinbox = tk.Spinbox(
            self.epochsBatchFrame,
            width=5,
            from_=1,
            to=99999,
            textvariable=self.epochsVar,
        )
        self.batchSizeVar = tk.StringVar(value="128")
        self.batchSizeLabel = tk.Label(
            self.epochsBatchFrame, text="Batch size:", anchor="e", width=11
        )
        self.batchSizeSpinbox = tk.Spinbox(
            self.epochsBatchFrame,
            width=5,
            from_=1,
            to=99999,
            textvariable=self.batchSizeVar,
        )

        # Buttons for training and testing the selected models, checkbox for plotting the confusion matrices
        self.trainModelButton = tk.Button(
            self.leftFrame, text="Train models", command=self.train_models_callback
        )
        self.confMatrixVar = tk.BooleanVar(value=False)
        self.confMatrixCheckbox = tk.Checkbutton(
            self.leftFrame, variable=self.confMatrixVar, text="Confusion matrix"
        )
        self.testModelButton = tk.Button(
            self.leftFrame, text="Test models", command=self.test_models_callback
        )

    # Callback that prints the help text
    def help_callback(self, event=None):
        print(ut.help_text)

    # Callback for saving the current weights of the selected models
    def save_screen_callback(self, event=None):
        initialfile = "log.txt"
        if self.experiment:
            initialfile = self.experiment.data_type + "_" + initialfile
        filename = filedialog.asksaveasfilename(
            initialdir=".",
            title="Select file",
            filetypes=[("txt files", "*.txt")],
            initialfile=initialfile,
        )
        if filename:
            with open(filename, "w") as f:
                f.write(self.textWindow.get("1.0", "end"))

    # Callback that clears all current text in the main text window
    def clear_screen_callback(self, event=None):
        if not messagebox.askyesno("Clear screen", ut.clear_screen_text):
            return
        self.textWindow.config(state="normal")
        self.textWindow.delete("1.0", "end")
        self.textWindow.config(state="disabled")

    # Callback for training the selected models
    def train_models_callback(self, event=None):
        try:
            epochs = int(self.epochsVar.get())
            if epochs < 1:
                raise Exception()
        except:
            print("Error: Number of epochs must be a positive integer!")
            return
        try:
            batch_size = int(self.batchSizeVar.get())
            if batch_size < 1:
                raise Exception()
        except:
            print("Error: Batch size must be a positive integer!")
            return
        if not self.experiment:
            print(
                "Error: You have to select a dataset before training selected models!"
            )
            return

        self.update_selected_networks()
        if not any(self.selected_networks):
            print("Error: You have to select at least one model to train!")
            return
        self.update_selected_metrics()

        loss_fn = (
            "binary_crossentropy"
            if self.experiment.dataset.num_classes == 2
            else "categorical_crossentropy"
        )
        optimizer = "rmsprop" if self.experiment.data_type == "Sequential" else "adam"
        histories = []

        for i in range(len(self.selected_networks)):
            if not self.selected_networks[i]:
                histories.append(None)
                continue

            if self.experiment.data_type == "Sequential":
                if not i:
                    X_train, X_test = (
                        self.experiment.dataset.X_train_vectorized,
                        self.experiment.dataset.X_test_vectorized,
                    )
                else:
                    X_train, X_test = (
                        self.experiment.dataset.X_train_padded,
                        self.experiment.dataset.X_test_padded,
                    )
            else:
                X_train, X_test = (
                    self.experiment.dataset.X_train,
                    self.experiment.dataset.X_test,
                )

            if not self.experiment.networks[i]:
                input_shape = (
                    self.experiment.dataset.vectorized_sample_shape
                    if not i and self.experiment.data_type == "Sequential"
                    else self.experiment.dataset.padded_sample_shape
                    if self.experiment.data_type == "Sequential"
                    else self.experiment.dataset.sample_shape
                )
                self.experiment.networks[i] = create_network(
                    self.experiment.data_type,
                    i,
                    input_shape=input_shape,
                    num_classes=self.experiment.dataset.num_classes,
                    show_summary=False,
                )

            self.experiment.networks[i].compile(
                loss=loss_fn, optimizer=optimizer, metrics=self.metrics
            )
            history = self.experiment.networks[i].fit(
                X_train,
                self.experiment.dataset.y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, self.experiment.dataset.y_test),
                callbacks=[
                    ut.TrainProgressCallback(
                        epochs, self.experiment.networks[i].name, self.val_metrics
                    )
                ],
                verbose=0,
            )
            histories.append(history)

        self.update_selected_plots()
        if any(self.selected_plots):
            ut.plot_graphs(histories, self.selected_plots, self.experiment.data_type)

    # Callback for testing the selected models
    def test_models_callback(self, event=None):
        try:
            batch_size = int(self.batchSizeVar.get())
            assert batch_size >= 1
        except:
            print("Error: Batch size must be a positive integer!")
            return
        if not self.experiment:
            print("Error: You have to select a dataset before testing selected models!")
            return
        self.update_selected_networks()
        if not any(self.selected_networks):
            print("Error: You have to select at least one model to test!")
            return

        self.update_selected_metrics()
        loss_fn = (
            "binary_crossentropy"
            if self.experiment.dataset.num_classes == 2
            else "categorical_crossentropy"
        )
        optimizer = "rmsprop" if self.experiment.data_type == "Sequential" else "adam"

        for i in range(len(self.selected_networks)):
            if not self.selected_networks[i]:
                continue
            if self.experiment.data_type == "Sequential":
                if i == 0:  # MLP - vectorized
                    X_test = self.experiment.dataset.X_test_vectorized
                else:
                    X_test = self.experiment.dataset.X_test_padded
            else:
                X_test = self.experiment.dataset.X_test

            if not self.experiment.networks[i]:
                input_shape = (
                    self.experiment.dataset.vectorized_sample_shape
                    if not i and self.experiment.data_type == "Sequential"
                    else self.experiment.dataset.padded_sample_shape
                    if self.experiment.data_type == "Sequential"
                    else self.experiment.dataset.sample_shape
                )
                self.experiment.networks[i] = create_network(
                    self.experiment.data_type,
                    i,
                    input_shape=input_shape,
                    num_classes=self.experiment.dataset.num_classes,
                    show_summary=False,
                )

            self.experiment.networks[i].compile(
                loss=loss_fn, optimizer=optimizer, metrics=self.metrics
            )
            self.experiment.networks[i].evaluate(
                X_test,
                self.experiment.dataset.y_test,
                batch_size=batch_size,
                callbacks=[ut.TestCallback(self.experiment.networks[i].name)],
                verbose=0,
            )

            if self.confMatrixVar.get():
                y_pred = self.experiment.networks[i].predict(X_test, verbose=0)
                ut.plot_conf_matrix(
                    self.experiment.dataset.y_test,
                    y_pred,
                    self.experiment.data_type,
                    self.experiment.dataset.num_classes,
                    i,
                )

    # Callback for chaning the current experiment along with the whole context
    def dsComboBox_callback(self, event=None):
        self.update_selected_networks()
        if self.experiment:
            if not messagebox.askokcancel(
                title="Change experiment",
                message=ut.ask_change_experiment_text,
                icon=messagebox.WARNING,
            ):
                self.dsSelected.set(self.experiment.data_type)
                return
        self.experiment = Experiment(self.dsSelected.get(), self.selected_networks)

    # Callback for saving the weights of the selected models
    def save_models_callback(self, event=None):
        if not self.experiment:
            print(
                "Error: You have to select a dataset before saving weights of selected models!"
            )
            return
        if not messagebox.askokcancel(
            title="Save weights of selected models",
            message=ut.ask_save_text,
            icon=messagebox.WARNING,
        ):
            return

        self.update_selected_networks()
        saved = []
        for i in range(len(self.selected_networks)):
            if not self.selected_networks[i]:
                continue
            try:
                self.experiment.networks[i].save_weights(
                    ut.saved_weights_path
                    + "/"
                    + self.experiment.data_type
                    + "/"
                    + architecture_names[self.experiment.data_type][i]
                    + ".h5"
                )
                saved.append(architecture_names[self.experiment.data_type][i])
            except:
                print(
                    "Error: Could not save weights of "
                    + architecture_names[self.experiment.data_type][i]
                    + " as it hasn't been initialized yet."
                )

        print(
            "The current weights of "
            + ", ".join([name for name in saved])
            + " have been saved to '"
            + ut.saved_weights_path
            + "/"
            + self.experiment.data_type
            + "'."
        )

    # Callback for loading the weights of the selected models
    def load_models_callback(self, event=None):
        if not self.experiment:
            print(
                "Error: You have to select a dataset before loading saved weights of selected models!"
            )
            return
        if not messagebox.askokcancel(
            title="Load weights of selected models",
            message=ut.ask_load_text,
            icon=messagebox.WARNING,
        ):
            return

        self.update_selected_networks()
        loaded = []
        for i in range(len(self.selected_networks)):
            if not self.selected_networks[i]:
                continue
            try:
                if not self.experiment.networks[i]:
                    input_shape = (
                        self.experiment.dataset.vectorized_sample_shape
                        if not i and self.experiment.data_type == "Sequential"
                        else self.experiment.dataset.padded_sample_shape
                        if self.experiment.data_type == "Sequential"
                        else self.experiment.dataset.sample_shape
                    )
                    self.experiment.networks[i] = create_network(
                        self.experiment.data_type,
                        i,
                        input_shape=input_shape,
                        num_classes=self.experiment.dataset.num_classes,
                        show_summary=False,
                    )
                self.experiment.networks[i].load_weights(
                    ut.saved_weights_path
                    + "/"
                    + self.experiment.data_type
                    + "/"
                    + architecture_names[self.experiment.data_type][i]
                    + ".h5"
                )
                loaded.append(architecture_names[self.experiment.data_type][i])
            except:
                print(
                    "Error: Could not load weights of "
                    + architecture_names[self.experiment.data_type][i]
                    + " as no savefile had been found."
                )

        print(
            "The saved weights of "
            + ", ".join([name for name in loaded])
            + " have been loaded from '"
            + ut.saved_weights_path
            + "/"
            + self.experiment.data_type
            + "'."
        )

    # Sets up the layout of the GUI (widgets, frames, etc.)
    def setup_layout(self):
        # Overall layout
        self.leftFrame.pack(side=tk.LEFT, expand=True)
        self.scrollbar.pack(side=RIGHT, fill="y")
        self.scrollbar.config(command=self.textWindow.yview)
        self.textWindow.pack()

        # Dataset selection
        self.dsCBLabel.pack(pady=(10, 5))
        self.dsComboBox.pack(padx=10, pady=(0, 40))

        # Network selection
        self.selectNetworksLabel.pack(pady=(0, 5))
        self.selectNetworksFrame.pack(pady=(0, 40))
        self.network0Checkbox.grid(row=0, column=0, pady=0)
        self.network1Checkbox.grid(row=0, column=1, pady=0)
        self.network2Checkbox.grid(row=0, column=2, pady=0)
        self.network3Checkbox.grid(row=1, column=1, pady=(4, 0))

        # Metrics selection
        self.metricsFrame.pack(pady=(0, 20))
        self.metricsLabel.grid(row=0, column=0, pady=(5, 10))
        self.accuracyCheckbox.grid(row=1, column=0, sticky="w", pady=(0, 5))
        self.precisionCheckbox.grid(row=2, column=0, sticky="w", pady=(0, 5))
        self.recallCheckbox.grid(row=3, column=0, sticky="w", pady=(0, 5))
        self.aucPRCheckbox.grid(row=4, column=0, sticky="w", pady=(0, 5))
        self.aucROCCheckbox.grid(row=5, column=0, sticky="w", pady=(0, 5))
        self.f1scoreCheckbox.grid(row=6, column=0, sticky="w", pady=(0, 10))
        self.metricsValidationCheckbox.grid(row=7, column=0, sticky="w", pady=(0, 5))

        # Training plots selection
        self.plotsFrame.pack(pady=(0, 40))
        self.plotsLabel.grid(row=0, column=0, columnspan=2, pady=(5, 5))
        self.plotLossCheckbox.grid(row=1, column=0, sticky="w", pady=(0, 5))
        self.plotValidationLossCheckbox.grid(row=1, column=1, sticky="w", pady=(0, 5))
        self.plotMetricsCheckbox.grid(row=2, column=0, sticky="w", pady=(0, 5))
        self.plotValidationMetricsCheckbox.grid(
            row=2, column=1, sticky="w", pady=(0, 5)
        )

        # Epochs and batch size selection
        self.epochsBatchFrame.pack(pady=(0, 20))
        self.epochsLabel.grid(row=0, column=0, pady=(0, 10))
        self.epochsSpinbox.grid(row=0, column=1, pady=(0, 10))
        self.batchSizeLabel.grid(row=1, column=0, pady=(0, 0))
        self.batchSizeSpinbox.grid(row=1, column=1, pady=(0, 0))

        # Training and testing buttons, confusion matrix checkbox (testing)
        self.trainModelButton.pack(pady=(0, 20))
        self.testModelButton.pack(pady=(0, 10))
        self.confMatrixCheckbox.pack(pady=(0, 10))
