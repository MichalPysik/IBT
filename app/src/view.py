import tkinter as tk
from tkinter import RIGHT, ttk, messagebox
from tkinter.constants import HORIZONTAL, UNITS, VERTICAL
import utils as ut
from experiment import Experiment
import sys
from architectures import create_network, architecture_names


# https://stackoverflow.com/questions/18517084/how-to-redirect-stdout-to-a-tkinter-text-widget
class StdoutRedirector(object):
    def __init__(self,text_widget):
        self.text_space = text_widget

    def write(self,string):
        self.text_space.config(state='normal')
        self.text_space.insert('end', string)
        self.text_space.see('end')
        self.text_space.config(state='disabled')
        self.text_space.update_idletasks()

    def flush(self):
        pass


class View:
    def __init__(self, parent):
        self.container = parent
        self.experiment = None
        self.metrics = ["accuracy"] # None, later get all in setup


    def setup(self):
        self.create_widgets()
        self.setup_layout()


    def create_widgets(self):
        self.menubar = tk.Menu(self.container)
        self.modelmenu = tk.Menu(self.menubar, tearoff=0)
        self.modelmenu.add_command(label="Save selected", command=self.save_models_callback)
        self.modelmenu.add_command(label="Load selected", command=self.load_models_callback)
        self.menubar.add_cascade(label="Models", menu=self.modelmenu)
        self.menubar.add_command(label="Help", command=self.help_callback)
        self.menubar.add_command(label="Clear screen", command=self.clear_screen_callback)
        self.container.bind('<Control-o>', lambda event:self.load_models_callback())
        self.container.bind('<Control-O>', lambda event:self.load_models_callback())
        self.container.bind('<Control-s>', lambda event:self.save_models_callback())
        self.container.bind('<Control-S>', lambda event:self.save_models_callback())
        self.container.bind('<Control-h>', lambda event:self.help_callback())
        self.container.bind('<Control-H>', lambda event:self.help_callback())
        self.container.bind('<Control-l>', lambda event:self.clear_screen_callback())
        self.container.bind('<Control-L>', lambda event:self.clear_screen_callback())
        self.container.config(menu=self.menubar)

        self.leftFrame = tk.Frame(self.container)
        self.scrollbar = tk.Scrollbar(self.container, orient='vertical')
        self.textWindow = tk.Text(self.container, width=500, height=500, state='disabled', yscrollcommand=self.scrollbar.set, bg='#121212', fg='#ffffff')
        sys.stdout = StdoutRedirector(self.textWindow)

        self.dsCBLabel = ttk.Label(self.leftFrame, text = "Selected experiment:")
        self.dsSelected = tk.StringVar(value="Click here to select")
        self.dsComboBox = ttk.Combobox(self.leftFrame, state='readonly', justify='center', textvariable=self.dsSelected)
        self.dsComboBox['values'] = ('Tabular', 'Image', 'Sequential')
        self.dsComboBox.bind('<<ComboboxSelected>>', self.dsComboBox_callback)

        self.selectNetworksLabel = ttk.Label(self.leftFrame, text = "Selected models:")
        self.selectNetworksFrame = tk.Frame(self.leftFrame)
        self.network0Var = tk.BooleanVar(value=True)
        self.network1Var = tk.BooleanVar(value=True)
        self.network2Var = tk.BooleanVar(value=True)
        self.network3Var = tk.BooleanVar(value=True)
        self.network0Checkbox = tk.Checkbutton(self.selectNetworksFrame, variable=self.network0Var, text="MLP")
        self.network1Checkbox = tk.Checkbutton(self.selectNetworksFrame, variable=self.network1Var, text="CNN")
        self.network2Checkbox = tk.Checkbutton(self.selectNetworksFrame, variable=self.network2Var, text="RNN")
        self.network3Checkbox = tk.Checkbutton(self.selectNetworksFrame, variable=self.network3Var, text="Extra")

        self.epochsBatchFrame = tk.Frame(self.leftFrame)
        self.epochsLabel = tk.Label(self.epochsBatchFrame, text="Epochs:", anchor="e", width=11)
        self.epochsVar = tk.StringVar(value="10")
        self.epochsSpinbox = tk.Spinbox(self.epochsBatchFrame, width=5, from_=1, to=99999, textvariable=self.epochsVar)
        self.batchSizeVar = tk.StringVar(value="128")
        self.batchSizeLabel = tk.Label(self.epochsBatchFrame, text="Batch size:", anchor="e", width=11)
        self.batchSizeSpinbox = tk.Spinbox(self.epochsBatchFrame, width=5, from_=1, to=99999, textvariable=self.batchSizeVar)

        self.trainModelButton = tk.Button(self.leftFrame, text="Train models", command=self.train_models_callback)
        self.testModelButton = tk.Button(self.leftFrame, text="Test models", command=self.test_models_callback)


    def help_callback(self, event=None):
        print("This is help")


    def clear_screen_callback(self, event=None):
        self.textWindow.config(state='normal')
        self.textWindow.delete('1.0', 'end')
        self.textWindow.config(state='disabled')


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
        loss_fn = 'binary_crossentropy' if self.experiment.dataset.num_classes == 2 else 'categorical_crossentropy'
        optimizer = 'rmsprop' if self.experiment.data_type == 'Sequential' else 'adam'
        selected_networks = [self.network0Var.get(), self.network1Var.get(), self.network2Var.get(), self.network3Var.get()]
        for i in range(len(selected_networks)):
            if selected_networks[i]:
                if self.experiment.data_type == "Sequential":
                    if not i:
                        X_train, X_test = self.experiment.dataset.X_train_vectorized, self.experiment.dataset.X_test_vectorized
                    else:
                        X_train, X_test = self.experiment.dataset.X_train_padded, self.experiment.dataset.X_test_padded
                else:
                    X_train, X_test = self.experiment.dataset.X_train, self.experiment.dataset.X_test
                if not self.experiment.networks[i]:
                    input_shape = self.experiment.dataset.vectorized_sample_shape if not i and self.experiment.data_type == 'Sequential' else self.experiment.dataset.padded_sample_shape if self.experiment.data_type == 'Sequential'else self.experiment.dataset.sample_shape
                    self.experiment.networks[i] = create_network(self.experiment.data_type, i, input_shape=input_shape, num_classes=self.experiment.dataset.num_classes, show_summary=False)
                self.experiment.networks[i].compile(loss=loss_fn, optimizer=optimizer, metrics=self.metrics)
                self.experiment.networks[i].fit(
                    X_train,
                    self.experiment.dataset.y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, self.experiment.dataset.y_test),
                    callbacks=[ut.TrainProgressCallback(epochs)],
                    verbose=0
                )

    def test_models_callback(self, event=None):
        try:
            batch_size = int(self.batchSizeVar.get())
            if batch_size < 1:
                raise Exception()
        except:
            print("Error: Batch size must be a positive integer!")
            return
        loss_fn = 'binary_crossentropy' if self.experiment.dataset.num_classes == 2 else 'categorical_crossentropy'
        optimizer = 'rmsprop' if self.experiment.data_type == 'Sequential' else 'adam'
        selected_networks = [self.network0Var.get(), self.network1Var.get(), self.network2Var.get(), self.network3Var.get()]
        for i in range(len(selected_networks)):
            if selected_networks[i]:
                if self.experiment.data_type == "Sequential":
                    if not i:
                        X_test = self.experiment.dataset.X_test_vectorized
                    else:
                        X_test = self.experiment.dataset.X_test_padded
                else:
                    X_test = self.experiment.dataset.X_test
                if not self.experiment.networks[i]:
                    input_shape = self.experiment.dataset.vectorized_sample_shape if not i and self.experiment.data_type == 'Sequential' else self.experiment.dataset.padded_sample_shape if self.experiment.data_type == 'Sequential'else self.experiment.dataset.sample_shape
                    self.experiment.networks[i] = create_network(self.experiment.data_type, i, input_shape=input_shape, num_classes=self.experiment.dataset.num_classes, show_summary=False)
                self.experiment.networks[i].compile(loss=loss_fn, optimizer=optimizer, metrics=self.metrics)
                self.experiment.networks[i].evaluate(
                    X_test,
                    self.experiment.dataset.y_test,
                    batch_size=batch_size,
                    verbose='auto'
                )


    def dsComboBox_callback(self, event=None):
        selected_networks = [self.network0Var.get(), self.network1Var.get(), self.network2Var.get(), self.network3Var.get()]
        self.experiment = Experiment(self.dsSelected.get(), selected_networks)


    
    def save_models_callback(self, event=None):
        if not self.experiment:
            print("You have to select a dataset before loading stored weights of selected models.")
        if not messagebox.askokcancel(title="Save weights of selected models", message=ut.ask_save_text, icon=messagebox.WARNING):
            return
        selected_networks = [self.network0Var.get(), self.network1Var.get(), self.network2Var.get(), self.network3Var.get()]
        for i in range(len(selected_networks)):
            if not selected_networks[i]:
                continue
            self.experiment.networks[i].save_weights(ut.saved_weights_path + "/" + self.experiment.data_type + "/" + architecture_names[self.experiment.data_type][i] + ".h5")
        print("The current weights of " + ", ".join([name for index,name in enumerate(architecture_names[self.experiment.data_type]) if selected_networks[index]]) + " have been saved successfully saved to \'" + ut.saved_weights_path + "/" + self.experiment.data_type + "\'.")



    def load_models_callback(self, event=None):
        if not messagebox.askokcancel(title="Load weights of selected models", message=ut.ask_load_text, icon=messagebox.WARNING):
            return


    def setup_layout(self):
        self.leftFrame.pack(side=tk.LEFT, expand=True)
        self.scrollbar.pack(side=RIGHT, fill='y')
        self.scrollbar.config(command=self.textWindow.yview)
        self.textWindow.pack()
        

        self.dsCBLabel.pack(pady=(10,5))
        self.dsComboBox.pack(pady=(0,10))

        self.selectNetworksLabel.pack(pady=(10,5))
        self.selectNetworksFrame.pack()
        self.network0Checkbox.grid(row=0, column=0)
        self.network1Checkbox.grid(row=0, column=1)
        self.network2Checkbox.grid(row=0, column=2)
        self.network3Checkbox.grid(row=0, column=3)

        self.epochsBatchFrame.pack()
        self.epochsLabel.grid(row=0, column=0)
        self.epochsSpinbox.grid(row=0, column=1)
        self.batchSizeLabel.grid(row=1, column=0)
        self.batchSizeSpinbox.grid(row=1, column=1)

        self.trainModelButton.pack(pady=5)
        self.testModelButton.pack(pady=5)


        


