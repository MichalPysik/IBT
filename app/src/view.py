import tkinter as tk
from tkinter import RIGHT, ttk
from tkinter.constants import HORIZONTAL, UNITS, VERTICAL
import utils as ut
from experiment import Experiment
import sys


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
        self.modelmenu.add_command(label="Load models", command=self.load_models_callback)
        self.modelmenu.add_command(label="Save models", command=self.save_models_callback)
        self.menubar.add_cascade(label="Model", menu=self.modelmenu)
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
        self.textWindow = tk.Text(self.container, width=500, height=500, state='disabled', bg='#121212', fg='#ffffff')
        sys.stdout = StdoutRedirector(self.textWindow)

        self.dsCBLabel = ttk.Label(self.leftFrame, text = "Selected experiment:")
        self.dsSelected = tk.StringVar(value="Click here to select")
        self.dsComboBox = ttk.Combobox(self.leftFrame, state='readonly', justify='center', textvariable=self.dsSelected)
        self.dsComboBox['values'] = ('Tabular', 'Image', 'Sequential')
        self.dsComboBox.bind('<<ComboboxSelected>>', self.dsComboBox_callback)

        self.selectNetworksLabel = ttk.Label(self.leftFrame, text = "Selected neural networks:")
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
        self.epochsSpinbox = tk.Spinbox(self.epochsBatchFrame, width=5, from_=1, to=99999)
        self.batchSizeLabel = tk.Label(self.epochsBatchFrame, text="Batch size:", anchor="e", width=11)
        self.bachSizeSpinbox = tk.Spinbox(self.epochsBatchFrame, width=5, from_=1, to=99999)
        self.trainLossVar = tk.IntVar()
        self.trainLossCheckbox = tk.Checkbutton(self.leftFrame, variable=self.trainLossVar, text="Show training loss graph", onvalue=1, offvalue=0)
        self.trainAccVar = tk.IntVar()
        self.trainAccCheckbox = tk.Checkbutton(self.leftFrame, variable=self.trainAccVar, text="Show training accuracy graph", onvalue=1, offvalue=0)
        self.trainModelButton = tk.Button(self.leftFrame, text="Train model", command=self.train_models_callback)


    def help_callback(self, event=None):
        print("This is help")


    def clear_screen_callback(self, event=None):
        self.textWindow.config(state='normal')
        self.textWindow.delete('1.0', 'end')
        self.textWindow.config(state='disabled')


    def train_models_callback(self, event=None):
        try:
            epochs = int(self.epochsSpinbox.get())
            if epochs < 1:
                raise Exception()
        except:
            print("Error: Number of epochs must be a positive integer!")
            return
        try:
            batch_size = int(self.bachSizeSpinbox.get())
            if batch_size < 1:
                raise Exception()
        except:
            print("Error: Batch size must be a positive integer!")
            return
        pass


    def dsComboBox_callback(self, event=None, setup=False):
        selected_networks = [self.network0Var.get(), self.network1Var.get(), self.network2Var.get(), self.network3Var.get()]
        self.experiment = Experiment(self.dsSelected.get(), selected_networks)


    def load_models_callback(self, event=None):
        loadPath = tk.filedialog.askopenfilename(filetypes=(("H5 file", "*.h5"),("All Files", "*.*") ))
        try:
            self.NN.loadModel(loadPath)
            self.savePath = loadPath
        except:
            pass


    def save_models_callback(self, event=None):
        if self.savePath is None:
            self.savePath = tk.filedialog.asksaveasfilename(defaultextension='.h5', filetypes=(("H5 file", "*.h5"),("All Files", "*.*") ))
        self.NN.saveModel(self.savePath)


    def setup_layout(self):
        self.leftFrame.pack(side=tk.LEFT, expand=True)
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
        self.bachSizeSpinbox.grid(row=1, column=1)
        self.trainLossCheckbox.pack()
        self.trainAccCheckbox.pack()
        self.trainModelButton.pack(pady=10)

        


