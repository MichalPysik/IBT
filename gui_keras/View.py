# inspired by https://www.youtube.com/watch?v=lmZpZCvMeEA&ab_channel=hnzlab
import tkinter as tk
from tkinter import RIGHT, ttk
from tkinter.constants import HORIZONTAL, UNITS, VERTICAL
from tkinter.filedialog import asksaveasfile
import Utils as ut
import NeuralNet
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
        self.hidden_layers = []
        self.NN = NeuralNet.NeuralNet()
        self.savePath = None


    def setup(self):
        self.create_widgets()
        self.setup_layout()


    def create_widgets(self):
        self.leftFrame = tk.Frame(self.container)
        self.textWindow = tk.Text(self.container, width=500, height=500, state='disabled', bg='#121212', fg='#ffffff')
        sys.stdout = StdoutRedirector(self.textWindow)

        self.dsCBLabel = ttk.Label(self.leftFrame, text = "Select active dataset:")
        self.dsComboBox = ttk.Combobox(self.leftFrame, state='readonly')
        self.dsComboBox['values'] = ('Heart failure', 'Rain in Australia')
        self.dsComboBox.bind('<<ComboboxSelected>>', self.dsComboBox_callback)
        self.dsComboBox.current(0)
        self.dsComboBox_callback()

        self.epochsBatchFrame = tk.Frame(self.leftFrame)
        self.epochsLabel = tk.Label(self.epochsBatchFrame, text="Epochs:", anchor="e", width=10)
        self.epochsEntry = tk.Entry(self.epochsBatchFrame, width=5)
        self.batchSizeLabel = tk.Label(self.epochsBatchFrame, text="Batch size:", anchor="e", width=10)
        self.bachSizeEntry = tk.Entry(self.epochsBatchFrame, width=5)
        self.trainModelButton = tk.Button(self.leftFrame, text="Train model", command=self.trainModelButton_callback)

        self.addLayerButton = tk.Button(self.leftFrame, text="Add hidden layer", command=self.addHiddenLayer)
        self.removeLayerButton = tk.Button(self.leftFrame, text="Remove hidden layer", command=self.removeHiddenLayer)
        self.inputLayerNum = tk.Scale(self.leftFrame, from_=1, to=32, orient=HORIZONTAL, length=200)
        self.hiddenLayersFrame = tk.Frame(self.leftFrame, height=1)
        self.outputLayerNum = tk.Scale(self.leftFrame, from_=1, to=32, orient=HORIZONTAL, length=200)

        self.createModelButton = tk.Button(self.leftFrame, text="Create model", command=self.createModelButton_callback)
        self.loadModelButton = tk.Button(self.leftFrame, text="Load model", command=self.loadModelButton_callback)
        self.saveModelButton = tk.Button(self.leftFrame, text="Save model", command=self.saveModelButton_callback)
        self.saveModelAsButton = tk.Button(self.leftFrame, text="Save model as...", command=self.saveModelAsButton_callback)


    def trainModelButton_callback(self, event=None):
        try:
            epochs = int(self.epochsEntry.get())
            if epochs < 1:
                raise Exception()
        except:
            print("Error: Number of epochs must be a positive integer!")
            return
        try:
            batch_size = int(self.bachSizeEntry.get())
            if batch_size < 1:
                raise Exception()
        except:
            print("Error: Batch size must be a positive integer!")
            return
        self.NN.trainModel(epochs, batch_size)


    def dsComboBox_callback(self, event=None):
        self.NN.setDataset(self.dsComboBox.get())


    def createModelButton_callback(self, event=None):
        neurons = []
        neurons.append(self.inputLayerNum.get())
        for hl in self.hidden_layers:
            neurons.append(hl.get())
        neurons.append(self.outputLayerNum.get())
        self.NN.createModel(2 + len(self.hidden_layers), neurons)


    def loadModelButton_callback(self, event=None):
        loadPath = tk.filedialog.askopenfilename(filetypes=(("H5 file", "*.h5"),("All Files", "*.*") ))
        self.NN.loadModel(loadPath)


    def saveModelButton_callback(self, event=None):
        if self.savePath is None:
            self.savePath = tk.filedialog.asksaveasfilename(defaultextension='.h5', filetypes=(("H5 file", "*.h5"),("All Files", "*.*") ))
        self.NN.saveModel(self.savePath)


    def saveModelAsButton_callback(self, event=None):
        self.savePath = tk.filedialog.asksaveasfilename(defaultextension='.h5', filetypes=(("H5 file", "*.h5"),("All Files", "*.*") ))
        self.NN.saveModel(self.savePath)


    def setup_layout(self):
        self.leftFrame.pack(side=tk.LEFT, expand=True)
        self.textWindow.pack()

        self.dsCBLabel.pack(pady=(10,5))
        self.dsComboBox.pack(pady=(0,10))
        self.epochsBatchFrame.pack()
        self.epochsLabel.grid(row=0, column=0)
        self.epochsEntry.grid(row=0, column=1)
        self.batchSizeLabel.grid(row=1, column=0)
        self.bachSizeEntry.grid(row=1, column=1)
        self.trainModelButton.pack(pady=10)
        self.addLayerButton.pack(pady=(10,5))
        self.removeLayerButton.pack(pady=5)

        self.inputLayerNum.pack()
        self.hiddenLayersFrame.pack()
        for hl in self.hidden_layers:
            hl.pack()
        self.outputLayerNum.pack()

        self.createModelButton.pack(pady=(15,10))
        self.loadModelButton.pack(pady=(10,5))
        self.saveModelButton.pack(pady=5)
        self.saveModelAsButton.pack(pady=(5,10))


    def addHiddenLayer(self):
        self.hidden_layers.append(tk.Scale(self.hiddenLayersFrame, from_=2, to=32, orient=HORIZONTAL, length=200))
        self.hidden_layers[-1].pack()


    def removeHiddenLayer(self):
        try:
            self.hidden_layers[-1].destroy()
            self.hidden_layers.pop()
            if len(self.hidden_layers) == 0:
                self.hiddenLayersFrame.config(height=1)
        except:
            pass

        


