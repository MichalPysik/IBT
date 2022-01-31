# inspired by https://www.youtube.com/watch?v=lmZpZCvMeEA&ab_channel=hnzlab
import tkinter as tk
from tkinter import ttk
from tkinter.constants import HORIZONTAL, UNITS, VERTICAL
import Utils as ut
import NeuralNet





class View:
    def __init__(self, parent):
        self.container = parent
        self.hidden_layers = []
        self.NN = NeuralNet.NeuralNet()

    def setup(self):
        self.create_widgets()
        self.setup_layout()

    def create_widgets(self):
        self.leftFrame = tk.Frame(self.container, width=250, height=ut.window_height)
        self.rightFrame = tk.Frame(self.container, width=700, height=ut.window_height, bg='#ffffff')

        self.dsComboBox = ttk.Combobox(self.leftFrame)
        self.dsComboBox['values'] = ('heart', 'Australia')
        self.dsComboBox['state'] = 'readonly'
        self.dsComboBox.bind('<<ComboboxSelected>>', self.NN.setDataset(self.dsComboBox.get())) #TODO DOES NOT WORK
        self.dsComboBox.set('heart')

        self.addLayerButton = tk.Button(self.leftFrame, text="Add hidden layer", command=self.addHiddenLayer)
        self.removeLayerButton = tk.Button(self.leftFrame, text="Remove hidden layer", command=self.removeHiddenLayer)
        self.inputLayerNum = tk.Scale(self.leftFrame, from_=1, to=32, orient=HORIZONTAL, length=200)
        self.hiddenLayersFrame = tk.Frame(self.leftFrame, height=1)
        self.outputLayerNum = tk.Scale(self.leftFrame, from_=1, to=32, orient=HORIZONTAL, length=200)


    def setup_layout(self):
        self.leftFrame.pack(side=tk.LEFT, expand=True)
        self.rightFrame.pack(side=tk.RIGHT)

        self.dsComboBox.pack()
        self.addLayerButton.pack()
        self.removeLayerButton.pack()

        self.inputLayerNum.pack()
        self.hiddenLayersFrame.pack()
        for hl in self.hidden_layers:
            hl.pack()
        self.outputLayerNum.pack()

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

        


