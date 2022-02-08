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
        self.menubar = tk.Menu(self.container)
        self.modelmenu = tk.Menu(self.menubar, tearoff=0)
        self.modelmenu.add_command(label="Load model", command=self.loadModel_callback)
        self.modelmenu.add_command(label="Save model", command=self.saveModel_callback)
        self.modelmenu.add_command(label="Save model as...", command=self.saveModelAs_callback)
        self.menubar.add_cascade(label="Model", menu=self.modelmenu)
        self.menubar.add_command(label="Help", command=self.help_callback)
        self.menubar.add_command(label="Clear screen", command=self.clearScreen_callback)
        self.container.bind('<Control-o>', lambda event:self.loadModel_callback())
        self.container.bind('<Control-O>', lambda event:self.loadModel_callback())
        self.container.bind('<Control-s>', lambda event:self.saveModel_callback())
        self.container.bind('<Control-S>', lambda event:self.saveModel_callback())
        self.container.bind('<Control-Shift-s>', lambda event:self.saveModelAs_callback())
        self.container.bind('<Control-Shift-S>', lambda event:self.saveModelAs_callback())
        self.container.bind('<Control-h>', lambda event:self.help_callback())
        self.container.bind('<Control-H>', lambda event:self.help_callback())
        self.container.bind('<Control-l>', lambda event:self.clearScreen_callback())
        self.container.bind('<Control-L>', lambda event:self.clearScreen_callback())
        self.container.config(menu=self.menubar)

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
        self.epochsLabel = tk.Label(self.epochsBatchFrame, text="Epochs:", anchor="e", width=11)
        self.epochsEntry = tk.Entry(self.epochsBatchFrame, width=5)
        self.batchSizeLabel = tk.Label(self.epochsBatchFrame, text="Batch size:", anchor="e", width=11)
        self.bachSizeEntry = tk.Entry(self.epochsBatchFrame, width=5)
        self.trainModelButton = tk.Button(self.leftFrame, text="Train model", command=self.trainModel_callback)

        self.addLayerButton = tk.Button(self.leftFrame, text="Add hidden layer", command=self.addHiddenLayer)
        self.removeLayerButton = tk.Button(self.leftFrame, text="Remove hidden layer", command=self.removeHiddenLayer)
        self.inputLayerNum = tk.Scale(self.leftFrame, from_=1, to=32, orient=HORIZONTAL, length=200)
        self.hiddenLayersFrame = tk.Frame(self.leftFrame, height=1)
        self.outputLayerNum = tk.Scale(self.leftFrame, from_=1, to=32, orient=HORIZONTAL, length=200)

        self.lossOptimizerFrame = tk.Frame(self.leftFrame)
        self.lossCBLabel = ttk.Label(self.lossOptimizerFrame, text = "Loss:", anchor="e", width=10)
        self.lossComboBox = ttk.Combobox(self.lossOptimizerFrame, state='readonly')
        self.lossComboBox['values'] = ('binary_crossentropy', 'mean_squared_error', 'hinge')
        #self.lossComboBox.bind('<<ComboboxSelected>>', self.lossComboBox_callback)
        self.lossComboBox.current(0)
        #self.lossComboBox_callback()
        self.optimizerCBLabel = ttk.Label(self.lossOptimizerFrame, text = "Optimizer:", anchor="e", width=10)
        self.optimizerComboBox = ttk.Combobox(self.lossOptimizerFrame, state='readonly')
        self.optimizerComboBox['values'] = ('adam', 'sgd', 'rmsprop')
        #self.optimizerComboBox.bind('<<ComboboxSelected>>', self.optimizerComboBox_callback)
        self.optimizerComboBox.current(0)
        #self.optimizerComboBox_callback()

        self.createModelButton = tk.Button(self.leftFrame, text="Create model", command=self.createModel_callback)


    def help_callback(self, event=None):
        print("This is help")


    def clearScreen_callback(self, event=None):
        self.textWindow.config(state='normal')
        self.textWindow.delete('1.0', 'end')
        self.textWindow.config(state='disabled')


    def trainModel_callback(self, event=None):
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


    def createModel_callback(self, event=None):
        neurons = []
        neurons.append(self.inputLayerNum.get())
        for hl in self.hidden_layers:
            neurons.append(hl.get())
        neurons.append(self.outputLayerNum.get())
        self.NN.createModel(2 + len(self.hidden_layers), neurons)


    def loadModel_callback(self, event=None):
        loadPath = tk.filedialog.askopenfilename(filetypes=(("H5 file", "*.h5"),("All Files", "*.*") ))
        try:
            self.NN.loadModel(loadPath)
            self.savePath = loadPath
        except:
            pass


    def saveModel_callback(self, event=None):
        if self.savePath is None:
            self.savePath = tk.filedialog.asksaveasfilename(defaultextension='.h5', filetypes=(("H5 file", "*.h5"),("All Files", "*.*") ))
        self.NN.saveModel(self.savePath)


    def saveModelAs_callback(self, event=None):
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

        self.lossOptimizerFrame.pack(pady=(20,5))
        self.lossCBLabel.grid(row=0, column=0)
        self.lossComboBox.grid(row=0, column=1)
        self.optimizerCBLabel.grid(row=1, column=0)
        self.optimizerComboBox.grid(row=1, column=1)

        self.createModelButton.pack(pady=10)


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

        


