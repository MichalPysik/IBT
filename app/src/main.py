# Project: Classification with Use of Neural Networks in the Keras Environment
# Application: Experimental application for neural network comparison with use of Keras
# Author: Michal Pyšík
# File: main.py

import tkinter as tk
import utils as ut
import view


# Main function that runs the main loop of the application
if __name__ == "__main__":
    root = tk.Tk()
    view = view.View(root)
    view.setup()

    root.option_add("*Dialog.msg.width", 34)
    root.option_add("*Dialog.msg.wrapLength", "6i")
    root.geometry("%sx%s" % (ut.window_width, ut.window_height))
    root.title("Neural network comparison with Keras")
    root.mainloop()
