import tkinter as tk
import utils as ut
import view


class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    view = view.View(root)
    view.setup()

    root.option_add("*Dialog.msg.width", 34)
    root.option_add("*Dialog.msg.wrapLength", "6i")
    root.geometry("%sx%s" % (ut.window_width, ut.window_height))
    root.title("Neural network comparison with Keras")
    root.mainloop()
