import tkinter as tk
import sys
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


if __name__ == '__main__':
    root = tk.Tk()
    view = view.View(root)
    view.setup()

    #textbox = tk.Text(root, width=60)
    #sys.stdout = TextRedirector(textbox)
    #root.geometry('900x800')
    #root.configure(bg='#606060')
    root.geometry("%sx%s" % (ut.window_width, ut.window_height))
    root.title('Neural Network Keras')
    root.mainloop()




