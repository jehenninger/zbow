def get_screen_size():
    import sys
    import matplotlib

    if sys.platform is 'darwin':
        matplotlib.use('TkAgg')
    import tkinter as tk

    root = tk.Tk()

    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()

    root.destroy()

    return width, height
