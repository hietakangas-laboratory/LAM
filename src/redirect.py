# -*- coding: utf-8 -*-
"""
LAM-module for output window when using executibles.

Created on Thu Jan 16 10:37:42 2020

@author: Arto I. Viitanen
"""
import tkinter as tk
import sys

# LAM module
from src.settings import Settings as Sett


class TextWindow:
    """Container for printing output of exec."""

    index = 1.0

    def __init__(self, master, check_val):
        self.txtwin = tk.Toplevel(master)
        self.txtwin.title("Output")
        self.txtwin.protocol("WM_DELETE_WINDOW", self.func_destroy)
        self.upframe = tk.Frame(self.txtwin, bd=3, relief='groove', bg='lightgray')
        self.upframe.grid(row=0, column=0, rowspan=1, columnspan=7, sticky="nwe")
        self.dnframe = tk.Frame(self.txtwin, bd=3, relief='sunken')
        self.dnframe.grid(row=1, column=0, rowspan=10, columnspan=7, sticky="nwe")
        self.rframe = tk.Frame(self.txtwin, bd=3, relief='raised')
        self.rframe.grid(row=1, column=8, rowspan=10, sticky="nes")
        # Exit button
        self.exit_b = tk.Button(self.upframe, text="Close", font=('Arial', 9, 'bold'), command=self.func_destroy)
        self.exit_b.configure(height=1, width=5, fg="red")
        self.exit_b.grid(row=0, column=5, sticky="nwe")
        # Text area
        self.text_area = tk.Text(self.dnframe, height=25, width=75)
        # self.text_area.grid(row=1, column=0, columnspan=10)
        self.text_area.pack(side='left', fill='both', expand=True)
        self.scroll = tk.Scrollbar(self.dnframe)
        self.text_area.config(yscrollcommand=self.scroll.set)
        self.scroll.config(command=self.text_area.yview)
        self.scroll.pack(side='right', fill='y')
        self.orig_out = sys.stdout
        self.orig_err = sys.stderr
        # Redirect stderr and stdout to the same window:
        sys.stdout = StdoutRedirector(self.text_area)
        sys.stderr = StdoutRedirector(self.text_area)
        self.recheck = check_val

    def func_destroy(self, event=None):
        """Destroy window without saving."""
        sys.stdout = self.orig_out
        sys.stderr = self.orig_err
        self.txtwin.destroy()
        Sett.non_stdout = False
        self.recheck.set(False)


class IORedirector:
    """Redirect I/O to GUI widget."""

    def __init__(self, text_area):
        self.text_area = text_area


class StdoutRedirector(IORedirector):
    """Redirect text to IORedirector."""

    def write(self, string):
        """Write to stdout."""
        self.text_area.insert(TextWindow.index, string)
        self.text_area.update_idletasks()
        TextWindow.index += 1.0
        self.text_area.see("end")
