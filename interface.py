# -*- coding: utf-8 -*-
from multiprocessing import Process
from run import main
import tkinter as tk
import sys
from tkinter import filedialog
from settings import settings

class base_GUI(tk.Toplevel):   
    def __init__(self, master=None, width=325, height=500):
#        master.geometry("{}x{}".format(width, height))
        master.title("LAM-v1.0")
#        self.frame = tk.Frame(master)
        # create all of the main containers
        self.topf = tk.Frame(master, pady=3)
        self.midf = tk.Frame(master, pady=3)
        self.leftf = tk.Frame(master, pady=3)
        self.rightf = tk.Frame(master, pady=3)
        self.bottomf = tk.Frame(master, pady=3)
        # LAYOUT:
        self.topf.grid(row=0, columnspan=8, sticky="new")
        self.midf.grid(row=1, columnspan=8, sticky="nsew")
        self.leftf.grid(row=3, column=0, columnspan=4, rowspan=8, sticky="ew")
        self.rightf.grid(row=3, column=4, columnspan=4, rowspan=8, sticky="ew")
        self.bottomf.grid(row=11, columnspan=8, sticky="se")
        col_count, row_count = master.grid_size()
        for col in range(col_count):
            master.grid_columnconfigure(col, minsize=20)        
        for row in range(row_count):
            master.grid_rowconfigure(row, minsize=20)
        # TOP FRAME / WORK DIRECTORY
        self.folder_path = tk.StringVar()
        self.folder_path.set(settings.workdir)
        self.lbl1 = tk.Label(self.topf, text=self.folder_path.get(), bg='white', 
                             textvariable=self.folder_path, bd=2, relief='sunken')
        self.lbl1.grid(row=0, column=1, columnspan=7)
        self.browse = tk.Button(self.topf, text="Directory", command=self.browse_button)
        self.browse.grid(row=0, column=0)
        # MIDDLE FRAME / PRIMARY SETTINGS BOX
        global SampleV, CountV, DistV, PlotV
        SampleV, CountV = tk.BooleanVar(), tk.BooleanVar()
        DistV, PlotV = tk.BooleanVar(), tk.BooleanVar()
        SampleV.set(True), CountV.set(True), DistV.set(True), PlotV.set(True)
        self.pSample = tk.Checkbutton(self.midf, text="Process",variable=SampleV, 
                                      relief='groove', padx=3)
        self.pCounts = tk.Checkbutton(self.midf, text="Counts", variable=CountV, 
                                      relief='groove', padx=3)
        self.pDists = tk.Checkbutton(self.midf, text="Distances", variable=DistV, 
                                     relief='groove', padx=3)
        self.pPlots = tk.Checkbutton(self.midf, text="Plots", variable=PlotV, 
                                     relief='groove', padx=8)
        self.pSample.grid(row=2, column=1, columnspan=1)
        self.pCounts.grid(row=2, column=2, columnspan=1)
        self.pDists.grid(row=2, column=3, columnspan=1)
        self.pPlots.grid(row=2, column=4, columnspan=1)
        self.pSample.select(), self.pCounts.select(), self.pDists.select(), self.pPlots.select()        
        # RUN & QUIT BUTTONS
        self.Run_b = tk.Button(self.bottomf, text='Run', font=('Arial', 12), 
                               command=self.RUN_button)
        self.Run_b.configure(height=2, width=7, bg='lightgreen', fg="darkgreen", )
        self.Run_b.grid(row=0, column=3, columnspan=2)
        self.quitbutton = tk.Button(self.bottomf, text="Quit", command=master.destroy)
        self.quitbutton.configure(height=1, width=5, fg="red")
        self.quitbutton.grid(row=0, column=8)
        # RIGHT FRAME / PLOTTING
        self.lbl2 = tk.Label(self.rightf, text='Plotting:', bd=2)
        self.lbl2.grid(row=0, column=0)
        self.PlotC = tk.Checkbutton(self.rightf, text="Process",variable=SampleV, 
                                      relief='groove', padx=3)
        
    def browse_button(self):
        filename = filedialog.askdirectory()
        self.folder_path.set(filename)
        settings.workdir = str(self.folder_path.get())
        print(settings.workdir)
        
    def RUN_button(self):
        settings.process_samples = SampleV.get()
        settings.process_counts = CountV.get()
        settings.process_dists = DistV.get()
        settings.Create_Plots = PlotV.get()
        main()
#        self.Run_b.configure(text='Stop', bg="tomato", fg='darkred', command=self.STOP_button)
#        global mainThread
#        mainThread = Process(target=main)
#        mainThread.start()
#        mainThread.join()
#        sys.stdout.flush()
        

    def STOP_button(self):
        self.Run_b.configure(text='Run', bg='lightgreen', fg="darkgreen", command=self.RUN_button)
        mainThread.terminate()
        
    