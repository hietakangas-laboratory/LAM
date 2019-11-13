# -*- coding: utf-8 -*-
from run import main
import tkinter as tk
from tkinter import filedialog
from settings import settings as Sett
import numpy as np
import copy

class base_GUI(tk.Toplevel):   
    def __init__(self, master=None):
        master.title("LAM-v1.0")
        self.master = master
        self.master.bind('<Escape>', self.func_destroy)
        self.master.bind('<Return>', self.RUN_button)
        
        ## create all of the main containers
        self.topf = tk.Frame(self.master, pady=3)
        self.midf = tk.Frame(self.master, pady=3)
        self.Up_leftf = tk.Frame(self.master, bd=2, relief='groove')
        self.rightf = tk.Frame(self.master, bd=2, relief='groove')
        self.distf = tk.Frame(self.master, bd=2, relief='groove')
        self.bottomf = tk.Frame(self.master, pady=10)
        
        ## LAYOUT:
        self.topf.grid(row=0, columnspan=6, sticky="new")
        self.midf.grid(row=1, columnspan=6, sticky="new")
        self.Up_leftf.grid(row=2, column=0, columnspan=3, rowspan=4, pady=(3,0), 
                           sticky="new")
        self.rightf.grid(row=2, column=3, columnspan=3, rowspan=10, pady=(3,0), 
                         sticky="new")
        self.distf.grid(row=9, columnspan=6, sticky="new", pady=(3,5))
        self.bottomf.grid(row=10, columnspan=6, sticky="sew")
        col_count, row_count = self.master.grid_size()
        for col in range(col_count):
            self.master.grid_columnconfigure(col, minsize=45)        
        for row in range(row_count):
            self.master.grid_rowconfigure(row, minsize=32)
            
        ## TOP FRAME / WORK DIRECTORY
        self.folder_path = tk.StringVar()
        self.folder_path.set(Sett.workdir)
        self.lbl1 = tk.Label(self.topf, text=self.folder_path.get(), bg='white', 
                             textvariable=self.folder_path, bd=2, relief='sunken')
        self.lbl1.grid(row=0, column=1, columnspan=7)
        self.browse = tk.Button(self.topf, text="Directory", command=self.browse_button)
        self.browse.grid(row=0, column=0)
        
        ## MIDDLE FRAME / PRIMARY SETTINGS BOX
        global SampleV, CountV, DistV, PlotV
        SampleV, CountV = tk.BooleanVar(), tk.BooleanVar()
        DistV, PlotV = tk.BooleanVar(), tk.BooleanVar()
        SampleV.set(Sett.process_samples), CountV.set(Sett.process_counts) 
        DistV.set(Sett.process_dists), PlotV.set(Sett.Create_Plots)
        self.pSample = tk.Checkbutton(self.midf, text="Process ",variable=SampleV, 
                                      relief='groove', bd=4, font=('Arial', 8, 'bold'),  
                                      command=self.Process_check)
        self.pCounts = tk.Checkbutton(self.midf, text="Counts  ", variable=CountV, 
                                      relief='groove', bd=4, font=('Arial', 8, 'bold'))
        self.pDists = tk.Checkbutton(self.midf, text="Distance", variable=DistV, 
                                     relief='groove', bd=4, font=('Arial', 8, 'bold'))
        self.pPlots = tk.Checkbutton(self.midf, text="Plots    ", variable=PlotV, 
                                     relief='groove', bd=4, font=('Arial', 8, 'bold'), 
                                     command=self.Plot_check)
        self.pSample.grid(row=2, column=0, columnspan=2, padx=(2,5))
        self.pCounts.grid(row=2, column=2, columnspan=2, padx=(5,5))
        self.pDists.grid(row=2, column=4, columnspan=2, padx=(5,5))
        self.pPlots.grid(row=2, column=6, columnspan=2, padx=(5,0))
        
        ## RUN, QUIT & ADDITIONAL BUTTONS
        self.Run_b = tk.Button(self.bottomf, text='Run\n<Enter>', 
                               font=('Arial', 10, 'bold'), 
                               command=self.RUN_button)
        self.Run_b.configure(height=2, width=7, bg='lightgreen', fg="darkgreen")
        self.Run_b.grid(row=0, column=3, columnspan=2, padx=(0,10))
        self.quitbutton = tk.Button(self.bottomf, text="Quit", 
                                    font=('Arial', 9, 'bold'), 
                                    command=master.destroy)
        self.quitbutton.configure(height=1, width=5, fg="red")
        self.quitbutton.grid(row=0, column=8)
        self.additbutton = tk.Button(self.bottomf, text="Add. data ...", 
                                    font=('Arial', 9), 
                                    command=self.Open_AddSettings)
        self.additbutton.configure(height=2, width=10)
        self.additbutton.grid(row=0, column=1, columnspan=2, padx=(15,100), 
                             sticky="w")
        
        ## RIGHT FRAME / PLOTTING
        # header
        self.lbl2 = tk.Label(self.rightf, text='Plotting:', bd=2)
        self.lbl2.grid(row=0, column=0)
        # checkbox variables
        global Pchans, Padds, Ppairs, Pheats, Pdists, Pstats, PVSchan, PVSadd
        Pchans = tk.BooleanVar(value=Sett.Create_Channel_Plots)
        Padds = tk.BooleanVar(value=Sett.Create_AddData_Plots)
        Ppairs = tk.BooleanVar(value=Sett.Create_Channel_PairPlots)
        Pheats = tk.BooleanVar(value=Sett.Create_Heatmaps)
        Pdists = tk.BooleanVar(value=Sett.Create_Distribution_Plots)
        Pstats = tk.BooleanVar(value=Sett.Create_Statistics_Plots)
        PVSchan = tk.BooleanVar(value=Sett.Create_ChanVSAdd_Plots)
        PVSadd = tk.BooleanVar(value=Sett.Create_AddVSAdd_Plots)
        # create checkboxes
        self.chanC = tk.Checkbutton(self.rightf, text="Channels",
                                    variable=Pchans)
        self.addC = tk.Checkbutton(self.rightf, text="Additional Data", 
                                   variable=Padds)
        self.pairC = tk.Checkbutton(self.rightf, text="Channel pairplots", 
                                    variable=Ppairs)
        self.heatC = tk.Checkbutton(self.rightf, text="Heatmaps", 
                                    variable=Pheats)
        self.distC = tk.Checkbutton(self.rightf, text="Distributions",
                                    variable=Pdists)
        self.statC = tk.Checkbutton(self.rightf, text="Statistics", 
                                    variable=Pstats)
        self.chanVSC = tk.Checkbutton(self.rightf, text="Channel VS. Add.", 
                                      variable=PVSchan)
        self.addVSC = tk.Checkbutton(self.rightf, text="Add. VS. Add.", 
                                     variable=PVSadd)
        self.chanC.grid(row=1, column=0, sticky='w')
        self.addC.grid(row=2, column=0, sticky='w')
        self.pairC.grid(row=3, column=0, sticky='w')
        self.heatC.grid(row=4, column=0, sticky='w')
        self.distC.grid(row=5, column=0, sticky='w')
        self.statC.grid(row=6, column=0, sticky='w')
        self.chanVSC.grid(row=7, column=0, sticky='w', pady=(7,0))
        self.addVSC.grid(row=8, column=0, sticky='w', pady=(0,0))
        if PlotV.get() == False:
            for child in self.rightf.winfo_children():
                child.configure(state = 'disable')
        
        ## LEFT FRAME (UP) / VECTOR CREATION
        global VType, setCh, setBin
        # header
        self.lbl3 = tk.Label(self.Up_leftf, text='Vector:', bd=2, font=('Arial', 10))
        self.lbl3.grid(row=0, column=0)
        # vector type radio buttons
        VType = tk.BooleanVar(value=Sett.SkeletonVector)
        self.Vbut1 = tk.Radiobutton(self.Up_leftf, text="Skeleton", variable=VType, 
                                    value=1, command=self.switch_pages)
        self.Vbut2 = tk.Radiobutton(self.Up_leftf, text="Median", variable=VType, 
                                    value=0, command=self.switch_pages)
        self.Vbut1.grid(row=1, column=0)
        self.Vbut2.grid(row=1, column=1)
        # vector channel input
        self.lbl4 = tk.Label(self.Up_leftf, text='Channel: ', bd=1, font=('Arial', 10))
        self.lbl4.grid(row=2, column=0)
        setCh = tk.StringVar(value=Sett.vectChannel)
        self.chIn = tk.Entry(self.Up_leftf, text=setCh.get(), bg='white', 
                             textvariable=setCh, bd=2, relief='sunken')
        self.chIn.grid(row=2, column=1, columnspan=1)
        # Bin number input
        self.lbl5 = tk.Label(self.Up_leftf, text='Bin #: ', bd=1, font=('Arial', 10))
        self.lbl5.grid(row=3, column=0)
        setBin = tk.IntVar(value=len(Sett.projBins))
        self.binIn = tk.Entry(self.Up_leftf, text=setBin.get(), bg='white', 
                             textvariable=setBin, bd=2, relief='sunken')
        self.binIn.grid(row=3, column=1, columnspan=1)
        
        ## LEFT FRAME (LOWER) / VECTOR SETTINGS    
        self.frames = {}
        for F in (Skel_settings, Median_settings):
            frame = F(self.master, self)
            self.frames[F] = frame
            frame.grid(row=5, column=0, columnspan=3, rowspan=5, sticky="new")
            frame.grid_remove()
        if VType.get():
            self.show_VSett(Skel_settings)
        else:
            self.show_VSett(Median_settings)
        self.Process_check()
        
        # UPPER BOTTOM / DISTANCES
        global clustV, FdistV
        # header
        self.lbldist = tk.Label(self.distf, text='Distance Calculations:', bd=2, font=('Arial', 10))
        self.lbldist.grid(row=0, column=0, columnspan=6)
        # distance and cluster checkbuttons
        clustV = tk.BooleanVar(value=Sett.Find_Clusters)
        FdistV = tk.BooleanVar(value=Sett.Find_Distances)
        self.clustC = tk.Checkbutton(self.distf, text="Find clusters",
                                    variable=clustV)
        self.FdistC = tk.Checkbutton(self.distf, text="Find distances", 
                                   variable=FdistV)
        self.clustC.grid(row=1, column=0, columnspan=3, sticky='nw')
        self.FdistC.grid(row=1, column=3, columnspan=3, sticky='ne')
        # cluster settings
        
        
    def Plot_check(self):
        if PlotV.get() == False:
            for child in self.rightf.winfo_children():
                child.configure(state='disable')
        else:
            for child in self.rightf.winfo_children():
                child.configure(state='normal')
                
    def Process_check(self):
        if SampleV.get() == False:
            for child in self.Up_leftf.winfo_children():
                child.configure(state = 'disable')
            hidev = 'disable'
        else:
            for child in self.Up_leftf.winfo_children():
                child.configure(state = 'normal')
                self.switch_pages()
            hidev = 'normal'
                
        if not VType.get():
            for child in self.frames[Median_settings].winfo_children():
                child.configure(state = hidev)
        else:
            for child in self.frames[Skel_settings].winfo_children():
                child.configure(state = hidev)
        
    def browse_button(self):
        filename = filedialog.askdirectory()
        self.folder_path.set(filename)
        Sett.workdir = str(self.folder_path.get())
        
    def RUN_button(self, event=None):
        Sett.process_samples = SampleV.get()
        Sett.process_counts = CountV.get()
        Sett.process_dists = DistV.get()
        Sett.Create_Plots = PlotV.get()
        Sett.Create_Channel_Plots = Pchans.get()
        Sett.Create_AddData_Plots = Padds.get()
        Sett.Create_Channel_PairPlots = Ppairs.get()
        Sett.Create_Heatmaps = Pheats.get()
        Sett.Create_Distribution_Plots = Pdists.get()
        Sett.Create_Statistics_Plots = Pstats.get()
        Sett.Create_ChanVSAdd_Plots = PVSchan.get()
        Sett.Create_AddVSAdd_Plots = PVSadd.get()
        Sett.vectChannel = setCh.get()        
        Sett.projBins = np.linspace(0, 1, setBin.get())
        if not VType.get():
            Sett.simplifyTol = SimpTol.get()
            Sett.medianBins = medBins.get()
        else:
            Sett.simplifyTol = SimpTol.get()
            Sett.SkeletonResize = reSz.get()
            Sett.find_dist = fDist.get()
            Sett.BDiter = dilI.get()
            Sett.SigmaGauss = SSmooth.get()
        
        main()
        
    def show_VSett(self, name):
        for frame in self.frames.values():
            frame.grid_remove()
        frame = self.frames[name]
        frame.grid()
        
    def switch_pages(self):
        if not VType.get():
            self.show_VSett(Median_settings)
        else:
            self.show_VSett(Skel_settings)
    
    def func_destroy(self, event=None):
        self.master.destroy()
        
    def Open_AddSettings(self):
        Additional_data(self.master)
        
class Skel_settings(tk.Frame):
    def __init__(self, parent, master):
        tk.Frame.__init__(self,parent, bd=2, relief='groove')
        self.lblSetS = tk.Label(self, text='Vector Parameters:', bd=1, 
                                font=('Arial', 10))
        self.lblSetS.grid(row=0, column=0, columnspan=3, pady=(0,10))
        
        global SimpTol, reSz, fDist, dilI, SSmooth
        self.lbl6 = tk.Label(self, text='Simplify tol.', bd=1, 
                             font=('Arial', 9))
        self.lbl6.grid(row=1, column=0, columnspan=1)
        SimpTol = tk.IntVar(value=Sett.simplifyTol)
        self.simpIn = tk.Entry(self, text=SimpTol.get(), bg='white', 
                             textvariable=SimpTol, bd=2, relief='sunken')
        self.simpIn.grid(row=1, column=1)
        
        self.lbl7 = tk.Label(self, text='Resize', bd=1, 
                             font=('Arial', 9))
        self.lbl7.grid(row=2, column=0, columnspan=1)
        reSz = tk.IntVar(value=Sett.SkeletonResize)
        self.rszIn = tk.Entry(self, text=reSz.get(), bg='white', 
                             textvariable=reSz, bd=2, relief='sunken')
        self.rszIn.grid(row=2, column=1)
        
        self.lbl8 = tk.Label(self, text='Find distance', bd=1, 
                             font=('Arial', 9))
        self.lbl8.grid(row=3, column=0, columnspan=1)
        fDist = tk.IntVar(value=Sett.find_dist)
        self.distIn = tk.Entry(self, text=fDist.get(), bg='white', 
                             textvariable=fDist, bd=2, relief='sunken')
        self.distIn.grid(row=3, column=1)
        
        self.lbl9 = tk.Label(self, text='Dilation iter', bd=1, 
                             font=('Arial', 9))
        self.lbl9.grid(row=4, column=0, columnspan=1)
        dilI = tk.IntVar(value=Sett.BDiter)
        self.dilIn = tk.Entry(self, text=dilI.get(), bg='white', 
                             textvariable=dilI, bd=2, relief='sunken')
        self.dilIn.grid(row=4, column=1)
        
        self.lbl10 = tk.Label(self, text='Smoothing', bd=1, 
                              font=('Arial', 9))
        self.lbl10.grid(row=5, column=0, columnspan=1)
        SSmooth = tk.IntVar(value=Sett.SigmaGauss)
        self.smoothIn = tk.Entry(self, text=SSmooth.get(), bg='white', 
                             textvariable=SSmooth, bd=2, relief='sunken')
        self.smoothIn.grid(row=5, column=1)
            
class Median_settings(tk.Frame):
    def __init__(self, parent, master):
        tk.Frame.__init__(self,parent, bd=2, relief='groove')
        
        self.lblSetM = tk.Label(self, text='Vector Parameters:', bd=1, 
                                font=('Arial', 10))
        self.lblSetM.grid(row=0, column=0, columnspan=3, pady=(0,3))
        
        global SimpTol, medBins
        self.lbl6 = tk.Label(self, text='Simplify tol.', bd=1, 
                             font=('Arial', 9))
        self.lbl6.grid(row=1, column=0, columnspan=1)
        SimpTol = tk.IntVar(value=Sett.simplifyTol)
        self.simpIn = tk.Entry(self, text=SimpTol.get(), bg='white', 
                             textvariable=SimpTol, bd=2, relief='sunken')
        self.simpIn.grid(row=1, column=1)
        
        self.lbl7 = tk.Label(self, text='Median bins', bd=1, 
                             font=('Arial', 9))
        self.lbl7.grid(row=2, column=0, columnspan=1, pady=(0,64))
        medBins = tk.IntVar(value=Sett.medianBins)
        self.mbinIn = tk.Entry(self, text=medBins.get(), bg='white', 
                             textvariable=medBins, bd=2, relief='sunken')
        self.mbinIn.grid(row=2, column=1, pady=(0,64))
        
class Additional_data(tk.Toplevel):
    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.grab_set()
        self.window.title("Additional Data Settings")
        self.window.bind('<Escape>', self.window.destroy)
        self.window.bind('<Return>', self.save_setts)
        self.frame = tk.Frame(self.window)
        self.frame.grid(row=0, rowspan=11, columnspan=9, sticky="new")
        self.Bframe = tk.Frame(self.window)
        self.Bframe.grid(row=11, rowspan=2, columnspan=9, sticky="ne")
        col_count, row_count = self.window.grid_size()
        for col in range(col_count):
            self.window.grid_columnconfigure(col, minsize=45)        
        for row in range(row_count):
            self.window.grid_rowconfigure(row, minsize=32)
        # Adding data descriptors:
        global setLbl, setcsv, setUnit
        self.lbl1 = tk.Label(self.frame, text='Column label', bd=1, 
                             font=('Arial', 9))
        self.lbl1.grid(row=0, column=0, columnspan=2)
        setLbl = tk.StringVar(value="Area")
        self.lblIn = tk.Entry(self.frame, text=setLbl.get(), bg='white', 
                             textvariable=setLbl, bd=2, relief='sunken')
        self.lblIn.grid(row=1, column=0, columnspan=2,pady=(0,10))
        
        self.lbl2 = tk.Label(self.frame, text='csv-file', bd=1, 
                             font=('Arial', 9))
        self.lbl2.grid(row=0, column=3, columnspan=2)
        setcsv = tk.StringVar(value="Area.csv")
        self.fileIn = tk.Entry(self.frame, text=setcsv.get(), bg='white', 
                             textvariable=setcsv, bd=2, relief='sunken')
        self.fileIn.grid(row=1, column=3, columnspan=2,pady=(0,10))
        
        self.lbl3 = tk.Label(self.frame, text='Unit', bd=1, 
                             font=('Arial', 9))
        self.lbl3.grid(row=0, column=5, columnspan=2)
        setUnit = tk.StringVar(value="um^2")
        self.unitIn = tk.Entry(self.frame, text=setUnit.get(), bg='white', 
                             textvariable=setUnit, bd=2, relief='sunken')
        self.unitIn.grid(row=1, column=5, columnspan=2,pady=(0,10)) 
        # buttons
        self.Add_b = tk.Button(self.frame, text='Add', 
                               font=('Arial', 10, 'bold'), 
                               command=self.add_data)
        self.Add_b.configure(bg='lightgreen',fg="darkgreen")
        self.Add_b.grid(row=0, column=7, rowspan=2, padx=(5,10),pady=(0,10))
        self.Save_b = tk.Button(self.Bframe, text='Save & Return\n<Enter>', 
                               font=('Arial', 10, 'bold'), 
                               command=self.save_setts)
        self.Save_b.configure(height=2, width=12, bg='lightgreen',fg="darkgreen")
        self.Save_b.grid(row=11, column=5, rowspan=2, columnspan=2, padx=(0,10))
        self.exit_b = tk.Button(self.Bframe, text="Return", 
                                    font=('Arial', 9, 'bold'), 
                                    command=self.window.destroy)
        self.exit_b.configure(height=1, width=5, fg="red")
        self.exit_b.grid(row=11, column=7)
        
        # additional data labels and removal buttons
        addkeys = sorted(list(Sett.AddData.keys()))
        self.addDict = copy.deepcopy(Sett.AddData)
        self.rowN = len(addkeys)
        self.buttons = []
        for i, key in enumerate(addkeys):
            row = i+2
            datalist = self.addDict.get(key)
            name = tk.StringVar(value=key)
            file = tk.StringVar(value=datalist[0])
            unit = tk.StringVar(value=datalist[1])
            l1 = tk.Label(self.frame, text=name.get(), bd=2, bg='lightgrey', 
                          relief='groove')
            l1.grid(row=row, column=0, columnspan=2)
            l2 = tk.Label(self.frame, text=file.get(), bd=2, bg='lightgrey', 
                          relief='groove')
            l2.grid(row=row, column=3, columnspan=2)
            l3 = tk.Label(self.frame, text=unit.get(), bd=2, bg='lightgrey', 
                          relief='groove')
            l3.grid(row=row, column=5, columnspan=2)
            self.buttons.append(tk.Button(self.frame, text='x', font=('Arial',10), 
                              relief='raised', command=lambda i=i: self.rmv_data(i)))
            self.buttons[i].grid(row=row, column=7, sticky='w')        
        
    def add_data(self):
        if setLbl.get() not in self.addDict.keys():
            i = self.rowN
            row = self.rowN+2
            l1 = tk.Label(self.frame, text=setLbl.get(), bd=2, bg='lightgrey', 
                          relief='groove')
            l1.grid(row=row, column=0, columnspan=2)
            l2 = tk.Label(self.frame, text=setcsv.get(), bd=2, bg='lightgrey', 
                          relief='groove')
            l2.grid(row=row, column=3, columnspan=2)
            l3 = tk.Label(self.frame, text=setUnit.get(), bd=2, bg='lightgrey', 
                          relief='groove')
            l3.grid(row=row, column=5, columnspan=2)
            self.buttons.append(tk.Button(self.frame, text='x', font=('Arial',10), 
                              relief='raised', command=lambda i=i: self.rmv_data(i)))
            self.buttons[i].grid(row=row, column=7, sticky='w')        
            self.addDict.update({setLbl.get(): [setcsv.get(), setUnit.get()]})
            self.rowN = self.rowN +1
        else:
            print("USER-WARNING: Attempted overwrite of additional data label!")
            print("Delete old label of same name before adding.")
        
    def rmv_data(self, i):
        for widget in self.frame.grid_slaves():
            if int(widget.grid_info()["row"]) == i+2 and int(widget.grid_info()[
                                                                "column"]) == 0:
                key = widget.cget("text")
                if key in self.addDict.keys():
                    self.addDict.pop(key, None)
#                    self.rowN = self.rowN -1
                    widget.grid_forget()
                else:
                    print("USER-WARNING: removed label not found in add. data.")
            elif int(widget.grid_info()["row"]) == i+2:
                widget.grid_forget()
            
    def save_setts(self):
        Sett.AddData = self.addDict
        self.window.destroy()