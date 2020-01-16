# -*- coding: utf-8 -*-
"""
LAM-module for the creation of graphical user interface.

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""
# LAM modules
from run import MAIN_catch_exit
from settings import settings as Sett
import sys
# Standard libraries
import copy
import tkinter as tk
from tkinter import filedialog
# Other packages
import numpy as np
import pathlib as pl


class base_GUI(tk.Toplevel):
    """Container for the most important settings of the GUI."""

    def __init__(self, master=None):
        master.title("LAM-v1.0")
        self.master = master
        self.master.grab_set()
        self.master.bind('<Escape>', self.func_destroy)
        self.master.bind('<Return>', self.RUN_button)

        # create all of the main containers
        self.topf = tk.Frame(self.master, pady=3)
        self.midf = tk.Frame(self.master, pady=3)
        self.Up_leftf = tk.Frame(self.master, bd=2, relief='groove')
        self.rightf = tk.Frame(self.master, bd=2, relief='groove')
        self.distf = tk.Frame(self.master, bd=2, relief='groove')
        self.bottomf = tk.Frame(self.master)

        # LAYOUT:
        self.topf.grid(row=0, rowspan=2, columnspan=6, pady=(1, 0),
                       sticky="new")
        self.midf.grid(row=2, rowspan=3, columnspan=6, pady=(0, 0),
                       sticky="new")
        self.Up_leftf.grid(row=5, column=0, columnspan=3, rowspan=4,
                           pady=(0, 0), sticky="new")
        self.rightf.grid(row=5, column=3, columnspan=3, rowspan=11,
                         pady=(0, 0), sticky="new")
        self.distf.grid(row=13, rowspan=8, columnspan=6, sticky="new",
                        pady=(0, 0))
        self.bottomf.grid(row=20, rowspan=2, columnspan=6, sticky="sew", pady=(0, 2))
        col_count, row_count = self.master.grid_size()
        for col in range(col_count):
            self.master.grid_columnconfigure(col, minsize=45)
        for row in range(row_count):
            self.master.grid_rowconfigure(row, minsize=32)

        # TOP FRAME / WORK DIRECTORY
        self.folder_path = tk.StringVar()
        self.folder_path.set(Sett.workdir)
        self.lbl1 = tk.Label(self.topf, text=self.folder_path.get(),
                             bg='white', textvariable=self.folder_path, bd=2,
                             relief='sunken')
        self.lbl1.grid(row=0, column=1, columnspan=7)
        self.browse = tk.Button(self.topf, text="Directory",
                                command=self.browse_button)
        self.browse.grid(row=0, column=0)
        self.DetChans = tk.StringVar(value="Detected channels:")
        self.DetGroups = tk.StringVar(value="Detected groups:")
        self.detect = tk.Button(self.topf, text="Detect",
                                command=self.Detect_Channels)
        self.detect.grid(row=1, column=0)
        self.lblGroups = tk.Label(self.topf, text=self.DetGroups.get(),
                                  textvariable=self.DetGroups)
        self.lblGroups.grid(row=1, column=1, columnspan=8, pady=(0, 0))
        self.lblChannels = tk.Label(self.topf, text=self.DetChans.get(),
                                    textvariable=self.DetChans)
        self.lblChannels.grid(row=2, column=1, columnspan=8, pady=(0, 0))

        # MIDDLE FRAME / PRIMARY SETTINGS BOX
        global SampleV, CountV, DistV, PlotV, StatsV, MPV, setMP, setHead
        SampleV = tk.BooleanVar(value=Sett.process_samples)
        CountV = tk.BooleanVar(value=Sett.process_counts)
        DistV = tk.BooleanVar(value=Sett.process_dists)
        PlotV = tk.BooleanVar(value=Sett.Create_Plots)
        StatsV = tk.BooleanVar(value=Sett.statistics)
        MPV = tk.BooleanVar(value=Sett.useMP)
        self.pSample = tk.Checkbutton(self.midf, text="Process",
                                      variable=SampleV, relief='groove', bd=4,
                                      font=('Arial', 8, 'bold'),
                                      command=self.Process_check,
                                      bg='lightgrey')
        self.pCounts = tk.Checkbutton(self.midf, text="Count  ",
                                      variable=CountV, relief='groove', bd=4,
                                      font=('Arial', 8, 'bold'),
                                      bg='lightgrey', command=self.Count_check)
        self.pDists = tk.Checkbutton(self.midf, text="Distance",
                                     variable=DistV, relief='groove', bd=4,
                                     font=('Arial', 8, 'bold'),
                                     command=self.Distance_check,
                                     bg='lightgrey')
        self.pPlots = tk.Checkbutton(self.midf, text="Plots   ",
                                     variable=PlotV, relief='groove', bd=4,
                                     font=('Arial', 8, 'bold'),
                                     command=self.Plot_check, bg='lightgrey')
        self.pStats = tk.Checkbutton(self.midf, text="Stats   ",
                                     variable=StatsV, relief='groove', bd=4,
                                     font=('Arial', 8, 'bold'),
                                     command=self.Stat_check, bg='lightgrey')
        self.pSample.grid(row=0, column=0, columnspan=1, padx=(2, 2))
        self.pCounts.grid(row=0, column=1, columnspan=1, padx=(2, 2))
        self.pDists.grid(row=0, column=2, columnspan=1, padx=(2, 2))
        self.pPlots.grid(row=0, column=3, columnspan=1, padx=(2, 2))
        self.pStats.grid(row=0, column=4, columnspan=1, padx=(2, 2))
        # Measurement point & file header settings
        self.pMP = tk.Checkbutton(self.midf, text="Use MP ", variable=MPV,
                                  relief='groove', bd=4, font=('Arial', 8),
                                  command=self.MP_check)
        self.pMP.grid(row=1, column=0, columnspan=2, padx=(2, 2))
        self.lblMP = tk.Label(self.midf, text='MP label:', bd=1,
                              font=('Arial', 8))
        self.lblMP.grid(row=1, column=2)
        setMP = tk.StringVar(value=Sett.MPname)
        self.MPIn = tk.Entry(self.midf, text=setMP.get(), bg='white',
                             textvariable=setMP, bd=2, relief='sunken')
        self.MPIn.grid(row=1, column=3, columnspan=3)
        lbltext = 'Data file header row:\n(Starts from zero)'
        self.lblHead = tk.Label(self.midf, text=lbltext, bd=1,
                                font=('Arial', 8))
        self.lblHead.grid(row=2, column=0, columnspan=3)
        setHead = tk.IntVar(value=Sett.header_row)
        self.HeadIn = tk.Entry(self.midf, text=setHead.get(), bg='white',
                               textvariable=setHead, bd=2, relief='sunken')
        self.HeadIn.grid(row=2, column=2, columnspan=3)

        # BOTTOM BUTTONS
        self.r_stdout = tk.BooleanVar(value=Sett.non_stdout)
        self.r_stdoutC = tk.Checkbutton(self.bottomf, text="Redirect stdout",
                                        variable=self.r_stdout, 
                                        relief='groove',
                                        bd=1, command=self.redirect_stdout)
        self.r_stdoutC.grid(row=0, column=4, columnspan=4)
        self.Run_b = tk.Button(self.bottomf, text='Run\n<Enter>',
                               font=('Arial', 10, 'bold'),
                               command=self.RUN_button)
        self.Run_b.configure(height=2, width=7, bg='lightgreen',
                             fg="darkgreen")
        self.Run_b.grid(row=1, column=4, columnspan=1, padx=(75, 25),
                        sticky='es')
        self.quitbutton = tk.Button(self.bottomf, text="Quit",
                                    font=('Arial', 9, 'bold'),
                                    command=self.func_destroy)
        self.quitbutton.configure(height=1, width=5, fg="red")
        self.quitbutton.grid(row=1, column=5, sticky='es')

        self.additbutton = tk.Button(self.bottomf, text="Other",
                                     font=('Arial', 9, 'bold'),
                                     command=self.Open_AddSettings)
        self.additbutton.configure(height=2, width=7)
        self.additbutton.grid(row=1, column=0, columnspan=1, padx=(0, 5),
                              sticky='ws')
        self.plotbutton = tk.Button(self.bottomf, text="Plots",
                                    font=('Arial', 9, 'bold'),
                                    command=self.Open_PlotSettings)
        self.plotbutton.configure(height=2, width=7)
        self.plotbutton.grid(row=1, column=1, columnspan=1, padx=(0, 5),
                             sticky='ws')
        self.statsbutton = tk.Button(self.bottomf, text="Stats",
                                     font=('Arial', 9, 'bold'),
                                     command=self.Open_StatSettings)
        self.statsbutton.configure(height=2, width=7)
        self.statsbutton.grid(row=1, column=2, columnspan=1, sticky='ws')
        # self.stdout_win = None
        self.redirect_stdout()

        # RIGHT FRAME / PLOTTING
        # header
        self.lbl2 = tk.Label(self.rightf, text='Plotting:', bd=2,
                             font=('Arial', 9, 'bold'))
        self.lbl2.grid(row=0, column=0)
        # checkbox variables
        global Pchans, Padds, Ppairs, Pheats, Pdists, Pstats, Pclusts
        global PVSchan, PVSadd
        Pchans = tk.BooleanVar(value=Sett.Create_Channel_Plots)
        Padds = tk.BooleanVar(value=Sett.Create_AddData_Plots)
        Ppairs = tk.BooleanVar(value=Sett.Create_Channel_PairPlots)
        Pheats = tk.BooleanVar(value=Sett.Create_Heatmaps)
        Pdists = tk.BooleanVar(value=Sett.Create_Distribution_Plots)
        Pstats = tk.BooleanVar(value=Sett.Create_Statistics_Plots)
        Pclusts = tk.BooleanVar(value=Sett.Create_Cluster_Plots)
        PVSchan = tk.BooleanVar(value=Sett.Create_ChanVSAdd_Plots)
        PVSadd = tk.BooleanVar(value=Sett.Create_AddVSAdd_Plots)
        # create checkboxes
        self.chanC = tk.Checkbutton(self.rightf, text="Channels",
                                    variable=Pchans)
        self.addC = tk.Checkbutton(self.rightf, text="Additional Data",
                                   variable=Padds)
        self.pairC = tk.Checkbutton(self.rightf, text="Channel pair plots",
                                    variable=Ppairs)
        self.heatC = tk.Checkbutton(self.rightf, text="Heat maps",
                                    variable=Pheats)
        self.distC = tk.Checkbutton(self.rightf, text="Distributions",
                                    variable=Pdists)
        self.statC = tk.Checkbutton(self.rightf, text="Statistics",
                                    variable=Pstats)
        self.clustC = tk.Checkbutton(self.rightf, text="Clusters",
                                     variable=Pclusts)
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
        self.clustC.grid(row=7, column=0, sticky='w')
        self.chanVSC.grid(row=8, column=0, sticky='w')
        self.addVSC.grid(row=9, column=0, sticky='w')
        if PlotV.get() is False:
            for child in self.rightf.winfo_children():
                child.configure(state='disable')
        self.Stat_check()
        # LEFT FRAME (UP) / VECTOR CREATION
        global VType, setCh, setBin
        # header
        self.lbl3 = tk.Label(self.Up_leftf, text='Vector:', bd=2,
                             font=('Arial', 9, 'bold'))
        self.lbl3.grid(row=0, column=0)
        # vector type radio buttons
        VType = tk.BooleanVar(value=Sett.SkeletonVector)
        self.Vbut1 = tk.Radiobutton(self.Up_leftf, text="Skeleton",
                                    variable=VType, value=1,
                                    command=self.switch_pages)
        self.Vbut2 = tk.Radiobutton(self.Up_leftf, text="Median",
                                    variable=VType, value=0,
                                    command=self.switch_pages)
        self.Vbut1.grid(row=1, column=0)
        self.Vbut2.grid(row=1, column=1)
        # vector channel input
        self.lbl4 = tk.Label(self.Up_leftf, text='Channel: ', bd=1,
                             font=('Arial', 10))
        self.lbl4.grid(row=2, column=0)
        setCh = tk.StringVar(value=Sett.vectChannel)
        self.chIn = tk.Entry(self.Up_leftf, text=setCh.get(), bg='white',
                             textvariable=setCh, bd=2, relief='sunken')
        self.chIn.grid(row=2, column=1, columnspan=1)
        # Bin number input
        self.lbl5 = tk.Label(self.Up_leftf, text='Bin #: ', bd=1,
                             font=('Arial', 10))
        self.lbl5.grid(row=3, column=0)
        setBin = tk.IntVar(value=len(Sett.projBins))
        self.binIn = tk.Entry(self.Up_leftf, text=setBin.get(), bg='white',
                              textvariable=setBin, bd=2, relief='sunken')
        self.binIn.grid(row=3, column=1, columnspan=1)

        # LEFT FRAME (LOWER) / VECTOR SETTINGS
        self.frames = {}
        for F in (Skel_settings, Median_settings):
            frame = F(self.master, self)
            self.frames[F] = frame
            frame.grid(row=8, column=0, columnspan=3, rowspan=5, sticky="new")
            frame.grid_remove()
        if VType.get():
            self.show_VSett(Skel_settings)
        else:
            self.show_VSett(Median_settings)

        # UPPER BOTTOM / DISTANCES
        global clustV, FdistV
        # header
        self.lbldist = tk.Label(self.distf, text='Distance Calculations:',
                                bd=2, font=('Arial', 9, 'bold'))
        self.lbldist.grid(row=0, column=0, columnspan=6)
        # distance and cluster checkbuttons
        self.clustV = tk.BooleanVar(value=Sett.Find_Clusters)
        self.FdistV = tk.BooleanVar(value=Sett.Find_Distances)
        self.UseSubV = tk.BooleanVar(value=False)
        self.clustC = tk.Checkbutton(self.distf, text="Find clusters ",
                                     variable=self.clustV,
                                     command=self.Cluster_check, bd=1,
                                     relief='raised')
        self.FdistC = tk.Checkbutton(self.distf, text="Find distances",
                                     variable=self.FdistV,
                                     command=self.Dist_check,
                                     bd=1, relief='raised')
        self.USubC = tk.Checkbutton(self.distf, text="Filter Size   ",
                                    variable=self.UseSubV,
                                    command=self.Filter_check,
                                    bd=1, relief='raised')
        self.clustC.grid(row=1, column=0, columnspan=2, sticky='n')
        self.FdistC.grid(row=1, column=2, columnspan=2, sticky='n')
        self.USubC.grid(row=1, column=4, columnspan=2, sticky='n')
        self.Cllbl = tk.Label(self.distf, text="Clusters:")
        self.Cllbl.grid(row=2, column=0, columnspan=2)
        self.Distlbl = tk.Label(self.distf, text="Cell Distances:")
        self.Distlbl.grid(row=2, column=4, columnspan=2)

        # cluster settings
        global setClCh, setClDist, setClMin, setClMax, setClSiz
        self.ClChanlbl = tk.Label(self.distf, text="Channels:")
        self.ClChanlbl.grid(row=3, column=0, columnspan=2)
        setClCh = tk.StringVar(value=','.join(Sett.Cluster_Channels))
        self.ClChIn = tk.Entry(self.distf, text=setClCh.get(), bg='white',
                               textvariable=setClCh, bd=2, relief='sunken')
        self.ClChIn.grid(row=3, column=2, columnspan=2)

        self.ClDistlbl = tk.Label(self.distf, text='Max Dist.:')
        self.ClDistlbl.grid(row=4, column=0, columnspan=2)
        setClDist = tk.DoubleVar(value=Sett.Cl_maxDist)
        self.ClDistIn = tk.Entry(self.distf, text=setClDist.get(), bg='white',
                                 textvariable=setClDist, bd=2, relief='sunken')
        self.ClDistIn.grid(row=4, column=2, columnspan=2)

        self.ClMinlbl = tk.Label(self.distf, text='Min cell #:')
        self.ClMinlbl.grid(row=5, column=0, columnspan=2)
        setClMin = tk.IntVar(value=Sett.Cl_min)
        self.ClMinIn = tk.Entry(self.distf, text=setClMin.get(), bg='white',
                                textvariable=setClMin, bd=2, relief='sunken')
        self.ClMinIn.grid(row=5, column=2, columnspan=2)

        self.ClMaxlbl = tk.Label(self.distf, text='Max cell #:')
        self.ClMaxlbl.grid(row=6, column=0, columnspan=2)
        setClMax = tk.IntVar(value=Sett.Cl_max)
        self.ClMaxIn = tk.Entry(self.distf, text=setClMax.get(), bg='white',
                                textvariable=setClMax, bd=2, relief='sunken')
        self.ClMaxIn.grid(row=6, column=2, columnspan=2)

        self.ClSizlbl = tk.Label(self.distf, text='Size incl.:')
        self.ClSizlbl.grid(row=7, column=0, columnspan=2)
        setClSiz = tk.DoubleVar(value=Sett.Cl_Vol_inclusion)
        self.ClSizIn = tk.Entry(self.distf, text=setClSiz.get(), bg='white',
                                textvariable=setClSiz, bd=2, relief='sunken')
        self.ClSizIn.grid(row=7, column=2, columnspan=2)

        self.Vselect = tk.BooleanVar()
        self.VClbut1 = tk.Radiobutton(self.distf, text="Greater", value=1,
                                      variable=self.Vselect)
        self.VClbut2 = tk.Radiobutton(self.distf, text="Smaller", value=0,
                                      variable=self.Vselect)
        self.VClbut1.grid(row=8, column=0)
        self.VClbut2.grid(row=8, column=2)

        # distances
        global setDCh, setDDist, setDTarget, setDSiz
        self.DChanlbl = tk.Label(self.distf, text="Channels:")
        self.DChanlbl.grid(row=3, column=4, columnspan=2)
        setDCh = tk.StringVar(value=','.join(Sett.Distance_Channels))
        self.DChIn = tk.Entry(self.distf, text=setDCh.get(), bg='white',
                              textvariable=setDCh, bd=2, relief='sunken')
        self.DChIn.grid(row=3, column=6, columnspan=2)

        self.DDistlbl = tk.Label(self.distf, text='Max Dist.:')
        self.DDistlbl.grid(row=4, column=4, columnspan=2)
        setDDist = tk.DoubleVar(value=Sett.maxDist)
        self.DDistIn = tk.Entry(self.distf, text=setDDist.get(), bg='white',
                                textvariable=setDDist, bd=2, relief='sunken')
        self.DDistIn.grid(row=4, column=6, columnspan=2)

        self.UseTargetV = tk.BooleanVar(value=Sett.use_target)
        self.DTarget = tk.Checkbutton(self.distf, text='Use target:',
                                      variable=self.UseTargetV,
                                      command=self.Target_check)
        self.DTarget.grid(row=5, column=4, columnspan=2)
        setDTarget = tk.StringVar(value=Sett.target_chan)
        self.DTargetIn = tk.Entry(self.distf, text=setDTarget.get(),
                                  bg='white', textvariable=setDTarget, bd=2,
                                  relief='sunken')
        self.DTargetIn.grid(row=5, column=6, columnspan=2)

        self.DSizlbl = tk.Label(self.distf, text='Size incl.:')
        self.DSizlbl.grid(row=7, column=4, columnspan=2)
        setDSiz = tk.DoubleVar(value=Sett.Vol_inclusion)
        self.DSizIn = tk.Entry(self.distf, text=setDSiz.get(), bg='white',
                               textvariable=setDSiz, bd=2, relief='sunken')
        self.DSizIn.grid(row=7, column=6, columnspan=2)

        self.VDselect = tk.BooleanVar()
        self.VDbut1 = tk.Radiobutton(self.distf, text="Greater", value=1,
                                     variable=self.VDselect)
        self.VDbut2 = tk.Radiobutton(self.distf, text="Smaller", value=0,
                                     variable=self.VDselect)
        self.VDbut1.grid(row=8, column=5)
        self.VDbut2.grid(row=8, column=7)
        # Disable / enable widgets
        self.Process_check()
        self.Count_check()
        self.Distance_check()

    def Distance_check(self):
        """Relevant changes when distances-setting is checked."""
        if not DistV.get():
            for widget in self.distf.winfo_children():
                widget.configure(state='disable')
        else:
            for widget in self.distf.winfo_children():
                if int(widget.grid_info()["row"]) in [0, 1, 2]:
                    widget.configure(state='normal')
            self.Cluster_check()
            self.Dist_check()
            self.Filter_check()
        self.run_check()

    def Cluster_check(self):
        """Relevant changes when cluster-setting is checked."""
        if not self.clustV.get():
            for widget in [self.ClChanlbl, self.ClChIn, self.ClDistlbl,
                           self.ClDistIn, self.ClMinlbl, self.ClMinIn,
                           self.ClMaxlbl, self.ClMaxIn, self.ClSizlbl,
                           self.ClSizIn, self.VClbut1, self.VClbut2,
                           self.Cllbl]:
                widget.configure(state='disable')
        else:
            for widget in [self.ClChanlbl, self.ClChIn, self.ClDistlbl,
                           self.ClDistIn, self.ClMinlbl, self.ClMinIn,
                           self.ClMaxlbl, self.ClMaxIn, self.ClSizlbl,
                           self.ClSizIn, self.VClbut1, self.VClbut2,
                           self.Cllbl]:
                widget.configure(state='normal')
            self.Filter_check()

    def Dist_check(self):
        """Relevant changes when find distance-setting is checked."""
        if not self.FdistV.get():
            for widget in [self.DChanlbl, self.DChIn, self.DDistlbl,
                           self.DDistIn, self.DTarget, self.DTargetIn,
                           self.DSizlbl, self.DSizIn, self.VDbut1, self.VDbut2,
                           self.Distlbl]:
                widget.configure(state='disable')
        else:
            for widget in [self.DChanlbl, self.DChIn, self.DDistlbl,
                           self.DDistIn, self.DTarget, self.DTargetIn,
                           self.DSizlbl, self.DSizIn, self.VDbut1, self.VDbut2,
                           self.Distlbl]:
                widget.configure(state='normal')
            self.Target_check()
            self.Filter_check()

    def Filter_check(self):
        """Relevant changes when filtering by size is checked."""
        if not self.UseSubV.get():
            for widget in [self.DSizlbl, self.DSizIn, self.VDbut1, self.VDbut2,
                           self.ClSizlbl, self.ClSizIn, self.VClbut1,
                           self.VClbut2]:
                widget.configure(state='disable')
        else:
            if self.FdistV.get():
                for widget in [self.DSizlbl, self.DSizIn, self.VDbut1,
                               self.VDbut2]:
                    widget.configure(state='normal')
            if self.clustV.get():
                for widget in [self.ClSizlbl, self.ClSizIn, self.VClbut1,
                               self.VClbut2]:
                    widget.configure(state='normal')

    def MP_check(self):
        """Relevant changes when MP is in use or not."""
        if not MPV.get():
            self.lblMP.configure(state='disable')
            self.MPIn.configure(state='disable')
        else:
            self.lblMP.configure(state='normal')
            self.MPIn.configure(state='normal')

    def Target_check(self):
        """Relevant changes when target-setting is checked."""
        if not self.UseTargetV.get():
            self.DTargetIn.configure(state='disable')
        else:
            self.DTargetIn.configure(state='normal')

    def Stat_check(self):
        """Relevant changes when statistics-setting is checked."""
        if not StatsV.get():
            self.statsbutton.configure(state='disable')
            self.statC.configure(state='disable')
        else:
            self.statsbutton.configure(state='normal')
            if PlotV.get():
                self.statC.configure(state='normal')
        self.run_check()

    def Plot_check(self):
        """Relevant changes when plot-setting is checked."""
        if PlotV.get() is False:
            self.plotbutton.configure(state='disable')
            for widget in self.rightf.winfo_children():
                widget.configure(state='disable')
        else:
            self.plotbutton.configure(state='normal')
            for widget in self.rightf.winfo_children():
                widget.configure(state='normal')
        self.run_check()

    def Process_check(self):
        """Relevant changes when Process-setting is checked."""
        if not SampleV.get():
            for widget in self.Up_leftf.winfo_children():
                if widget not in [self.binIn, self.lbl5]:
                    widget.configure(state='disable')
            hidev = 'disable'
        else:
            for widget in self.Up_leftf.winfo_children():
                if widget not in [self.binIn, self.lbl5]:
                    widget.configure(state='normal')
            self.switch_pages()
            hidev = 'normal'
        if not VType.get():
            for widget in self.frames[Median_settings].winfo_children():
                widget.configure(state=hidev)
        else:
            for widget in self.frames[Skel_settings].winfo_children():
                widget.configure(state=hidev)
        self.run_check()

    def run_check(self):
        """Determine whether run button is active."""
        Ps = [SampleV.get(), CountV.get(), DistV.get(), PlotV.get(),
              StatsV.get()]
        if not any(Ps):
            self.Run_b.configure(state='disable', bg='lightgrey')
        else:
            self.Run_b.configure(state='normal', bg='lightgreen')

    def Count_check(self):
        """Relevant changes when count-setting is checked."""
        if not CountV.get():
            self.binIn.configure(state='disable')
            self.lbl5.configure(state='disable')
            self.lblMP.configure(state='disable')
            self.MPIn.configure(state='disable')
            self.pMP.configure(state='disable')
        else:
            self.binIn.configure(state='normal')
            self.lbl5.configure(state='normal')
            self.pMP.configure(state='normal')
            self.lblMP.configure(state='normal')
            self.MPIn.configure(state='normal')
            self.lblHead.configure(state='normal')
            self.HeadIn.configure(state='normal')
        self.run_check()

    def browse_button(self):
        """Allow input of path when browse-button is pressed."""
        filename = filedialog.askdirectory()
        self.folder_path.set(filename)
        Sett.workdir = str(self.folder_path.get())
        self.Detect_Channels()

    def RUN_button(self, event=None):
        """Relevant changes when Run-button is pressed + run initialization."""
        Sett.workdir = pl.Path(self.folder_path.get())
        Sett.process_samples = SampleV.get()
        Sett.process_counts = CountV.get()
        Sett.process_dists = DistV.get()
        Sett.Create_Plots = PlotV.get()
        Sett.statistics = StatsV.get()
        if not Sett.process_counts:
            Sett.useMP = False
        else:
            Sett.useMP = MPV.get()
        Sett.MPname = setMP.get()
        Sett.header_row = setHead.get()
        Sett.Create_Channel_Plots = Pchans.get()
        Sett.Create_AddData_Plots = Padds.get()
        Sett.Create_Channel_PairPlots = Ppairs.get()
        Sett.Create_Heatmaps = Pheats.get()
        Sett.Create_Distribution_Plots = Pdists.get()
        Sett.Create_Statistics_Plots = Pstats.get()
        Sett.Create_Cluster_Plots = Pclusts.get()
        Sett.Create_ChanVSAdd_Plots = PVSchan.get()
        Sett.Create_AddVSAdd_Plots = PVSadd.get()
        Sett.vectChannel = setCh.get()
        Sett.projBins = np.linspace(0, 1, setBin.get())
        if not VType.get():
            Sett.SkeletonVector = False
            Sett.simplifyTol = SimpTol.get()
            Sett.medianBins = medBins.get()
        else:
            Sett.SkeletonVector = True
            Sett.simplifyTol = SimpTol.get()
            Sett.SkeletonResize = reSz.get()
            Sett.find_dist = fDist.get()
            Sett.BDiter = dilI.get()
            Sett.SigmaGauss = SSmooth.get()
        # Distance calculations
        Sett.Find_Distances = self.FdistV.get()
        Sett.Find_Clusters = self.clustV.get()
        if Sett.Find_Distances:
            ChStr = setDCh.get().split(',')
            for i, channel in enumerate(ChStr):
                ChStr[i] = channel.strip()
            Sett.Distance_Channels = ChStr
            Sett.maxDist = setDDist.get()
            Sett.use_target = self.UseTargetV.get()
            if Sett.use_target:
                ChStr = setDTarget.get().split(',')
                for i, channel in enumerate(ChStr):
                    ChStr[i] = channel.strip()
                if len(ChStr) > 1:
                    print("WARNING: 'Use Target' accepts only one channel.")
                    print("Using '{}' as target".format(ChStr[0]))
                    flag = 1
                Sett.target_chan = ChStr[0]
            if self.FdistV.get():
                Sett.Vol_inclusion = setDSiz.get()
                if self.VDselect.get():
                    Sett.incl_type = "greater"
                else:
                    Sett.incl_type = ""
        else:
            Sett.Vol_inclusion = 0
        if Sett.Find_Clusters:
            ChStr2 = setClCh.get().split(',')
            for i, channel in enumerate(ChStr2):
                ChStr2[i] = channel.strip()
            Sett.Cluster_Channels = ChStr2
            Sett.Cl_maxDist = setClDist.get()
            Sett.Cl_min = setClMin.get()
            Sett.Cl_max = setClMax.get()
            if self.UseSubV.get():
                Sett.Cl_Vol_inclusion = setClSiz.get()
                if self.Vselect.get():
                    Sett.Cl_incl_type = "greater"
                else:
                    Sett.Cl_incl_type = ""
        else:
            Sett.Cl_Vol_inclusion = 0
        import logger as lg
        import logging
        if lg.log_created is True:
            # Close old loggers and create new:
            lg.Close()
            lg.Update()
            LAM_logger = logging.getLogger(__name__)
        else:
            LAM_logger = lg.setup_logger(__name__)
        lg.print_settings(LAM_logger)
        lg.logprint(LAM_logger, 'Run parameters set', 'i')
        lg.logprint(LAM_logger, 'Begin run', 'i')
        if 'flag' in locals():
            msg = "'Use Target' accepts only one channel. Using '{}'".format(
                                                                    ChStr[0])
            lg.logprint(LAM_logger, msg, 'w')
        MAIN_catch_exit()

    def redirect_stdout(self):
        import redirect as rd
        Sett.non_stdout = self.r_stdout.get()
        if Sett.non_stdout:
            self.stdout_win = rd.text_window(self.master, self.r_stdout)
        else:
            self.stdout_win.func_destroy()

    def show_VSett(self, name):
        """Change shown vector settings based on type."""
        for frame in self.frames.values():
            frame.grid_remove()
        frame = self.frames[name]
        frame.grid()

    def switch_pages(self):
        """Switch page of vector settings."""
        if not VType.get():
            self.show_VSett(Median_settings)
        else:
            self.show_VSett(Skel_settings)

    def func_destroy(self, event=None):
        """Destroy GUI."""
        import logger as lg
        lg.log_Shutdown()
        self.stdout_win.func_destroy()
        self.master.destroy()

    def Open_AddSettings(self):
        """Open additional settings window."""
        Additional_data(self.master)

    def Open_PlotSettings(self):
        """Open plot settings window."""
        Plot_Settings(self.master)

    def Open_StatSettings(self):
        """Open statistics settings window."""
        Stat_Settings(self.master)

    def Detect_Channels(self):
        """Detect channels and groups found at current set path."""
        workdir = pl.Path(self.folder_path.get())
        global DetChans, DetGroups
        DetChans = []
        DetGroups = []
        # Loop found sample directories
        for samplepath in [p for p in workdir.iterdir() if p.is_dir() and
                           'Analysis Data' not in p.name]:
            try:  # Get groups from folder names
                group = str(samplepath.name).split('_')[0]
                if group not in DetGroups:
                    DetGroups.append(group)
            except IndexError:
                pass
            # Loop through channels of found samples
            for channelpath in [p for p in samplepath.iterdir() if p.is_dir()]:
                try:
                    channel = str(channelpath.name).split('_')[-2]
                    if channel not in DetChans:
                        DetChans.append(channel)
                except IndexError:
                    pass
        # Change text variables to contain new groups and channels
        if DetChans:
            chanstring = tk.StringVar(value="Detected channels: {}".format(
                                            ', '.join(sorted(DetChans))))
        else:
            chanstring = tk.StringVar(value='No detected channels!')
        if DetGroups:
            grpstring = tk.StringVar(value="Detected groups: {}".format(
                                            ', '.join(sorted(DetGroups))))
        else:
            grpstring = tk.StringVar(value='No detected groups!')
        # Set new text variables to be shown
        self.DetGroups.set(grpstring.get())
        self.DetChans.set(chanstring.get())
        from settings import store
        store.samplegroups = DetGroups
        store.channels = DetChans


class Skel_settings(tk.Frame):
    """Container for skeleton vector-related settings."""

    def __init__(self, parent, master):
        tk.Frame.__init__(self, parent, bd=2, relief='groove')
        # Container label
        self.lblSetS = tk.Label(self, text='Vector Parameters:', bd=1,
                                font=('Arial', 10))
        self.lblSetS.grid(row=0, column=0, columnspan=3, pady=(0, 3))

        # Container widgets
        global SimpTol, reSz, fDist, dilI, SSmooth
        self.lbl6 = tk.Label(self, text='Simplify tol.', bd=1,
                             font=('Arial', 9))
        self.lbl6.grid(row=1, column=0, columnspan=1)
        SimpTol = tk.DoubleVar(value=Sett.simplifyTol)
        self.simpIn = tk.Entry(self, text=SimpTol.get(), bg='white',
                               textvariable=SimpTol, bd=2, relief='sunken')
        self.simpIn.grid(row=1, column=1)

        self.lbl7 = tk.Label(self, text='Resize', bd=1,
                             font=('Arial', 9))
        self.lbl7.grid(row=2, column=0, columnspan=1)
        reSz = tk.DoubleVar(value=Sett.SkeletonResize)
        self.rszIn = tk.Entry(self, text=reSz.get(), bg='white',
                              textvariable=reSz, bd=2, relief='sunken')
        self.rszIn.grid(row=2, column=1)

        self.lbl8 = tk.Label(self, text='Find distance', bd=1,
                             font=('Arial', 9))
        self.lbl8.grid(row=3, column=0, columnspan=1)
        fDist = tk.DoubleVar(value=Sett.find_dist)
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
        self.lbl10.grid(row=5, column=0, columnspan=1, pady=(0, 22))
        SSmooth = tk.DoubleVar(value=Sett.SigmaGauss)
        self.smoothIn = tk.Entry(self, text=SSmooth.get(), bg='white',
                                 textvariable=SSmooth, bd=2, relief='sunken')
        self.smoothIn.grid(row=5, column=1, pady=(0, 22))


class Median_settings(tk.Frame):
    """Container for median vector-related settings."""

    def __init__(self, parent, master):
        # Container label
        tk.Frame.__init__(self, parent, bd=2, relief='groove')

        self.lblSetM = tk.Label(self, text='Vector Parameters:', bd=1,
                                font=('Arial', 10))
        self.lblSetM.grid(row=0, column=0, columnspan=3, pady=(0, 3))

        # Container widgets
        global SimpTol, medBins
        self.lbl6 = tk.Label(self, text='Simplify tol.', bd=1,
                             font=('Arial', 9))
        self.lbl6.grid(row=1, column=0, columnspan=1)
        SimpTol = tk.DoubleVar(value=Sett.simplifyTol)
        self.simpIn = tk.Entry(self, text=SimpTol.get(), bg='white',
                               textvariable=SimpTol, bd=2, relief='sunken')
        self.simpIn.grid(row=1, column=1)

        self.lbl7 = tk.Label(self, text='Median bins  ', bd=1,
                             font=('Arial', 9))
        self.lbl7.grid(row=2, column=0, columnspan=1, pady=(0, 85))
        medBins = tk.IntVar(value=Sett.medianBins)
        self.mbinIn = tk.Entry(self, text=medBins.get(), bg='white',
                               textvariable=medBins, bd=2, relief='sunken')
        self.mbinIn.grid(row=2, column=1, pady=(0, 85))


class Additional_data(tk.Toplevel):
    """Container for Other-window settings."""

    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.grab_set()
        self.window.title("Additional Data Settings")
        self.window.bind('<Escape>', self.window.destroy)
        self.window.bind('<Return>', self.save_setts)
        self.frame = tk.Frame(self.window)
        self.frame.grid(row=0, rowspan=11, columnspan=9, sticky="new")
        self.Dframe = tk.Frame(self.window, relief='groove')
        self.Dframe.grid(row=8, rowspan=5, columnspan=9, sticky="new")
        self.Bframe = tk.Frame(self.window)
        self.Bframe.grid(row=13, rowspan=2, columnspan=9, sticky="ne")
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
        self.lblIn.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        self.lbl2 = tk.Label(self.frame, text='csv-file', bd=1,
                             font=('Arial', 9))
        self.lbl2.grid(row=0, column=3, columnspan=2)
        setcsv = tk.StringVar(value="Area.csv")
        self.fileIn = tk.Entry(self.frame, text=setcsv.get(), bg='white',
                               textvariable=setcsv, bd=2, relief='sunken')
        self.fileIn.grid(row=1, column=3, columnspan=2, pady=(0, 10))

        self.lbl3 = tk.Label(self.frame, text='Unit', bd=1,
                             font=('Arial', 9))
        self.lbl3.grid(row=0, column=5, columnspan=2)
        setUnit = tk.StringVar(value="um^2")
        self.unitIn = tk.Entry(self.frame, text=setUnit.get(), bg='white',
                               textvariable=setUnit, bd=2, relief='sunken')
        self.unitIn.grid(row=1, column=5, columnspan=2, pady=(0, 10))

        # buttons
        self.Add_b = tk.Button(self.frame, text='Add',
                               font=('Arial', 10, 'bold'),
                               command=self.add_data)
        self.Add_b.configure(bg='lightgreen', fg="darkgreen")
        self.Add_b.grid(row=0, column=7, rowspan=2, padx=(5, 10), pady=(0, 10))
        self.Save_b = tk.Button(self.Bframe, text='Save & Return\n<Enter>',
                                font=('Arial', 10, 'bold'),
                                command=self.save_setts)
        self.Save_b.configure(height=2, width=12, bg='lightgreen',
                              fg="darkgreen")
        self.Save_b.grid(row=11, column=5, rowspan=2, columnspan=2,
                         padx=(0, 10))
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
            self.buttons.append(tk.Button(self.frame, text='x',
                                          font=('Arial', 10), relief='raised',
                                          command=lambda i=i:
                                              self.rmv_data(i)))
            self.buttons[i].grid(row=row, column=7, sticky='w')
        # additional data ID-replacement
        self.repID = tk.BooleanVar(value=Sett.replaceID)
        self.repIDC = tk.Checkbutton(self.Dframe, text="Replace file ID",
                                     variable=self.repID,  relief='groove',
                                     bd=4, command=self.replace_check)
        self.repIDC.grid(row=0, column=0, columnspan=4)
        self.chanlbl = tk.Label(self.Dframe, text='File descriptor:', bd=1)
        self.chanlbl.grid(row=1, column=0, columnspan=3)
        self.changelbl = tk.Label(self.Dframe, text='Change to:', bd=1)
        self.changelbl.grid(row=1, column=3, columnspan=3)
        for i, key in enumerate(Sett.channelID.keys()):
            changevalues = Sett.channelID.get(key)
            fileID = tk.StringVar(value=key)
            ChanID = tk.StringVar(value=changevalues)
            self.fIDIn = tk.Entry(self.Dframe, text=fileID.get(), bg='white',
                                  textvariable=fileID, bd=2, relief='sunken')
            self.cIDIn = tk.Entry(self.Dframe, text=ChanID.get(), bg='white',
                                  textvariable=ChanID, bd=2, relief='sunken')
            row = i+2
            self.fIDIn.grid(row=row, column=0, columnspan=3)
            self.cIDIn.grid(row=row, column=3, columnspan=3)
        self.replace_check()

    def replace_check(self):
        """Change relevant settings when replaceID is checked."""
        if not self.repID.get():
            for child in self.Dframe.winfo_children():
                if isinstance(child, tk.Entry):
                    child.configure(state='disable')
        else:
            for child in self.Dframe.winfo_children():
                if isinstance(child, tk.Entry):
                    child.configure(state='normal')

    def add_data(self):
        """Addition of data input to the additional data table."""
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
            self.buttons.append(tk.Button(self.frame, text='x',
                                          font=('Arial', 10), relief='raised',
                                          command=lambda i=i:
                                              self.rmv_data(i)))
            self.buttons[i].grid(row=row, column=7, sticky='w')
            self.addDict.update({setLbl.get(): [setcsv.get(), setUnit.get()]})
            self.rowN = self.rowN + 1
        else:
            print("WARNING: Attempted overwrite of additional data label!")
            print("Delete old label of same name before adding.")

    def rmv_data(self, i):
        """Remove data from the additional data table."""
        for widget in self.frame.grid_slaves():
            if int(widget.grid_info()["row"]) == i+2 and int(
                    widget.grid_info()["column"]) == 0:
                key = widget.cget("text")
                if key in self.addDict.keys():
                    self.addDict.pop(key, None)
                    widget.grid_forget()
                else:
                    print("WARNING: removed label not found in add. data.")
            elif int(widget.grid_info()["row"]) == i+2:
                widget.grid_forget()

    def save_setts(self, event=None):
        """Save settings when exiting Other-window."""
        Sett.AddData = self.addDict
        Sett.replaceID = self.repID.get()
        if Sett.replaceID:
            fileIDs = []
            changeIDs = []
            for child in self.Dframe.winfo_children():
                if isinstance(child, tk.Entry)and int(child.grid_info()[
                                                            "column"]) == 0:
                    fileIDs.append(child.get())
                if isinstance(child, tk.Entry)and int(child.grid_info()[
                                                            "column"]) == 3:
                    changeIDs.append(child.get())
            Sett.channelID = {}
            for i, key in enumerate(fileIDs):
                Sett.channelID.update({key: changeIDs[i]})
        self.window.destroy()


class Plot_Settings(tk.Toplevel):
    """Container for Other-window settings."""

    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.grab_set()
        self.window.title("Plot Settings")
        self.window.bind('<Escape>', self.func_destroy)
        self.window.bind('<Return>', self.save_setts)
        self.frame = tk.Frame(self.window, bd=3, relief='groove')
        self.frame.grid(row=0, column=0, rowspan=9, columnspan=4, sticky="nw")
        self.statframe = tk.Frame(self.window, bd=2, relief='groove')
        self.statframe.grid(row=9, column=0, rowspan=4, columnspan=4,
                            sticky="n")
        self.Bframe = tk.Frame(self.window)
        self.Bframe.grid(row=13, column=0, rowspan=2, columnspan=4, pady=5,
                         sticky="ne")
        # General settings:
        # Outliers
        self.lbl1 = tk.Label(self.frame, text="General Settings:",
                             font=('Arial', 10, 'bold'))
        self.lbl1.grid(row=0, column=0, columnspan=2)
        self.DropV = tk.BooleanVar(value=Sett.Drop_Outliers)
        self.DropC = tk.Checkbutton(self.frame, text="Drop outliers",
                                    variable=self.DropV,  relief='groove',
                                    bd=1, command=self.Drop_check)
        self.DropC.grid(row=1, column=0, columnspan=2)
        self.lbl2 = tk.Label(self.frame, text='Std dev.:')
        self.lbl2.grid(row=2, column=0, columnspan=2)
        self.setSTD = tk.DoubleVar(value=Sett.dropSTD)
        self.stdIn = tk.Entry(self.frame, text=self.setSTD.get(), bg='white',
                              textvariable=self.setSTD, bd=2, relief='sunken')
        self.stdIn.grid(row=2, column=2, columnspan=2)
        self.Drop_check()
        # Pairplot jitter
        self.JitterV = tk.BooleanVar(value=Sett.Drop_Outliers)
        self.JitterC = tk.Checkbutton(self.frame, text="Pair plot jitter",
                                      variable=self.JitterV)
        self.JitterC.grid(row=3, column=0, columnspan=2)
        # Save format
        self.lbl3 = tk.Label(self.frame, text='Save format:')
        self.lbl3.grid(row=4, column=0, columnspan=1)
        self.setSF = tk.StringVar(value=Sett.saveformat)
        self.SFIn = tk.Entry(self.frame, text=self.setSF.get(), bg='white',
                             textvariable=self.setSF, bd=2, relief='sunken')
        self.SFIn.grid(row=4, column=1, columnspan=2)
        comment = "Supported formats:\n\
        eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba,\nsvg, svgz, tif, tiff"
        self.lbl4 = tk.Label(self.frame, text=comment, fg='dimgrey')
        self.lbl4.grid(row=5, column=0, columnspan=4, pady=(0, 10))
        # versus
        self.lbl8 = tk.Label(self.frame, text='Versus plots:')
        self.lbl8.grid(row=6, column=0, columnspan=3)

        self.lbl9 = tk.Label(self.frame, text='Plotted channels:')
        self.lbl9.grid(row=7, column=0, columnspan=2)
        self.setVsCh = tk.StringVar(value=','.join(Sett.vs_channels))
        self.VsChIn = tk.Entry(self.frame, text=self.setVsCh.get(), bg='white',
                               textvariable=self.setVsCh, bd=2,
                               relief='sunken')
        self.VsChIn.grid(row=7, column=2, columnspan=2)

        self.lbl8 = tk.Label(self.frame, text='Plotted add. data:')
        self.lbl8.grid(row=8, column=0, columnspan=2)
        self.setVsAdds = tk.StringVar(value=','.join(Sett.vs_adds))
        self.VsAddIn = tk.Entry(self.frame, text=self.setVsAdds.get(),
                                bg='white', textvariable=self.setVsAdds, bd=2,
                                relief='sunken')
        self.VsAddIn.grid(row=8, column=2, columnspan=2)

        # Statistics settings:
        self.lbl5 = tk.Label(self.statframe, text='Statistical Plotting:')
        self.lbl5.grid(row=0, column=0, columnspan=3)

        self.starsV = tk.BooleanVar(value=Sett.Drop_Outliers)
        self.starC = tk.Checkbutton(self.statframe, text="Sign. stars",
                                    variable=self.starsV,
                                    command=self.sign_check)
        self.starC.grid(row=1, column=0, columnspan=2)

        self.CfillV = tk.BooleanVar(value=Sett.fill)
        self.CfillC = tk.Checkbutton(self.statframe, text="Sign. color",
                                     variable=self.CfillV)
        self.CfillC.grid(row=1, column=2, columnspan=2)

        self.neglogV = tk.BooleanVar(value=Sett.negLog2)
        self.neglogC = tk.Checkbutton(self.statframe, text="Neg. log2",
                                      variable=self.neglogV,
                                      command=self.sign_check)
        self.neglogC.grid(row=2, column=0, columnspan=1)

        self.lbl7 = tk.Label(self.statframe, text='y-limit:')
        self.lbl7.grid(row=2, column=1, columnspan=1)
        self.setLogy = tk.DoubleVar(value=Sett.ylim)
        self.ylimIn = tk.Entry(self.statframe, text=self.setLogy.get(),
                               bg='white', textvariable=self.setLogy, bd=2,
                               relief='sunken')
        self.ylimIn.grid(row=2, column=2, columnspan=2)
        self.sign_check()
        self.CObs = tk.BooleanVar(value=Sett.observations)
        self.CobsC = tk.Checkbutton(self.statframe, text="Observations",
                                    variable=self.CObs)
        self.CobsC.grid(row=3, column=0, columnspan=2)
        # Buttons
        self.Save_b = tk.Button(self.Bframe, text='Save & Return\n<Enter>',
                                font=('Arial', 10, 'bold'),
                                command=self.save_setts)
        self.Save_b.configure(height=2, width=12, bg='lightgreen',
                              fg="darkgreen")
        self.Save_b.grid(row=0, column=2, rowspan=2, columnspan=2,
                         padx=(0, 10))
        self.exit_b = tk.Button(self.Bframe, text="Return",
                                font=('Arial', 9, 'bold'),
                                command=self.func_destroy)
        self.exit_b.configure(height=1, width=5, fg="red")
        self.exit_b.grid(row=0, column=4)

    def Drop_check(self):
        """Relevant changes when dropping of outliers is selected."""
        if not self.DropV.get():
            self.stdIn.configure(state='disable')
        else:
            self.stdIn.configure(state='normal')

    def sign_check(self):
        """Relevant changes when neglog2-setting is selected."""
        if not self.neglogV.get():
            self.starC.configure(state='normal')
            self.lbl7.configure(state='disable')
            self.ylimIn.configure(state='disable')
        else:
            self.starsV.set(False)
            self.starC.configure(state='disable')
            self.lbl7.configure(state='normal')
            self.ylimIn.configure(state='normal')

    def save_setts(self, event=None):
        """Save settings when Plot-window is exited."""
        Sett.Drop_Outliers = self.DropV.get()
        Sett.dropSTD = self.setSTD.get()
        Sett.plot_jitter = self.JitterV.get()
        Sett.saveformat = self.setSF.get()
        Sett.negLog2 = self.neglogV.get()
        Sett.stars = self.starsV.get()
        Sett.fill = self.CfillV.get()
        Sett.ylim = self.setLogy.get()
        Sett.observations = self.CObs.get()
        ChStr = self.setVsCh.get().split(',')
        for i, channel in enumerate(ChStr):
            ChStr[i] = channel.strip()
        Sett.vs_channels = ChStr
        ChStr = self.setVsAdds.get().split(',')
        for i, channel in enumerate(ChStr):
            ChStr[i] = channel.strip()
        Sett.vs_adds = ChStr
        self.window.destroy()

    def func_destroy(self, event=None):
        """Destroy window without saving."""
        self.window.destroy()


class Stat_Settings(tk.Toplevel):
    """Container for statistics-window settings."""

    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.grab_set()
        self.window.title("Statistics Settings")
        self.window.bind('<Escape>', self.func_destroy)
        self.window.bind('<Return>', self.save_setts)
        self.frame = tk.Frame(self.window)
        self.frame.grid(row=0, rowspan=4, columnspan=4, sticky="new")
        self.Winframe = tk.Frame(self.window, bd=1, relief='groove')
        self.Winframe.grid(row=5, rowspan=2, columnspan=4, sticky="n")
        self.But_frame = tk.Frame(self.window)
        self.But_frame.grid(row=7, rowspan=2, columnspan=4, sticky="ne")
        # Control group
        self.lbl1 = tk.Label(self.frame, text='Control Group:')
        self.lbl1.grid(row=0, column=0, columnspan=2, pady=4)
        self.setCtrlGrp = tk.StringVar(value=Sett.cntrlGroup)
        self.CtrlIn = tk.Entry(self.frame, text=self.setCtrlGrp.get(),
                               bg='white', textvariable=self.setCtrlGrp, bd=2,
                               relief='sunken')
        self.CtrlIn.grid(row=0, column=2, columnspan=2, pady=4)
        # Statistic types
        self.TotalV = tk.BooleanVar(value=Sett.stat_total)
        self.VersusV = tk.BooleanVar(value=Sett.stat_versus)
        self.StatT = tk.Checkbutton(self.frame, text="Total Statistics",
                                    variable=self.TotalV, relief='groove',
                                    bd=1)
        self.StatV = tk.Checkbutton(self.frame, text="Group vs Group",
                                    variable=self.VersusV, relief='groove',
                                    bd=1)
        self.StatT.grid(row=1, column=0, columnspan=2, pady=4)
        self.StatV.grid(row=1, column=2, columnspan=2, pady=4)
        # Alpha
        self.lbl2 = tk.Label(self.frame, text='Alpha:')
        self.lbl2.grid(row=2, column=0, columnspan=2, pady=4)
        self.setAlpha = tk.DoubleVar(value=Sett.alpha)
        self.AlphaIn = tk.Entry(self.frame, text=self.setAlpha.get(),
                                bg='white', textvariable=self.setAlpha, bd=2,
                                relief='sunken')
        self.AlphaIn.grid(row=2, column=2, columnspan=2, pady=4)
        # windowed
        self.WindV = tk.BooleanVar(value=Sett.windowed)
        self.WindC = tk.Checkbutton(self.frame, text="Windowed statistics",
                                    variable=self.WindV, relief='raised', bd=1,
                                    command=self.Window_check)
        self.WindC.grid(row=3, column=0, columnspan=3, pady=(10, 0))
        # windowed options
        self.lbl2 = tk.Label(self.Winframe, text='Trailing window:')
        self.lbl2.grid(row=0, column=0, columnspan=2, pady=1)
        self.setTrail = tk.IntVar(value=Sett.trail)
        self.TrailIn = tk.Entry(self.Winframe, text=self.setTrail.get(),
                                bg='white', textvariable=self.setTrail, bd=2,
                                relief='sunken')
        self.TrailIn.grid(row=0, column=2, columnspan=2, pady=1)

        self.lbl3 = tk.Label(self.Winframe, text='Leading window:')
        self.lbl3.grid(row=1, column=0, columnspan=2, pady=(1, 5))
        self.setLead = tk.IntVar(value=Sett.lead)
        self.LeadIn = tk.Entry(self.Winframe, text=self.setLead.get(),
                               bg='white', textvariable=self.setLead, bd=2,
                               relief='sunken')
        self.LeadIn.grid(row=1, column=2, columnspan=2, pady=(1, 5))
        self.Window_check()
        # Buttons
        self.Save_b = tk.Button(self.But_frame, text='Save & Return\n<Enter>',
                                font=('Arial', 10, 'bold'),
                                command=self.save_setts)
        self.Save_b.configure(height=2, width=12, bg='lightgreen',
                              fg="darkgreen")
        self.Save_b.grid(row=0, column=2, rowspan=2, columnspan=2,
                         padx=(0, 10))
        self.exit_b = tk.Button(self.But_frame, text="Return",
                                font=('Arial', 9, 'bold'),
                                command=self.func_destroy)
        self.exit_b.configure(height=1, width=5, fg="red")
        self.exit_b.grid(row=0, column=4)

    def Window_check(self):
        """Relevant changes when windowed statistics is selected."""
        if not self.WindV.get():
            for widget in self.Winframe.winfo_children():
                widget.configure(state='disable')
        else:
            for widget in self.Winframe.winfo_children():
                widget.configure(state='normal')

    def save_setts(self, event=None):
        """Save settings when exiting stats-window."""
        Sett.cntrlGroup = self.setCtrlGrp.get()
        Sett.stat_total = self.TotalV.get()
        Sett.stat_versus = self.VersusV.get()
        Sett.alpha = self.setAlpha.get()
        Sett.windowed = self.WindV.get()
        if Sett.windowed:
            Sett.trail = self.setTrail.get()
            Sett.lead = self.setLead.get()
        self.window.destroy()

    def func_destroy(self, event=None):
        """Destroy stats-window."""
        self.window.destroy()
