# -*- coding: utf-8 -*-
"""
LAM-module for the creation of graphical user interface.

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""
# Standard libraries
import tkinter as tk
from copy import deepcopy
# Other packages
import pandas as pd
import pathlib as pl
# LAM modules
from run import main_catch_exit
from settings import settings as Sett
from tkinter import filedialog


class base_GUI(tk.Toplevel):
    """Container for the most important settings of the GUI."""

    def __init__(self, master=None):
        master.title("LAM-0.2.4")
        self.master = master
        self.master.grab_set()
        self.master.bind('<Escape>', self.func_destroy)
        self.master.bind('<Return>', self.RUN_button)
        # Fetch settings and transform to tkinter variables:
        self.handle = SettingHandler()  # !!!

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
        self.distf.grid(row=12, rowspan=8, columnspan=6, sticky="new",
                        pady=(0, 0))
        self.bottomf.grid(row=18, rowspan=2, columnspan=6, sticky="new",
                          pady=(15, 2))
        col_count, row_count = self.master.grid_size()
        for col in range(col_count):
            self.master.grid_columnconfigure(col, minsize=45)
        for row in range(row_count):
            self.master.grid_rowconfigure(row, minsize=32)

        # TOP FRAME / WORK DIRECTORY
        self.lbl1 = tk.Label(self.topf, text=self.handle('workdir').get(),
                             bg='white', textvariable=self.handle('workdir'),
                             bd=2, relief='sunken')
        self.browse = tk.Button(self.topf, text="Directory",
                                command=self.browse_button)
        self.DetChans = tk.StringVar(value="Detected channels:")
        self.DetGroups = tk.StringVar(value="Detected groups:")
        self.detect = tk.Button(self.topf, text="Detect",
                                command=self.Detect_Channels)
        self.lblGroups = tk.Label(self.topf, text=self.DetGroups.get(),
                                  textvariable=self.DetGroups)
        self.lblChannels = tk.Label(self.topf, text=self.DetChans.get(),
                                    textvariable=self.DetChans)
        self.lbl1.grid(row=0, column=1, columnspan=7)
        self.browse.grid(row=0, column=0)
        self.detect.grid(row=1, column=0)
        self.lblGroups.grid(row=1, column=1, columnspan=8, pady=(0, 0))
        self.lblChannels.grid(row=2, column=1, columnspan=8, pady=(0, 0))

        # MIDDLE FRAME / PRIMARY SETTINGS BOX
        stl = {1: 'groove', 2: 'lightgrey', 3: ('Arial', 8, 'bold')}
        self.pSample = tk.Checkbutton(self.midf, text="Process",
                                      variable=self.handle('process_samples'),
                                      relief=stl[1], bg=stl[2], font=stl[3],
                                      bd=4, command=self.Process_check)
        self.pSample.var = self.handle('process_samples')
        self.pCounts = tk.Checkbutton(self.midf, text="Count  ",
                                      variable=self.handle('process_counts'),
                                      relief=stl[1], bd=4, bg=stl[2],
                                      font=stl[3], command=self.Count_check)
        self.pDists = tk.Checkbutton(self.midf, text="Distance",
                                     variable=self.handle('process_dists'),
                                     relief=stl[1], bd=4, bg=stl[2],
                                     font=stl[3], command=self.Distance_check)
        self.pPlots = tk.Checkbutton(self.midf, text="Plots   ",
                                     variable=self.handle('Create_Plots'),
                                     relief=stl[1], bd=4, bg=stl[2],
                                     font=stl[3], command=self.Plot_check)
        self.pStats = tk.Checkbutton(self.midf, text="Stats   ",
                                     variable=self.handle('statistics'),
                                     relief=stl[1], bd=4, bg=stl[2],
                                     font=stl[3], command=self.Stat_check)
        self.pSample.grid(row=0, column=0, columnspan=1, padx=(2, 2))
        self.pCounts.grid(row=0, column=1, columnspan=1, padx=(2, 2))
        self.pDists.grid(row=0, column=2, columnspan=1, padx=(2, 2))
        self.pPlots.grid(row=0, column=3, columnspan=1, padx=(2, 2))
        self.pStats.grid(row=0, column=4, columnspan=1, padx=(2, 2))

        # Projection, Measurement point, widths, borders & file header settings
        self.pProj = tk.Checkbutton(self.midf, text="Project", relief='groove',
                                    font=('Arial', 8), bd=3,
                                    variable=self.handle('project'))
        self.pMP = tk.Checkbutton(self.midf, text="Use MP ", relief='groove',
                                  variable=self.handle('useMP'), bd=3,
                                  font=('Arial', 8), command=self.MP_check)
        self.pWidth = tk.Checkbutton(self.midf, text="Widths",
                                     variable=self.handle('measure_width'),
                                     relief='groove', bd=3, font=('Arial', 8),
                                     command=self.width_check)
        self.pBorder = tk.Checkbutton(self.midf, text="Borders", bd=3,
                                      relief='groove', font=('Arial', 8),
                                      variable=self.handle('border_detection'),
                                      command=self.border_check)
        self.lblMP = tk.Label(self.midf, text='MP label:', bd=1, font=('Arial',
                                                                       8))
        self.MPIn = tk.Entry(self.midf, text=self.handle('MPname').get(),
                             bg='white', textvariable=self.handle('MPname'),
                             bd=2, relief='sunken')
        self.lblHead = tk.Label(self.midf, bd=1, font=('Arial', 8),
                                text='Data header row:\n(from zero)')
        self.HeadIn = tk.Entry(self.midf, text=self.handle('header_row').get(),
                               bg='white', bd=2, relief='sunken',
                               textvariable=self.handle('header_row'))
        self.pProj.grid(row=1, column=0, columnspan=1, padx=(2, 2))
        self.pMP.grid(row=1, column=1, columnspan=1, padx=(2, 2))
        self.pWidth.grid(row=2, column=0, columnspan=1, padx=(2, 2))
        self.pBorder.grid(row=2, column=1, columnspan=1, padx=(2, 2))
        self.lblMP.grid(row=1, column=2)
        self.MPIn.grid(row=1, column=3, columnspan=3)
        self.lblHead.grid(row=2, column=2, columnspan=1)
        self.HeadIn.grid(row=2, column=3, columnspan=2)

        # BOTTOM BUTTONS
        self.r_stdoutC = tk.Checkbutton(self.bottomf, text="Redirect stdout",
                                        variable=self.handle('non_stdout'),
                                        relief='groove', bd=1,
                                        command=self.redirect_stdout)
        self.Run_b = tk.Button(self.bottomf, font=('Arial', 10, 'bold'),
                               text='Run\n<Enter>', command=self.RUN_button)
        self.quit_b = tk.Button(self.bottomf, font=('Arial', 9, 'bold'),
                                text="Quit", command=self.func_destroy)
        self.add_b = tk.Button(self.bottomf, font=('Arial', 9, 'bold'),
                               text="Other", command=self.Open_AddSettings)
        self.plot_b = tk.Button(self.bottomf, font=('Arial', 9, 'bold'),
                                text="Plots", command=self.Open_PlotSettings)
        self.stats_b = tk.Button(self.bottomf, font=('Arial', 9, 'bold'),
                                 text="Stats", command=self.Open_StatSettings)
        # Style
        self.Run_b.configure(height=2, width=7, bg='lightgreen', fg="black")
        self.quit_b.configure(height=1, width=5, fg="red")
        self.add_b.configure(height=2, width=7)
        self.plot_b.configure(height=2, width=7)
        self.stats_b.configure(height=2, width=7)
        # Grid
        self.r_stdoutC.grid(row=0, column=4, columnspan=4, sticky='n')
        self.Run_b.grid(row=1, column=4, columnspan=1, padx=(75, 25),
                        pady=(0, 0), sticky='ne')
        self.quit_b.grid(row=1, column=5, pady=(0, 0), sticky='nes')
        self.add_b.grid(row=1, column=0, columnspan=1, padx=(0, 5),
                        pady=(0, 0), sticky='nw')
        self.plot_b.grid(row=1, column=1, columnspan=1, padx=(0, 5),
                         pady=(0, 0), sticky='nw')
        self.stats_b.grid(row=1, column=2, columnspan=1, pady=(0, 0),
                          sticky='nw')

        # RIGHT FRAME / PLOTTING
        # header
        self.lbl2 = tk.Label(self.rightf, text='Plotting:', bd=2,
                             font=('Arial', 9, 'bold'))
        self.lbl2.grid(row=0, column=0)

        # create checkboxes
        self.chanC = tk.Checkbutton(self.rightf, text="Channels",
                                    variable=self.handle(
                                        'Create_Channel_Plots'))
        self.addC = tk.Checkbutton(self.rightf, text="Additional Data",
                                   variable=self.handle(
                                       'Create_AddData_Plots'))
        self.pairC = tk.Checkbutton(self.rightf, text="Channel Matrix",
                                    variable=self.handle(
                                        'Create_Channel_PairPlots'))
        self.heatC = tk.Checkbutton(self.rightf, text="Heatmaps",
                                    variable=self.handle('Create_Heatmaps'))
        self.distC = tk.Checkbutton(self.rightf, text="Distributions",
                                    variable=self.handle(
                                        'Create_Distribution_Plots'))
        self.statC = tk.Checkbutton(self.rightf, text="Statistics",
                                    variable=self.handle(
                                        'Create_Statistics_Plots'))
        self.clustC = tk.Checkbutton(self.rightf, text="Clusters",
                                     variable=self.handle(
                                         'Create_Cluster_Plots'))
        self.chanVSC = tk.Checkbutton(self.rightf, text="Channel VS. Add.",
                                      variable=self.handle(
                                          'Create_ChanVSAdd_Plots'))
        self.addVSC = tk.Checkbutton(self.rightf, text="Add. VS. Add.",
                                     variable=self.handle(
                                         'Create_AddVSAdd_Plots'))
        self.borderC = tk.Checkbutton(self.rightf, text="Borders",
                                      variable=self.handle(
                                          'Create_Border_Plots'))
        self.widthC = tk.Checkbutton(self.rightf, text="Widths",
                                     variable=self.handle('plot_width'))
        # Make grid
        self.chanC.grid(row=1, column=0, sticky='w', pady=(2, 0))
        self.heatC.grid(row=1, column=1, sticky='w', pady=(2, 0))
        self.addC.grid(row=2, column=0, sticky='w')
        self.clustC.grid(row=2, column=1, sticky='w')
        self.pairC.grid(row=3, column=0, sticky='w')
        self.statC.grid(row=3, column=1, sticky='w')
        self.distC.grid(row=4, column=0, sticky='w')
        self.borderC.grid(row=4, column=1, sticky='w')
        self.widthC.grid(row=5, column=0, sticky='w')
        self.chanVSC.grid(row=7, column=0, sticky='w', pady=(10, 0))
        self.addVSC.grid(row=8, column=0, sticky='ws', pady=(2, 30))

        # LEFT FRAME (UP) / VECTOR CREATION
        # header
        self.lbl3 = tk.Label(self.Up_leftf, text='Vector:', bd=2,
                             font=('Arial', 9, 'bold'))
        self.lbl3.grid(row=0, column=0)

        # vector type radio buttons
        self.Vbut1 = tk.Radiobutton(self.Up_leftf, text="Skeleton", value=1,
                                    variable=self.handle('SkeletonVector'),
                                    command=self.switch_pages)
        self.Vbut2 = tk.Radiobutton(self.Up_leftf, text="Median", value=0,
                                    variable=self.handle('SkeletonVector'),
                                    command=self.switch_pages)
        self.Vbut1.grid(row=1, column=0)
        self.Vbut2.grid(row=1, column=1)

        # vector channel input
        self.lbl4 = tk.Label(self.Up_leftf, text='Channel: ', bd=1,
                             font=('Arial', 10))
        self.lbl4.grid(row=2, column=0)
        self.chIn = tk.Entry(self.Up_leftf, bg='white', relief='sunken', bd=2,
                             text=self.handle('vectChannel').get(),
                             textvariable=self.handle('vectChannel'))
        self.chIn.grid(row=2, column=1, columnspan=1)

        # Bin number input
        self.lbl5 = tk.Label(self.Up_leftf, text='Bin #: ', bd=1,
                             font=('Arial', 10))
        self.binIn = tk.Entry(self.Up_leftf, bg='white', bd=2, relief='sunken',
                              text=self.handle('projBins').get(),
                              textvariable=self.handle('projBins'))
        self.lbl5.grid(row=3, column=0)
        self.binIn.grid(row=3, column=1, columnspan=1)

        # LEFT FRAME (LOWER) - VECTOR SETTINGS
        self.frames = {}
        self.vector_frame()

        # UPPER BOTTOM / DISTANCES
        # header
        self.lbldist = tk.Label(self.distf, text='Distance Calculations:',
                                bd=2, font=('Arial', 9, 'bold'))
        self.lbldist.grid(row=0, column=0, columnspan=6)

        # distance and cluster checkbuttons
        self.clustC = tk.Checkbutton(self.distf, text="Find clusters ",
                                     variable=self.handle('Find_Clusters'),
                                     command=self.Cluster_check, bd=1,
                                     relief='raised')
        self.FdistC = tk.Checkbutton(self.distf, text="Find distances",
                                     variable=self.handle('Find_Distances'),
                                     command=self.nearest_dist_check,
                                     bd=1, relief='raised')
        # Filtering
        test = any([bool(self.handle(v).get()) for v in ('inclusion',
                                                         'Cl_inclusion')])
        self.UseSubV = tk.BooleanVar(value=test)
        self.USubC = tk.Checkbutton(self.distf, text="Filter", relief='raised',
                                    variable=self.UseSubV, bd=1,
                                    command=self.Filter_check)
        # Add distance calculation's filter column name variable:
        self.colIn = tk.Entry(self.distf, text=self.handle('incl_col').get(),
                              bg='white', textvariable=self.handle('incl_col'),
                              bd=1, relief='sunken')
        # Create grid placement:
        self.colIn.grid(row=1, column=6, columnspan=2, sticky='n', pady=(2, 0))
        self.clustC.grid(row=1, column=0, columnspan=2, sticky='n')
        self.FdistC.grid(row=1, column=2, columnspan=2, sticky='n')
        self.USubC.grid(row=1, column=4, columnspan=2, sticky='n')
        self.Cllbl = tk.Label(self.distf, text="Clusters:")
        self.Cllbl.grid(row=2, column=0, columnspan=2)
        self.Distlbl = tk.Label(self.distf, text="Cell Distances:")
        self.Distlbl.grid(row=2, column=4, columnspan=2)

        # CLUSTERING settings
        self.ClChanlbl = tk.Label(self.distf, text="Channels:")
        self.ClChIn = tk.Entry(self.distf, bg='white', bd=2, relief='sunken',
                               text=self.handle('Cluster_Channels').get(),
                               textvariable=self.handle('Cluster_Channels'))
        self.ClChanlbl.grid(row=3, column=0, columnspan=2)
        self.ClChIn.grid(row=3, column=2, columnspan=2)

        self.ClDistlbl = tk.Label(self.distf, text='Max Dist.:')
        self.ClDistIn = tk.Entry(self.distf, bg='white', bd=2, relief='sunken',
                                 text=self.handle('Cl_maxDist').get(),
                                 textvariable=self.handle('Cl_maxDist'))
        self.ClDistlbl.grid(row=4, column=0, columnspan=2)
        self.ClDistIn.grid(row=4, column=2, columnspan=2)

        self.ClMinlbl = tk.Label(self.distf, text='Min cell #:')
        self.ClMinIn = tk.Entry(self.distf, bg='white', bd=2, relief='sunken',
                                text=self.handle('Cl_min').get(),
                                textvariable=self.handle('Cl_min'))
        self.ClMinlbl.grid(row=5, column=0, columnspan=2)
        self.ClMinIn.grid(row=5, column=2, columnspan=2)

        self.ClMaxlbl = tk.Label(self.distf, text='Max cell #:')
        self.ClMaxIn = tk.Entry(self.distf, bg='white', bd=2, relief='sunken',
                                text=self.handle('Cl_max').get(),
                                textvariable=self.handle('Cl_max'))
        self.ClMaxlbl.grid(row=6, column=0, columnspan=2)
        self.ClMaxIn.grid(row=6, column=2, columnspan=2)
        # Filtering
        self.ClSizlbl = tk.Label(self.distf, text='Filter value:')
        self.ClSizIn = tk.Entry(self.distf, bg='white', bd=2, relief='sunken',
                                text=self.handle('Cl_inclusion').get(),
                                textvariable=self.handle('Cl_inclusion'))
        self.ClSizlbl.grid(row=7, column=0, columnspan=2)
        self.ClSizIn.grid(row=7, column=2, columnspan=2)
        self.VClbut1 = tk.Radiobutton(self.distf, text="Greater", value=1,
                                      variable=self.handle('Cl_incl_type'))
        self.VClbut2 = tk.Radiobutton(self.distf, text="Smaller", value=0,
                                      variable=self.handle('Cl_incl_type'))
        self.VClbut1.grid(row=8, column=0)
        self.VClbut2.grid(row=8, column=2)

        # DISTANCE settings
        self.DChanlbl = tk.Label(self.distf, text="Channels:")
        self.DChIn = tk.Entry(self.distf, bg='white', bd=2, relief='sunken',
                              text=self.handle('Distance_Channels').get(),
                              textvariable=self.handle('Distance_Channels'))
        self.DChanlbl.grid(row=3, column=4, columnspan=2)
        self.DChIn.grid(row=3, column=6, columnspan=2)

        self.DDistlbl = tk.Label(self.distf, text='Max Dist.:')
        self.DDistIn = tk.Entry(self.distf, bg='white', bd=2, relief='sunken',
                                text=self.handle('maxDist').get(),
                                textvariable=self.handle('maxDist'))
        self.DDistlbl.grid(row=4, column=4, columnspan=2)
        self.DDistIn.grid(row=4, column=6, columnspan=2)
        # Nearestdist target channel
        self.DTarget = tk.Checkbutton(self.distf, text='Use target:',
                                      variable=self.handle('use_target'),
                                      command=self.Target_check)
        self.DTargetIn = tk.Entry(self.distf, bg='white', relief='sunken',
                                  text=self.handle('target_chan').get(), bd=2,
                                  textvariable=self.handle('target_chan'))
        self.DTarget.grid(row=5, column=4, columnspan=2)
        self.DTargetIn.grid(row=5, column=6, columnspan=2)
        # Filtering
        self.DSizlbl = tk.Label(self.distf, text='Filter value:')
        self.DSizIn = tk.Entry(self.distf, bg='white', bd=2, relief='sunken',
                               text=self.handle('inclusion').get(),
                               textvariable=self.handle('inclusion'))
        self.DSizlbl.grid(row=7, column=4, columnspan=2)
        self.DSizIn.grid(row=7, column=6, columnspan=2)

        self.VDbut1 = tk.Radiobutton(self.distf, text="Greater", value=1,
                                     variable=self.handle('incl_type'))
        self.VDbut2 = tk.Radiobutton(self.distf, text="Smaller", value=0,
                                     variable=self.handle('incl_type'))
        self.VDbut1.grid(row=8, column=5)
        self.VDbut2.grid(row=8, column=7)

        # # Disable / enable widgets
        self.Process_check()
        self.Count_check()
        self.Distance_check()
        self.Stat_check()
        self.redirect_stdout()

    def browse_button(self):
        """Allow input of path when browse-button is pressed."""
        filename = filedialog.askdirectory()
        self.handle('workdir').set(filename)
        self.Detect_Channels()

    def border_check(self):
        """Control border detection related settings."""
        if self.handle('Create_Plots').get():
            if self.handle('border_detection').get():
                self.borderC.configure(state='normal')
            else:
                self.borderC.configure(state='disable')
        else:
            self.handle('Create_Border_Plots').set(False)

    def Cluster_check(self):
        """Relevant changes when cluster-setting is checked."""
        widgets = [self.ClChanlbl, self.ClChIn, self.ClDistlbl, self.ClDistIn,
                   self.ClMinlbl, self.ClMinIn, self.ClMaxlbl, self.ClMaxIn,
                   self.ClSizlbl, self.ClSizIn, self.VClbut1, self.VClbut2,
                   self.Cllbl]
        if not self.handle('Find_Clusters').get():
            for widget in widgets:
                widget.configure(state='disable')
        else:
            for widget in widgets:
                widget.configure(state='normal')
            self.Filter_check()

    def Count_check(self):
        """Relevant changes when count-setting is checked."""
        widgets = [self.binIn, self.lbl5, self.lblMP, self.MPIn, self.pMP,
                   self.pProj, self.pWidth]
        if not self.handle('process_counts').get():
            for wdg in widgets:
                wdg.configure(state='disable')
        else:
            self.lblHead.configure(state='normal')
            self.HeadIn.configure(state='normal')
            for wdg in widgets:
                wdg.configure(state='normal')
        check_switch(self.MP_check, self.width_check, self.border_check,
                     self.run_check)

    def Distance_check(self):
        """Relevant changes when distances-setting is checked."""
        if not self.handle('process_dists').get():
            for widget in self.distf.winfo_children():
                widget.configure(state='disable')
        else:
            for widget in self.distf.winfo_children():
                if int(widget.grid_info()["row"]) in [0, 1, 2]:
                    widget.configure(state='normal')
            check_switch(self.Cluster_check, self.nearest_dist_check,
                         self.Filter_check)
        self.run_check()

    def Filter_check(self):
        """Relevant changes when filtering by size is checked."""
        if not self.UseSubV.get():
            self.handle('Cl_inclusion').set(0)
            self.handle('inclusion').set(0)
            for widget in [self.DSizlbl, self.DSizIn, self.VDbut1, self.VDbut2,
                           self.ClSizlbl, self.ClSizIn, self.VClbut1,
                           self.VClbut2, self.colIn]:
                widget.configure(state='disable')
        else:
            if self.handle('Find_Distances').get():
                for widget in [self.DSizlbl, self.DSizIn, self.VDbut1,
                               self.VDbut2, self.colIn]:
                    widget.configure(state='normal')
            if self.handle('Find_Clusters').get():
                for widget in [self.ClSizlbl, self.ClSizIn, self.VClbut1,
                               self.VClbut2, self.colIn]:
                    widget.configure(state='normal')

    def MP_check(self):
        """Relevant changes when MP is in use or not."""
        if not self.handle('useMP').get():
            self.lblMP.configure(state='disable')
            self.MPIn.configure(state='disable')
        else:
            self.lblMP.configure(state='normal')
            self.MPIn.configure(state='normal')

    def nearest_dist_check(self):
        """Relevant changes when find distance-setting is checked."""
        widgets = [self.DChanlbl, self.DChIn, self.DDistlbl, self.DDistIn,
                   self.DTarget, self.DTargetIn, self.DSizlbl, self.DSizIn,
                   self.VDbut1, self.VDbut2, self.Distlbl]
        if not self.handle('Find_Distances').get():
            for widget in widgets:
                widget.configure(state='disable')
        else:
            for widget in widgets:
                widget.configure(state='normal')
            check_switch(self.Target_check, self.Filter_check)

    def Plot_check(self):
        """Relevant changes when plot-setting is checked."""
        if self.handle('Create_Plots').get() is False:
            self.handle('Create_Border_Plots').set(False)
            self.handle('plot_width').set(False)
            self.plot_b.configure(state='disable')
            for widget in self.rightf.winfo_children():
                widget.configure(state='disable')
        else:
            self.plot_b.configure(state='normal')
            for widget in self.rightf.winfo_children():
                if not self.handle('statistics').get() and (widget.cget('text')
                                                            == 'Statistics'):
                    continue
                widget.configure(state='normal')
        self.run_check()

    def Process_check(self):
        """Relevant changes when Process-setting is checked."""
        wids = [self.binIn, self.lbl5, self.chIn, self.lbl4]
        if not self.handle('process_samples').get():
            for widget in self.Up_leftf.winfo_children():
                if widget not in wids:
                    widget.configure(state='disable')
            hidev = 'disable'
        else:
            for widget in self.Up_leftf.winfo_children():
                if widget not in wids:
                    widget.configure(state='normal')
            self.switch_pages()
            hidev = 'normal'
        if not self.handle('SkeletonVector').get():
            for widget in self.frames[Median_settings].winfo_children():
                widget.configure(state=hidev)
        else:
            for widget in self.frames[Skel_settings].winfo_children():
                widget.configure(state=hidev)
        self.run_check()

    def run_check(self):
        """Determine if run button should be active."""
        sets = ['process_samples', 'process_counts', 'process_dists',
                'Create_Plots', 'statistics']
        if any([self.handle(v).get() for v in sets]):
            self.Run_b.configure(state='normal', bg='lightgreen')
        else:
            self.Run_b.configure(state='disable', bg='lightgrey')

    def Stat_check(self):
        """Relevant changes when statistics-setting is checked."""
        if not self.handle('statistics').get():
            self.stats_b.configure(state='disable')
            self.statC.configure(state='disable')
            self.handle('Create_Statistics_Plots').set(False)
        else:
            self.stats_b.configure(state='normal')
            if self.handle('Create_Plots').get():
                self.statC.configure(state='normal')
        self.run_check()

    def Target_check(self):
        """Relevant changes when target-setting is checked."""
        if not self.handle('use_target').get():
            self.DTargetIn.configure(state='disable')
        else:
            self.DTargetIn.configure(state='normal')

    def RUN_button(self, event=None):
        """Relevant changes when Run-button is pressed + run initialization."""
        # Get modified options
        options = self.handle.translate()
        options['workdir'] = pl.Path(options['workdir'])  # Transform workdir
        # If needed, change settings that have high risk of interfering
        if not options['process_counts']:
            ops = ('measure_width', 'useMP', 'project')
            options.update({k: False for k in ops})
        # Handle filtering options in distance calculations
        if self.UseSubV:
            options['Cl_incl_type'] = '' if options['Cl_incl_type'] == 0 else 'greater'
            options['incl_type'] = '' if options['incl_type'] == 0 else 'greater'

        # SAVE SETTING
        self.handle.change_settings(options)
        
        # CREATE LOGGER
        import logger as lg
        import logging
        if lg.log_created is True:
            # Close old loggers and create new:
            lg.Close()
            lg.Update()
            LAM_logger = logging.getLogger(__name__)
        else:
            LAM_logger = lg.setup_logger(__name__, new=True)
        lg.print_settings()

        # RUN
        lg.logprint(LAM_logger, '### Run parameters set. Begin run ###', 'i')
        main_catch_exit(gui_root=self.master)

    def redirect_stdout(self):
        """Change stdout direction based on r_stdout check box."""
        import redirect as rd
        if self.handle('non_stdout').get():
            self.stdout_win = rd.text_window(self.master,
                                             self.handle('non_stdout'))
        else:
            if hasattr(self, 'stdout_win'):
                self.stdout_win.func_destroy()

    def show_VSett(self, name):
        """Change shown vector settings based on type."""
        for frame in self.frames.values():
            frame.grid_remove()
        frame = self.frames[name]
        frame.grid()

    def switch_pages(self):
        """Switch page of vector settings."""
        if not self.handle('SkeletonVector').get():
            self.show_VSett(Median_settings)
        else:
            self.show_VSett(Skel_settings)

    def func_destroy(self, event=None):
        """Destroy GUI."""
        import logger as lg
        lg.log_Shutdown()
        if hasattr(self, 'stdout_win'):
            self.stdout_win.func_destroy()
        self.master.destroy()

    def Open_AddSettings(self):
        """Open additional settings window."""
        adds = Additional_data(self.master)
        adds.window.wait_window()
        ind = adds.handle.vars.loc[adds.handle.vars.check].index
        self.handle.vars.loc[ind, :] = adds.handle.vars.loc[ind, :]

    def Open_PlotSettings(self):
        """Open plot settings window."""
        pvars = Plot_Settings(self.master)
        pvars.window.wait_window()
        ind = pvars.handle.vars.loc[pvars.handle.vars.check].index
        self.handle.vars.loc[ind, :] = pvars.handle.vars.loc[ind, :]

    def Open_StatSettings(self):
        """Open statistics settings window."""
        svars = Stat_Settings(self.master)
        svars.window.wait_window()
        ind = svars.handle.vars.loc[svars.handle.vars.check].index
        self.handle.vars.loc[ind, :] = svars.handle.vars.loc[ind, :]

    def Detect_Channels(self):
        """Detect channels and groups found at current set path."""
        workdir = pl.Path(self.handle('workdir').get())
        DetChans, DetGroups = set(), set()
        # Loop found sample directories
        for samplepath in [p for p in workdir.iterdir() if p.is_dir() and
                           'Analysis Data' not in p.name]:
            try:  # Get groups from folder names
                group = str(samplepath.name).split('_')[0]
                DetGroups.add(group)
                # Loop through channels of found samples
                cpaths = [p for p in samplepath.iterdir() if p.is_dir()]
                for channelpath in cpaths:
                    channel = str(channelpath.name).split('_')[-2]
                    DetChans.add(channel)
            except (IndexError, TypeError):
                pass
        # Change text variables to contain new groups and channels
        if DetChans:
            chanstring = tk.StringVar(value="Detected channels: {}".format(
                                            ', '.join(sorted(DetChans))))
        else:
            chanstring = tk.StringVar(value='No detected channels!')
        if DetGroups:
            msg = ', '.join(sorted(DetGroups))
            grpstring = tk.StringVar(value=f"Detected groups: {msg}")
        else:
            grpstring = tk.StringVar(value='No detected groups!')
        # Set new text variables to be shown
        self.DetGroups.set(grpstring.get())
        self.DetChans.set(chanstring.get())
        from settings import store
        store.samplegroups = DetGroups
        store.channels = [c for c in DetChans if
                          c.lower() != self.handle('MPname').get().lower()]

    def vector_frame(self):
        for F in (Skel_settings, Median_settings):
            frame = F(self.master, self, self.handle)
            self.frames[F] = frame
            frame.grid(row=8, column=0, columnspan=3, rowspan=5, sticky="new")
            frame.grid_remove()
        if self.handle('SkeletonVector').get():
            self.show_VSett(Skel_settings)
        else:
            self.show_VSett(Median_settings)

    def width_check(self):
        """Enable width estimation related settings."""
        if self.handle('Create_Plots').get():
            if self.handle('measure_width').get():
                self.widthC.configure(state='normal')
            else:
                self.widthC.configure(state='disable')
        else:
            self.handle('plot_width').set(False)


class Skel_settings(tk.Frame):
    """Container for skeleton vector-related settings."""

    def __init__(self, parent, master, handle):
        tk.Frame.__init__(self, parent, bd=2, relief='groove')
        # Container label
        self.lblSetS = tk.Label(self, text='Vector Parameters:', bd=1,
                                font=('Arial', 10))
        # Container widgets
        self.lbl6 = tk.Label(self, text='Simplify tol.', bd=1,
                             font=('Arial', 9))
        self.simpIn = tk.Entry(self, text=handle('simplifyTol').get(),
                               textvariable=handle('simplifyTol'), bd=2,
                               bg='white', relief='sunken')
        self.lbl7 = tk.Label(self, text='Resize', bd=1,
                             font=('Arial', 9))
        self.rszIn = tk.Entry(self, text=handle('SkeletonResize').get(),
                              textvariable=handle('SkeletonResize'), bd=2,
                              bg='white', relief='sunken')
        self.lbl8 = tk.Label(self, text='Find distance', bd=1,
                             font=('Arial', 9))
        self.distIn = tk.Entry(self, text=handle('find_dist').get(),
                               textvariable=handle('find_dist'), bd=2,
                               bg='white', relief='sunken')
        self.lbl9 = tk.Label(self, text='Dilation iter', bd=1,
                             font=('Arial', 9))
        self.dilIn = tk.Entry(self, text=handle('BDiter').get(),
                              textvariable=handle('BDiter'), bd=2,
                              bg='white', relief='sunken')
        self.lbl10 = tk.Label(self, text='Smoothing', bd=1,
                              font=('Arial', 9))
        self.smoothIn = tk.Entry(self, text=handle('SigmaGauss').get(),
                                 textvariable=handle('SigmaGauss'), bd=2,
                                 bg='white', relief='sunken')
        # Make grid
        self.lblSetS.grid(row=0, column=0, columnspan=3, pady=(0, 3))
        self.lbl6.grid(row=1, column=0, columnspan=1)
        self.simpIn.grid(row=1, column=1)
        self.lbl7.grid(row=2, column=0, columnspan=1)
        self.rszIn.grid(row=2, column=1)
        self.lbl8.grid(row=3, column=0, columnspan=1)
        self.distIn.grid(row=3, column=1)
        self.lbl9.grid(row=4, column=0, columnspan=1)
        self.dilIn.grid(row=4, column=1)
        self.lbl10.grid(row=5, column=0, columnspan=1, pady=(0, 0))
        self.smoothIn.grid(row=5, column=1, pady=(0, 0))


class Median_settings(tk.Frame):
    """Container for median vector-related settings."""

    def __init__(self, parent, master, handle):
        # Container label
        tk.Frame.__init__(self, parent, bd=2, relief='groove')

        self.lblSetM = tk.Label(self, text='Vector Parameters:', bd=1,
                                font=('Arial', 10))
        self.lblSetM.grid(row=0, column=0, columnspan=3, pady=(0, 3))
        # Container widgets
        self.lbl6 = tk.Label(self, text='Simplify tol.', bd=1,
                             font=('Arial', 9))
        self.simpIn = tk.Entry(self, bg='white', bd=2, relief='sunken',
                               text=handle('simplifyTol').get(),
                               textvariable=handle('simplifyTol'))
        self.lbl7 = tk.Label(self, text='Median bins  ', bd=1,
                             font=('Arial', 9))
        self.mbinIn = tk.Entry(self, bg='white', bd=2, relief='sunken',
                               text=handle('medianBins').get(),
                               textvariable=handle('medianBins'))
        # Make grid
        self.lbl6.grid(row=1, column=0, columnspan=1)
        self.mbinIn.grid(row=2, column=1, pady=(0, 63))
        self.simpIn.grid(row=1, column=1)
        self.lbl7.grid(row=2, column=0, columnspan=1, pady=(0, 63))


class Additional_data():
    """Container for Other-window settings."""

    def __init__(self, master):
        self.handle = SettingHandler()

        self.window = tk.Toplevel(master)
        self.window.grab_set()
        self.window.title("Additional Data Settings")
        self.window.bind('<Escape>', self.func_destroy)
        self.window.bind('<Return>', self.window.destroy)
        self.window.protocol("WM_DELETE_WINDOW", self.func_destroy)
        # Frames
        self.frame = tk.Frame(self.window)
        self.Dframe = tk.Frame(self.window, relief='groove')
        self.Bframe = tk.Frame(self.window)
        self.frame.grid(row=0, rowspan=11, columnspan=9, sticky="new")
        self.Dframe.grid(row=8, rowspan=5, columnspan=9, sticky="new")
        self.Bframe.grid(row=13, rowspan=2, columnspan=9, sticky="ne")
        # Adjust grid pixel sizes
        col_count, row_count = self.window.grid_size()
        for col in range(col_count):
            self.window.grid_columnconfigure(col, minsize=45)
        for row in range(row_count):
            self.window.grid_rowconfigure(row, minsize=32)

        # ADDITIONAL DATA ENTRIES
        # Create example variables for entries
        ex_str = ["Area", "Area.csv", "Area, $\u03BCm^2$"]
        self.insert = [tk.StringVar(value=s) for s in ex_str]

        # Entry for names of data columns
        self.lbl1 = tk.Label(self.frame, text='Column label', bd=1,
                             font=('Arial', 9))
        self.lblIn = tk.Entry(self.frame, text=self.insert[0].get(),
                              bg='white', textvariable=self.insert[0], bd=2,
                              relief='sunken')
        self.lbl1.grid(row=0, column=0, columnspan=2)
        self.lblIn.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        # Entry for data file name:
        self.lbl2 = tk.Label(self.frame, text='csv-file', bd=1,
                             font=('Arial', 9))
        self.fileIn = tk.Entry(self.frame, text=self.insert[1].get(),
                               bg='white', textvariable=self.insert[1], bd=2,
                               relief='sunken')
        self.lbl2.grid(row=0, column=3, columnspan=2)
        self.fileIn.grid(row=1, column=3, columnspan=2, pady=(0, 10))

        # Unit of data type:
        self.lbl3 = tk.Label(self.frame, text='Unit', bd=1,
                             font=('Arial', 9))
        self.unitIn = tk.Entry(self.frame, text=self.insert[2].get(),
                               bg='white', textvariable=self.insert[2], bd=2,
                               relief='sunken')
        self.lbl3.grid(row=0, column=5, columnspan=2)
        self.unitIn.grid(row=1, column=5, columnspan=2, pady=(0, 10))

        # BUTTONS
        self.Add_b = tk.Button(self.frame, text='Add', command=self.add_data,
                               font=('Arial', 10, 'bold'))
        self.Add_b.configure(bg='lightgreen', fg="darkgreen")
        self.Save_b = tk.Button(self.Bframe, text='Save & Return\n<Enter>',
                                font=('Arial', 10, 'bold'),
                                command=self.window.destroy)
        self.Save_b.configure(height=2, width=12, bg='lightgreen',
                              fg="darkgreen")
        self.exit_b = tk.Button(self.Bframe, text="Return",
                                font=('Arial', 9, 'bold'),
                                command=self.func_destroy)
        self.exit_b.configure(height=1, width=5, fg="red")

        self.Add_b.grid(row=0, column=7, rowspan=2, padx=(5, 10), pady=(0, 10))
        self.Save_b.grid(row=11, column=5, rowspan=2, columnspan=2, padx=(0,
                                                                          10))
        self.exit_b.grid(row=11, column=7)

        # additional data labels and removal buttons:
        self.rowN = len(self.handle('AddData'))
        self.buttons = []

        for i, (key, vals) in enumerate(self.handle('AddData').items()):
            row = i+2
            val = vals.get().split(', ')
            tk.Label(self.frame, text=key, bd=2, bg='lightgrey',
                     relief='groove').grid(row=row, column=0, columnspan=2)
            tk.Label(self.frame, text=val[0], bd=2, bg='lightgrey',
                     relief='groove').grid(row=row, column=3, columnspan=2)
            tk.Label(self.frame, text=', '.join(val[1:]), bd=2,
                     bg='lightgrey', relief='groove').grid(row=row, column=5,
                                                           columnspan=2)

            self.buttons.append(tk.Button(self.frame, text='x',
                                          font=('Arial', 10), relief='raised',
                                          command=lambda i=i:
                                              self.rmv_data(i)))
            self.buttons[i].grid(row=row, column=7, sticky='w')

        # additional data ID-replacement
        self.repIDC = tk.Checkbutton(self.Dframe, text="Replace file ID",
                                     variable=self.handle('replaceID'),
                                     relief='groove', bd=4,
                                     command=self.replace_check)
        self.chanlbl = tk.Label(self.Dframe, text='File descriptor:', bd=1)
        self.changelbl = tk.Label(self.Dframe, text='Change to:', bd=1)
        self.repIDC.grid(row=0, column=0, columnspan=4)
        self.chanlbl.grid(row=1, column=0, columnspan=3)
        self.changelbl.grid(row=1, column=3, columnspan=3)

        for i, (key, val) in enumerate(self.handle('channelID').items()):
            fileID = tk.StringVar(value=key)
            self.fIDIn = tk.Entry(self.Dframe, text=fileID.get(), bg='white',
                                  textvariable=fileID, bd=2, relief='sunken')
            self.cIDIn = tk.Entry(self.Dframe, text=val.get(), bg='white',
                                  textvariable=val, bd=2, relief='sunken')
            self.cIDIn.var = val
            row = i+2
            self.fIDIn.grid(row=row, column=0, columnspan=3)
            self.cIDIn.grid(row=row, column=3, columnspan=3)
        self.replace_check()

    def replace_check(self):
        """Change relevant settings when replaceID is checked."""
        if not self.handle('replaceID').get():
            for child in self.Dframe.winfo_children():
                if isinstance(child, tk.Entry):
                    child.configure(state='disable')
        else:
            for child in self.Dframe.winfo_children():
                if isinstance(child, tk.Entry):
                    child.configure(state='normal')

    def add_data(self):
        """Addition of data input to the additional data table."""
        if self.insert[0].get() not in self.handle('AddData').keys():
            row = self.rowN+2
            tk.Label(self.frame, text=self.insert[0].get(), bd=2,
                     bg='lightgrey', relief='groove').grid(row=row, column=0,
                                                           columnspan=2)
            tk.Label(self.frame, text=self.insert[1].get(), bd=2,
                     bg='lightgrey', relief='groove').grid(row=row, column=3,
                                                           columnspan=2)
            tk.Label(self.frame, text=self.insert[2].get(), bd=2,
                     bg='lightgrey', relief='groove').grid(row=row, column=5,
                                                           columnspan=2)
            self.buttons.append(tk.Button(self.frame, text='x',
                                          font=('Arial', 10), relief='raised',
                                          command=lambda i=self.rowN:
                                              self.rmv_data(i)))
            self.buttons[self.rowN].grid(row=row, column=7, sticky='w')
            var = [get_tkvar([self.insert[1].get(), self.insert[2].get()])]
            self.handle('AddData').update({self.insert[0].get(): var})
            self.rowN = self.rowN + 1
        else:
            print("WARNING: Attempted to overwrite a data label!")
            print(" --> Delete old label of same name before adding.")

    def rmv_data(self, i):
        """Remove data from the additional data table."""
        for widget in self.frame.grid_slaves():
            if int(widget.grid_info()["row"]) == i+2 and int(
                    widget.grid_info()["column"]) == 0:
                key = widget.cget("text")
                if key in self.handle('AddData').keys():
                    self.handle('AddData').pop(key, None)
                    widget.grid_forget()
                else:
                    print("WARNING: removed label not found in add. data.")
            elif int(widget.grid_info()["row"]) == i+2:
                widget.grid_forget()

    def func_destroy(self, event=None):
        """Destroy window without saving."""
        self.window.destroy()
        self.handle.vars.loc[:, 'check'] = False


class Plot_Settings():
    """Container for Other-window settings."""

    def __init__(self, master):
        self.handle = SettingHandler()
        self.window = tk.Toplevel(master)
        self.window.grab_set()
        self.window.title("Plot Settings")
        self.window.bind('<Escape>', self.func_destroy)
        self.window.bind('<Return>', self.window.destroy)
        self.window.protocol("WM_DELETE_WINDOW", self.func_destroy)
        # Frames
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
        self.DropC = tk.Checkbutton(self.frame, text="Drop outliers", bd=1,
                                    variable=self.handle('Drop_Outliers'),
                                    relief='groove', command=self.Drop_check)
        self.lbl2 = tk.Label(self.frame, text='Std dev.:')
        self.stdIn = tk.Entry(self.frame, text=self.handle('dropSTD').get(),
                              textvariable=self.handle('dropSTD'), bg='white',
                              bd=2, relief='sunken')
        self.lbl1.grid(row=0, column=0, columnspan=2)
        self.DropC.grid(row=1, column=0, columnspan=2)
        self.lbl2.grid(row=2, column=0, columnspan=2)
        self.stdIn.grid(row=2, column=2, columnspan=2)
        self.Drop_check()
        # Pairplot jitter
        self.JitterC = tk.Checkbutton(self.frame, text="Pair plot jitter",
                                      variable=self.handle('plot_jitter'))
        self.JitterC.grid(row=3, column=0, columnspan=2)
        # Save format
        self.lbl3 = tk.Label(self.frame, text='Save format:')
        self.SFIn = tk.Entry(self.frame, bg='white', bd=2, relief='sunken',
                             text=self.handle('saveformat'),
                             textvariable=self.handle('saveformat'))
        comment = "Supported formats:\n\
        eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba,\nsvg, svgz, tif, tiff"
        self.lbl4 = tk.Label(self.frame, text=comment, fg='dimgrey')
        self.lbl3.grid(row=4, column=0, columnspan=1)
        self.SFIn.grid(row=4, column=1, columnspan=2)
        self.lbl4.grid(row=5, column=0, columnspan=4, pady=(0, 10))
        # versus
        self.lbl8 = tk.Label(self.frame, text='Versus plots:')
        self.lbl9 = tk.Label(self.frame, text='Plotted channels:')
        self.setVsCh = tk.StringVar(value=','.join(Sett.vs_channels))
        self.VsChIn = tk.Entry(self.frame, bg='white', bd=2, relief='sunken',
                               text=self.handle('vs_channels').get(),
                               textvariable=self.handle('vs_channels'))
        self.VsAddIn = tk.Entry(self.frame, bg='white', bd=2, relief='sunken',
                                text=self.handle('vs_adds').get(),
                                textvariable=self.handle('vs_adds'))
        self.lbl8 = tk.Label(self.frame, text='Plotted add. data:')
        self.lbl8.grid(row=6, column=0, columnspan=3)
        self.lbl9.grid(row=7, column=0, columnspan=2)
        self.VsChIn.grid(row=7, column=2, columnspan=2)
        self.lbl8.grid(row=8, column=0, columnspan=2)
        self.VsAddIn.grid(row=8, column=2, columnspan=2)

        # Statistics settings:
        self.lbl5 = tk.Label(self.statframe, text='Statistical Plotting:')
        self.starC = tk.Checkbutton(self.statframe, text="Sign. stars",
                                    variable=self.handle('stars'),
                                    command=self.sign_check)
        self.CfillC = tk.Checkbutton(self.statframe, text="Sign. color",
                                     variable=self.handle('fill'))
        self.neglogC = tk.Checkbutton(self.statframe, text="Neg. log2",
                                      variable=self.handle('negLog2'),
                                      command=self.sign_check)
        self.lbl7 = tk.Label(self.statframe, text='y-limit:')
        self.ylimIn = tk.Entry(self.statframe, text=self.handle('ylim').get(),
                               bg='white', textvariable=self.handle('ylim'),
                               bd=2, relief='sunken')
        self.lbl5.grid(row=0, column=0, columnspan=3)
        self.starC.grid(row=1, column=0, columnspan=2)
        self.CfillC.grid(row=1, column=2, columnspan=2)
        self.neglogC.grid(row=2, column=0, columnspan=1)
        self.lbl7.grid(row=2, column=1, columnspan=1)
        self.ylimIn.grid(row=2, column=2, columnspan=2)

        self.sign_check()
        # BUTTONS
        self.Save_b = tk.Button(self.Bframe, text='Save & Return\n<Enter>',
                                font=('Arial', 10, 'bold'),
                                command=self.window.destroy)
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
        if not self.handle('Drop_Outliers').get():
            self.stdIn.configure(state='disable')
        else:
            self.stdIn.configure(state='normal')

    def sign_check(self):
        """Relevant changes when neglog2-setting is selected."""
        if not self.handle('negLog2').get():
            self.starC.configure(state='normal')
            self.lbl7.configure(state='disable')
            self.ylimIn.configure(state='disable')
        else:
            self.handle('stars').set(False)
            self.starC.configure(state='disable')
            self.lbl7.configure(state='normal')
            self.ylimIn.configure(state='normal')

    def func_destroy(self, event=None):
        """Destroy window without saving."""
        self.window.destroy()
        self.handle.vars.loc[:, 'check'] = False


class Stat_Settings():
    """Container for statistics-window settings."""

    def __init__(self, master):
        self.handle = SettingHandler()
        self.window = tk.Toplevel(master)
        self.window.grab_set()
        self.window.title("Statistics Settings")
        self.window.bind('<Escape>', self.func_destroy)
        self.window.bind('<Return>', self.window.destroy)
        self.window.protocol("WM_DELETE_WINDOW", self.func_destroy)
        self.frame = tk.Frame(self.window)
        self.frame.grid(row=0, rowspan=4, columnspan=4, sticky="new")
        self.Winframe = tk.Frame(self.window, bd=1, relief='groove')
        self.Winframe.grid(row=5, rowspan=2, columnspan=4, sticky="n")
        self.But_frame = tk.Frame(self.window)
        self.But_frame.grid(row=7, rowspan=2, columnspan=4, sticky="ne")
        # Control group
        self.lbl1 = tk.Label(self.frame, text='Control Group:')
        self.CtrlIn = tk.Entry(self.frame, bg='white', bd=2, relief='sunken',
                               text=self.handle('cntrlGroup').get(),
                               textvariable=self.handle('cntrlGroup'))
        self.lbl1.grid(row=0, column=0, columnspan=2, pady=4)
        self.CtrlIn.grid(row=0, column=2, columnspan=2, pady=4)
        # Statistic types
        self.StatT = tk.Checkbutton(self.frame, text="Total Statistics",
                                    variable=self.handle('stat_total'),
                                    relief='groove', bd=1)
        self.StatV = tk.Checkbutton(self.frame, text="Group vs Group",
                                    variable=self.handle('stat_versus'),
                                    relief='groove', bd=1)
        self.StatT.grid(row=1, column=0, columnspan=2, pady=4)
        self.StatV.grid(row=1, column=2, columnspan=2, pady=4)
        # Alpha
        self.lbl2 = tk.Label(self.frame, text='Alpha:')
        self.AlphaIn = tk.Entry(self.frame, text=self.handle('alpha').get(),
                                bg='white', textvariable=self.handle('alpha'),
                                bd=2, relief='sunken')
        self.lbl2.grid(row=2, column=0, columnspan=2, pady=4)
        self.AlphaIn.grid(row=2, column=2, columnspan=2, pady=4)
        # Stat window
        self.WindC = tk.Checkbutton(self.frame, text="Windowed statistics",
                                    variable=self.handle('windowed'),
                                    relief='raised', bd=1,
                                    command=self.Window_check)
        self.WindC.grid(row=3, column=0, columnspan=3, pady=(10, 0))
        self.lbl2 = tk.Label(self.Winframe, text='Trailing window:')
        self.TrailIn = tk.Entry(self.Winframe, text=self.handle('trail').get(),
                                bg='white', textvariable=self.handle('trail'),
                                bd=2, relief='sunken')
        self.lbl3 = tk.Label(self.Winframe, text='Leading window:')
        self.LeadIn = tk.Entry(self.Winframe, text=self.handle('lead').get(),
                               bg='white', textvariable=self.handle('lead'),
                               bd=2, relief='sunken')
        self.lbl2.grid(row=0, column=0, columnspan=2, pady=1)
        self.TrailIn.grid(row=0, column=2, columnspan=2, pady=1)
        self.lbl3.grid(row=1, column=0, columnspan=2, pady=(1, 5))
        self.LeadIn.grid(row=1, column=2, columnspan=2, pady=(1, 5))
        self.Window_check()
        # Buttons
        self.Save_b = tk.Button(self.But_frame, text='Save & Return\n<Enter>',
                                font=('Arial', 10, 'bold'),
                                command=self.window.destroy)
        self.Save_b.configure(height=2, width=12, bg='lightgreen',
                              fg="darkgreen")
        self.exit_b = tk.Button(self.But_frame, text="Return",
                                font=('Arial', 9, 'bold'),
                                command=self.func_destroy)
        self.exit_b.configure(height=1, width=5, fg="red")
        self.Save_b.grid(row=0, column=2, rowspan=2, columnspan=2,
                         padx=(0, 10))
        self.exit_b.grid(row=0, column=4)

    def Window_check(self):
        """Relevant changes when windowed statistics is selected."""
        if not self.handle('windowed').get():
            for widget in self.Winframe.winfo_children():
                widget.configure(state='disable')
        else:
            for widget in self.Winframe.winfo_children():
                widget.configure(state='normal')

    def func_destroy(self, event=None):
        """Destroy stats-window."""
        self.window.destroy()
        self.handle.vars.loc[:, 'check'] = False


class SettingHandler:

    # Fetch defaults from settings.py (after parsing cmdl arguments)
    default_settings = {k: v for k, v in vars(Sett).items() if "__" not in k}
    setting_names = sorted(default_settings.keys())

    def __init__(self):
        self.vars = pd.DataFrame(index=SettingHandler.setting_names)
        self.vars = self.vars.assign(check=False, ref=None)

    def __call__(self, variable_name):
        if self.vars.at[variable_name, 'check']:
            return self.vars.at[variable_name, 'ref']
        vref = get_ref(variable_name)
        self.vars.loc[variable_name, :] = [True, vref]
        return self.vars.at[variable_name, 'ref']

    def translate(self):
        mods = self.vars.loc[self.vars.check, 'ref'].to_dict()
        for key, value in mods.items():
            if isinstance(value, (list, dict)):
                mods[key].update({kn: lst_get(kv) for kn, kv in value.items()})
            else:
                mods[key] = value.get()
        out = deepcopy(SettingHandler.default_settings)
        out.update(mods)
        return out

    def change_settings(self, options):
        for (key, value) in options.items():
            setattr(Sett, key, value)
        # SettingHandler.default_settings.update(options)


def lst_get(key_value):
    if isinstance(key_value, list):
        return [v.get() for v in key_value]
    return key_value.get()


def get_ref(name):
    try:
        value = SettingHandler.default_settings[name].copy()
    except AttributeError:
        value = SettingHandler.default_settings[name]
    if isinstance(value, dict):
        for (kn, kv) in value.items():
            dvar = get_tkvar(kv)
            value.update({kn: dvar})
        var = value
    else:
        var = get_tkvar(value)
    return var


def get_tkvar(value):
    if isinstance(value, bool):
        var = tk.BooleanVar(value=value)
    elif isinstance(value, str):
        var = tk.StringVar(value=value)
    elif isinstance(value, int):
        var = tk.IntVar(value=value)
    elif isinstance(value, float):
        var = tk.DoubleVar(value=value)
    elif isinstance(value, list):
        var = tk.StringVar(value=', '.join(value))
    else:
        var = value
    return var


def check_switch(*checks):
    for func in checks:
        func()
