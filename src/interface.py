# -*- coding: utf-8 -*-
"""
LAM-module for the creation of graphical user interface.

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""
# Standard libraries
import tkinter as tk
from tkinter import filedialog
from copy import deepcopy
from itertools import chain
# Other packages
import pathlib as pl
import pandas as pd
# LAM modules
from run import main_catch_exit
from settings import settings as Sett
import vector_loop

LAM_logger = None


class base_GUI(tk.Toplevel):
    """Container for the most important settings of the GUI."""

    def __init__(self, master=None):
        master.title("LAM-0.3.1")
        self.master = master
        self.master.grab_set()
        self.master.bind('<Escape>', self.func_destroy)
        self.master.bind('<Return>', self.run_button)
        # Fetch settings and transform to tkinter variables:
        self.handle = SettingHandler()

        master.option_add("*Font", "TkDefaultFont 8")

        # create all of the main containers
        self.topf = tk.Frame(self.master, pady=2)
        self.midf = tk.Frame(self.master, pady=2)
        self.leftf = tk.Frame(self.master, bd=2, relief='groove')
        self.rightf = tk.Frame(self.master, bd=2, relief='groove')
        self.distf = tk.Frame(self.master, bd=2, relief='groove')
        self.bottomf = tk.Frame(self.master)

        # LAYOUT:
        self.topf.grid(row=0, rowspan=2, columnspan=6, sticky="new")
        self.midf.grid(row=2, rowspan=3, columnspan=6, sticky="new")
        self.leftf.grid(row=5, column=0, columnspan=3, rowspan=4, sticky="new")
        self.rightf.grid(row=5, column=3, columnspan=3, rowspan=9,
                         sticky="new")
        self.distf.grid(row=12, rowspan=8, columnspan=6, sticky="new")
        self.bottomf.grid(row=18, rowspan=2, columnspan=6, sticky="new",
                          pady=(2, 2))
        col_count, row_count = self.master.grid_size()
        for col in range(col_count):
            self.master.grid_columnconfigure(col, minsize=45)
        for row in range(row_count):
            self.master.grid_rowconfigure(row, minsize=32)

        # TOP FRAME / WORK DIRECTORY
        tk.Label(self.topf, text=self.handle('workdir').get(),
                 textvariable=self.handle('workdir'), bd=2, bg='white',
                 relief='sunken').grid(row=0, column=1, columnspan=7)
        tk.Button(self.topf, text="Directory", command=self.browse
                  ).grid(row=0, column=0)

        # Detected groups and channels
        self.det_chans = tk.StringVar(value="Detected channels:")
        self.det_groups = tk.StringVar(value="Detected groups:")
        tk.Button(self.topf, text="Detect", command=self.detect_chans
                  ).grid(row=1, column=0)
        tk.Label(self.topf, text=self.det_groups.get(),
                 textvariable=self.det_groups
                 ).grid(row=1, column=1, columnspan=8)
        tk.Label(self.topf, text=self.det_chans.get(),
                 textvariable=self.det_chans
                 ).grid(row=2, column=1, columnspan=8)

        # MIDDLE FRAME / PRIMARY SETTINGS BOX
        stl = {1: 'groove', 2: 'lightgrey', 3: ('TkDefaultFont 8 bold')}
        tk.Checkbutton(self.midf, text="Process", bg=stl[2],
                       variable=self.handle('process_samples'), relief=stl[1],
                       font=stl[3], bd=4, command=self.process_check
                       ).grid(row=0, column=0, columnspan=1, padx=(2, 2))
        tk.Checkbutton(self.midf, text="Count  ",
                       variable=self.handle('process_counts'), relief=stl[1],
                       bd=4, bg=stl[2], font=stl[3], command=self.count_check
                       ).grid(row=0, column=1, columnspan=1, padx=(2, 2))
        tk.Checkbutton(self.midf, text="Distance", bg=stl[2],
                       variable=self.handle('process_dists'), relief=stl[1],
                       bd=4, font=stl[3], command=self.distance_check
                       ).grid(row=0, column=2, columnspan=1, padx=(2, 2))
        tk.Checkbutton(self.midf, text="Plots   ",
                       variable=self.handle('Create_Plots'), relief=stl[1],
                       bd=4, bg=stl[2], font=stl[3], command=self.plot_check
                       ).grid(row=0, column=3, columnspan=1, padx=(2, 2))
        tk.Checkbutton(self.midf, text="Stats   ",
                       variable=self.handle('statistics'), relief=stl[1], bd=4,
                       bg=stl[2], font=stl[3], command=self.stat_check
                       ).grid(row=0, column=4, columnspan=1, padx=(2, 2))

        # Projection, Measurement point, widths, borders & file header settings
        proj = tk.Checkbutton(self.midf, text="Project", relief='groove', bd=3,
                              variable=self.handle('project'))
        mp = tk.Checkbutton(self.midf, text="Use MP ", relief='groove',
                            variable=self.handle('useMP'), bd=3,
                            command=self.mp_check)
        width = tk.Checkbutton(self.midf, text="Widths",
                               variable=self.handle('measure_width'),
                               relief='groove', bd=3,
                               command=self.width_check)
        border = tk.Checkbutton(self.midf, text="Borders", bd=3,
                                relief='groove',
                                variable=self.handle('border_detection'),
                                command=self.border_check)
        lmp = tk.Label(self.midf, text='MP label:', bd=1)
        mpin = tk.Entry(self.midf, text=self.handle('MPname').get(), bd=2,
                        textvariable=self.handle('MPname'))
        self.lhead = tk.Label(self.midf, bd=1, text='Data header row:\n(0..n)')
        self.headin = tk.Entry(self.midf, text=self.handle('header_row').get(),
                               bd=2, textvariable=self.handle('header_row'))
        proj.grid(row=1, column=0, columnspan=1, padx=(2, 2))
        mp.grid(row=1, column=1, columnspan=1, padx=(2, 2))
        width.grid(row=2, column=0, columnspan=1, padx=(2, 2))
        border.grid(row=2, column=1, columnspan=1, padx=(2, 2))
        lmp.grid(row=1, column=2)
        mpin.grid(row=1, column=3, columnspan=3)
        self.lhead.grid(row=2, column=2, columnspan=1)
        self.headin.grid(row=2, column=3, columnspan=2)

        # BOTTOM BUTTONS
        style = ("TkDefaultFont 10 bold")
        tk.Checkbutton(self.bottomf, text="Redirect stdout",
                       variable=self.handle('non_stdout'), relief='groove',
                       bd=1, command=self.redirect_stdout
                       ).grid(row=0, column=4, columnspan=4, sticky='n')
        self.run_b = tk.Button(self.bottomf, font=style,
                               text='Run\n<Enter>', command=self.run_button)
        quit_b = tk.Button(self.bottomf, font=style, text="Quit",
                           command=self.func_destroy)
        wins = [OtherWin, PlotWin, StatWin]
        add_b = tk.Button(self.bottomf, font=style, text="Other",
                          command=lambda x=wins[0]: self.open_win(x))
        self.plot_b = tk.Button(self.bottomf, font=style, text="Plots",
                                command=lambda x=wins[1]: self.open_win(x))
        self.stats_b = tk.Button(self.bottomf, font=style, text="Stats",
                                 command=lambda x=wins[2]: self.open_win(x))
        self.vector_b = tk.Button(self.bottomf, font=style,
                                  text="Create\nvectors",
                                  command=self.vector_creation)
        # Style
        self.run_b.configure(height=2, width=7, bg='lightgreen', fg="black")
        quit_b.configure(height=1, width=5, fg="red")
        add_b.configure(height=2, width=7)
        self.plot_b.configure(height=2, width=7)
        self.stats_b.configure(height=2, width=7)
        self.vector_b.configure(height=2, width=6, bg='#ffe9ba', fg="black")
        # Grid
        self.run_b.grid(row=1, column=4, columnspan=1, padx=(5, 15),
                        sticky='ne')
        quit_b.grid(row=1, column=5, sticky='nes')
        add_b.grid(row=1, column=0, columnspan=1, padx=(0, 5), sticky='nw')
        self.plot_b.grid(row=1, column=1, columnspan=1, padx=(0, 5),
                         sticky='nw')
        self.stats_b.grid(row=1, column=2, columnspan=1, sticky='nw')
        self.vector_b.grid(row=1, column=3, columnspan=1, padx=(55, 0),
                           sticky='ne')

        # RIGHT FRAME / PLOTTING
        # header
        lbl2 = tk.Label(self.rightf, text='Plotting:', bd=2,
                        font=('Arial', 9, 'bold'))
        lbl2.grid(row=0, column=0)

        # create checkboxes
        tk.Checkbutton(self.rightf, text="Channels",
                       variable=self.handle('Create_Channel_Plots')
                       ).grid(row=1, column=0, sticky='w', pady=(2, 0))
        tk.Checkbutton(self.rightf, text="Additional Data",
                       variable=self.handle('Create_AddData_Plots')
                       ).grid(row=2, column=0, sticky='w')
        tk.Checkbutton(self.rightf, text="Channel Matrix",
                       variable=self.handle('Create_Channel_PairPlots')
                       ).grid(row=3, column=0, sticky='w')
        tk.Checkbutton(self.rightf, text="Heatmaps",
                       variable=self.handle('Create_Heatmaps')
                       ).grid(row=1, column=1, sticky='w', pady=(2, 0))
        tk.Checkbutton(self.rightf, text="Distributions",
                       variable=self.handle('Create_Distribution_Plots')
                       ).grid(row=4, column=0, sticky='w')
        tk.Checkbutton(self.rightf, text="Channel VS. Add.",
                       variable=self.handle('Create_ChanVSAdd_Plots')
                       ).grid(row=7, column=0, sticky='w', pady=(10, 0))
        tk.Checkbutton(self.rightf, text="Add. VS. Add.",
                       variable=self.handle('Create_AddVSAdd_Plots')
                       ).grid(row=8, column=0, sticky='ws', pady=(2, 30))
        self.statc = tk.Checkbutton(self.rightf, text="Statistics",
                                    variable=self.handle(
                                        'Create_Statistics_Plots'))
        self.statc.grid(row=3, column=1, sticky='w')
        self.clustc = tk.Checkbutton(self.rightf, text="Clusters",
                                     variable=self.handle(
                                         'Create_Cluster_Plots'))
        self.clustc.grid(row=2, column=1, sticky='w')
        self.borderc = tk.Checkbutton(self.rightf, text="Borders",
                                      variable=self.handle(
                                          'Create_Border_Plots'))
        self.borderc.grid(row=4, column=1, sticky='w')
        self.widthc = tk.Checkbutton(self.rightf, text="Widths",
                                     variable=self.handle('plot_width'))
        self.widthc.grid(row=5, column=0, sticky='w')

        # LEFT FRAME (UP) / VECTOR CREATION
        # header
        self.lbl3 = tk.Label(self.leftf, text='Vector:', bd=2,
                             font=('TkDefaultFont 9 bold'))
        self.lbl3.grid(row=0, column=0)

        # vector type radio buttons
        self.vbut1 = tk.Radiobutton(self.leftf, text="Skeleton", value=1,
                                    variable=self.handle('SkeletonVector'),
                                    command=self.switch_pages)
        self.vbut2 = tk.Radiobutton(self.leftf, text="Median", value=0,
                                    variable=self.handle('SkeletonVector'),
                                    command=self.switch_pages)
        self.vbut1.grid(row=1, column=0)
        self.vbut2.grid(row=1, column=1)

        # vector channel input
        lbl4 = tk.Label(self.leftf, text='Channel: ', bd=1)
        lbl4.grid(row=2, column=0)
        ch_in = tk.Entry(self.leftf, text=self.handle('vectChannel').get(),
                         textvariable=self.handle('vectChannel'))
        ch_in.grid(row=2, column=1, columnspan=1)

        # Bin number input
        lbl5 = tk.Label(self.leftf, text='Bin #: ', bd=1)
        lbl5.grid(row=3, column=0)
        bin_in = tk.Entry(self.leftf, text=self.handle('projBins').get(),
                          textvariable=self.handle('projBins'), bd=2)
        bin_in.grid(row=3, column=1, columnspan=1)

        # LEFT FRAME (LOWER) - VECTOR SETTINGS
        self.frames = {}
        self.vector_frame()

        # UPPER BOTTOM / DISTANCES
        # header
        tk.Label(self.distf, text='Distance Calculations:', bd=2,
                 font=('TkDefaultFont 9 bold')
                 ).grid(row=0, column=0, columnspan=6)

        # distance and cluster checkbuttons
        tk.Checkbutton(self.distf, text="Find clusters ",
                       variable=self.handle('Find_Clusters'),
                       command=self.cluster_check, bd=1, relief='raised'
                       ).grid(row=1, column=0, columnspan=2, sticky='n')
        tk.Checkbutton(self.distf, text="Find distances",
                       variable=self.handle('Find_Distances'),
                       command=self.nearest_dist_check, bd=1, relief='raised'
                       ).grid(row=1, column=2, columnspan=2, sticky='n')
        # Filtering
        test = any([bool(self.handle(v).get()) for v in ('inclusion',
                                                         'Cl_inclusion')])
        self.fltr_val = tk.BooleanVar(value=test)
        tk.Checkbutton(self.distf, text="Filter", relief='raised',
                       variable=self.fltr_val, bd=1, command=self.filter_check
                       ).grid(row=1, column=4, columnspan=2, sticky='n')
        # Add distance calculation's filter column name variable:
        col_in = tk.Entry(self.distf, text=self.handle('incl_col').get(),
                          textvariable=self.handle('incl_col'), bd=1)
        col_in.grid(row=1, column=6, columnspan=2, sticky='n', pady=(2, 0))
        cl_lbl = tk.Label(self.distf, text="Clusters:")
        cl_lbl.grid(row=2, column=0, columnspan=2)
        dist_lbl = tk.Label(self.distf, text="Cell Distances:")
        dist_lbl.grid(row=2, column=4, columnspan=2)
        # CLUSTERING settings
        cl_chanlbl = tk.Label(self.distf, text="Channels:")
        cl_ch_in = tk.Entry(self.distf, text=self.handle('Cluster_Channels'),
                            textvariable=self.handle('Cluster_Channels'), bd=2)
        cl_chanlbl.grid(row=3, column=0, columnspan=2)
        cl_ch_in.grid(row=3, column=2, columnspan=2)

        cl_distlbl = tk.Label(self.distf, text='Max Dist.:')
        cl_distlbl.grid(row=4, column=0, columnspan=2)
        cl_dist_in = tk.Entry(self.distf, text=self.handle('Cl_maxDist').get(),
                              textvariable=self.handle('Cl_maxDist'), bd=2)
        cl_dist_in.grid(row=4, column=2, columnspan=2)
        cl_minlbl = tk.Label(self.distf, text='Min cell #:')
        cl_minlbl.grid(row=5, column=0, columnspan=2)
        cl_min_in = tk.Entry(self.distf, text=self.handle('Cl_min').get(),
                             textvariable=self.handle('Cl_min'), bd=2)
        cl_min_in.grid(row=5, column=2, columnspan=2)

        cl_maxlbl = tk.Label(self.distf, text='Max cell #:')
        cl_maxlbl.grid(row=6, column=0, columnspan=2)
        cl_max_in = tk.Entry(self.distf, text=self.handle('Cl_max').get(),
                             textvariable=self.handle('Cl_max'), bd=2)
        cl_max_in.grid(row=6, column=2, columnspan=2)
        # Filtering
        cl_sizelbl = tk.Label(self.distf, text='Filter value:')
        cl_sizelbl.grid(row=7, column=0, columnspan=2)
        cl_size = tk.Entry(self.distf, text=self.handle('Cl_inclusion').get(),
                           textvariable=self.handle('Cl_inclusion'))
        cl_size.grid(row=7, column=2, columnspan=2)
        cl_but1 = tk.Radiobutton(self.distf, text="  keep greater  ",
                                 variable=self.handle('Cl_incl_type'),
                                 value='greater', indicatoron=0)
        cl_but2 = tk.Radiobutton(self.distf, text="  keep smaller  ",
                                 variable=self.handle('Cl_incl_type'),
                                 value='', indicatoron=0)
        cl_but1.grid(row=8, column=0, columnspan=2, sticky='nse')
        cl_but2.grid(row=8, column=2, columnspan=2, sticky='nsw')

        # DISTANCE settings
        d_chanlbl = tk.Label(self.distf, text="Channels:")
        d_chanlbl.grid(row=3, column=4, columnspan=2)
        d_chan = tk.Entry(self.distf, text=self.handle('Distance_Channels'),
                          textvariable=self.handle('Distance_Channels'), bd=2)
        d_chan.grid(row=3, column=6, columnspan=2)

        d_distlbl = tk.Label(self.distf, text='Max Dist.:')
        d_distlbl.grid(row=4, column=4, columnspan=2)
        d_dist_in = tk.Entry(self.distf, text=self.handle('maxDist').get(),
                             textvariable=self.handle('maxDist'), bd=2)
        d_dist_in.grid(row=4, column=6, columnspan=2)
        # Nearestdist target channel
        d_target = tk.Checkbutton(self.distf, text='Use target:',
                                  variable=self.handle('use_target'),
                                  command=self.target_check)
        d_target.grid(row=5, column=4, columnspan=2)
        self.d_trgt_in = tk.Entry(self.distf, bd=2,
                                  text=self.handle('target_chan').get(),
                                  textvariable=self.handle('target_chan'))
        self.d_trgt_in.grid(row=5, column=6, columnspan=2)
        # Filtering
        d_sizelbl = tk.Label(self.distf, text='Filter value:')
        d_sizelbl.grid(row=7, column=4, columnspan=2)
        d_size_in = tk.Entry(self.distf, text=self.handle('inclusion').get(),
                             textvariable=self.handle('inclusion'), bd=2)
        d_size_in.grid(row=7, column=6, columnspan=2)
        d_but1 = tk.Radiobutton(self.distf, text="  keep greater  ",
                                variable=self.handle('incl_type'),
                                value='greater', indicatoron=0)
        d_but2 = tk.Radiobutton(self.distf, text="  keep smaller  ",
                                variable=self.handle('incl_type'),
                                value='', indicatoron=0)
        d_but1.grid(row=8, column=4, columnspan=2, sticky='nse')
        d_but2.grid(row=8, column=6, columnspan=2, sticky='nsw')

        # Store created widgets for enabling/disabling
        self.wdgs = {'cluster': [cl_chanlbl, cl_ch_in, cl_distlbl, cl_dist_in,
                                 cl_minlbl, cl_min_in, cl_maxlbl, cl_max_in,
                                 cl_sizelbl, cl_size, cl_but1, cl_but2,
                                 cl_lbl],
                     'dist': [d_chanlbl, d_chan, d_distlbl, d_dist_in,
                              d_target, self.d_trgt_in, d_sizelbl, d_size_in,
                              d_but1, d_but2, dist_lbl],
                     'count': [bin_in, lbl5, lmp, mpin, mp, proj, width],
                     'fltr_cl': [cl_sizelbl, cl_size, cl_but1, cl_but2,
                                 col_in],
                     'fltr_dist': [d_sizelbl, d_size_in, d_but1, d_but2,
                                   col_in],
                     'process': [bin_in, lbl5, ch_in, lbl4], 'mp': [lmp, mpin]}

        # Disable / enable widgets
        self.process_check()
        self.count_check()
        self.distance_check()
        self.stat_check()
        self.redirect_stdout()

    def browse(self):
        """Allow input of path when browse-button is pressed."""
        filename = filedialog.askdirectory()
        self.handle('workdir').set(filename)
        self.detect_chans()

    def border_check(self):
        """Control border detection related settings."""
        if self.handle('Create_Plots').get():
            if self.handle('border_detection').get():
                self.borderc.configure(state='normal')
            else:
                self.borderc.configure(state='disable')
        else:
            self.handle('Create_Border_Plots').set(False)

    def cluster_check(self):
        """Relevant changes when cluster-setting is checked."""
        if not self.handle('Find_Clusters').get():
            configure('disable', self.wdgs['cluster'])
        else:
            configure('normal', self.wdgs['cluster'])
            self.filter_check()

    def count_check(self):
        """Relevant changes when count-setting is checked."""
        if not self.handle('process_counts').get():
            configure('disable', self.wdgs['count'])
        else:
            self.lhead.configure(state='normal')
            self.headin.configure(state='normal')
            configure('normal', self.wdgs['count'])
        check_switch(self.mp_check, self.width_check, self.border_check,
                     self.run_check)

    def distance_check(self):
        """Relevant changes when distances-setting is checked."""
        if not self.handle('process_dists').get():
            for widget in self.distf.winfo_children():
                widget.configure(state='disable')
        else:
            for widget in self.distf.winfo_children():
                if int(widget.grid_info()["row"]) in [0, 1, 2]:
                    widget.configure(state='normal')
            check_switch(self.cluster_check, self.nearest_dist_check,
                         self.filter_check)
        self.run_check()

    def filter_check(self):
        """Relevant changes when filtering by size is checked."""
        if not self.fltr_val.get():
            self.handle('Cl_inclusion').set(0)
            self.handle('inclusion').set(0)
            configure('disable', chain(self.wdgs['fltr_cl'],
                                       self.wdgs['fltr_dist']))
        else:
            if self.handle('Find_Distances').get():
                configure('normal', self.wdgs['fltr_dist'])
            if self.handle('Find_Clusters').get():
                configure('normal', self.wdgs['fltr_cl'])

    def mp_check(self):
        """Relevant changes when MP is in use or not."""
        if not self.handle('useMP').get():
            configure('disable', self.wdgs['mp'])
        else:
            configure('normal', self.wdgs['mp'])

    def nearest_dist_check(self):
        """Relevant changes when find distance-setting is checked."""
        if not self.handle('Find_Distances').get():
            configure('disable', self.wdgs['dist'])
        else:
            configure('normal', self.wdgs['dist'])
        check_switch(self.target_check, self.filter_check)

    def plot_check(self):
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

    def process_check(self):
        """Relevant changes when Process-setting is checked."""
        widgets = (wdg for wdg in self.leftf.winfo_children() if wdg not in
                   self.wdgs['process'])
        if not self.handle('process_samples').get():
            self.vector_b.configure(state='disable', bg='#d9cfbd')
            configure('disable', widgets)
            hidev = 'disable'
        else:
            self.vector_b.configure(state='normal', bg='#e3bf78')
            configure('normal', widgets)
            self.switch_pages()
            hidev = 'normal'
        if not self.handle('SkeletonVector').get():
            for widget in self.frames[MedianSettings].winfo_children():
                widget.configure(state=hidev)
        else:
            for widget in self.frames[SkelSettings].winfo_children():
                widget.configure(state=hidev)
        self.run_check()

    def run_check(self):
        """Determine if run button should be active."""
        sets = ['process_samples', 'process_counts', 'process_dists',
                'Create_Plots', 'statistics']
        if any([self.handle(v).get() for v in sets]):
            self.run_b.configure(state='normal', bg='lightgreen')
        else:
            self.run_b.configure(state='disable', bg='lightgrey')

    def stat_check(self):
        """Relevant changes when statistics-setting is checked."""
        if not self.handle('statistics').get():
            self.stats_b.configure(state='disable')
            self.statc.configure(state='disable')
            self.handle('Create_Statistics_Plots').set(False)
        else:
            self.stats_b.configure(state='normal')
            if self.handle('Create_Plots').get():
                self.statc.configure(state='normal')
        self.run_check()

    def target_check(self):
        """Relevant changes when target-setting is checked."""
        if not self.handle('use_target').get():
            self.d_trgt_in.configure(state='disable')
        else:
            self.d_trgt_in.configure(state='normal')

    def run_button(self, event=None):
        """Relevant changes when Run-button is pressed + run initialization."""
        # Get modified options
        options = self.handle.translate()
        options['workdir'] = pl.Path(options['workdir'])  # Transform workdir
        self.detect_chans()
        # If needed, change settings that have high risk of interfering
        if not options['process_counts']:
            ops = ('measure_width', 'useMP', 'project')
            options.update({k: False for k in ops})
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

    def show_vector_settings(self, name):
        """Change shown vector settings based on type."""
        for frame in self.frames.values():
            frame.grid_remove()
        frame = self.frames[name]
        frame.grid()

    def switch_pages(self):
        """Switch page of vector settings."""
        if not self.handle('SkeletonVector').get():
            self.show_vector_settings(MedianSettings)
        else:
            self.show_vector_settings(SkelSettings)

    def func_destroy(self, event=None):
        """Destroy GUI."""
        import logger as lg
        lg.log_Shutdown()
        if hasattr(self, 'stdout_win'):
            self.stdout_win.func_destroy()
        self.master.destroy()

    def open_win(self, window):
        """Open given settings window."""
        ops = window(self.master, self.handle.vars.check, self.handle.vars.ref)
        ops.window.wait_window()
        ind = ops.handle.vars.loc[ops.handle.vars.check].index
        self.handle.vars.loc[ind, :] = ops.handle.vars.loc[ind, :]

    def detect_chans(self):
        """Detect channels and groups found at current set path."""
        workdir = pl.Path(self.handle('workdir').get())
        det_chans, det_groups = set(), set()
        # Loop found sample directories
        for samplepath in [p for p in workdir.iterdir() if p.is_dir() and
                           'Analysis Data' not in p.name]:
            try:  # Get groups from folder names
                group = str(samplepath.name).split('_')[0]
                det_groups.add(group)
                # Loop through channels of found samples
                cpaths = [p for p in samplepath.iterdir() if p.is_dir()]
                for channelpath in cpaths:
                    channel = str(channelpath.name).split('_')[-2]
                    det_chans.add(channel)
            except (IndexError, TypeError):
                pass
        # Change text variables to contain new groups and channels
        if det_chans:
            msg = ', '.join(sorted(det_chans))
            chanstring = tk.StringVar(value=f"Detected channels: {msg}")
        else:
            chanstring = tk.StringVar(value='No detected channels!')
        if det_groups:
            msg = ', '.join(sorted(det_groups))
            grpstring = tk.StringVar(value=f"Detected groups: {msg}")
        else:
            grpstring = tk.StringVar(value='No detected groups!')
        # Set new text variables to be shown
        self.det_groups.set(grpstring.get())
        self.det_chans.set(chanstring.get())
        from settings import store
        store.samplegroups = list(det_groups)
        store.channels = [c for c in det_chans if
                          c.lower() != self.handle('MPname').get().lower()]

    def vector_creation(self):
        self.run_b.configure(state='disable')
        win = vector_loop.VectorWin(self.master, self.handle)
        win.window.wait_window()
        self.process_check()
        self.run_b.configure(state='normal')

    def vector_frame(self):
        """Create frame for vector creation settings."""
        for F in (SkelSettings, MedianSettings):
            frame = F(self.master, self, self.handle)
            self.frames[F] = frame
            frame.grid(row=8, column=0, columnspan=3, rowspan=5, sticky="new")
            frame.grid_remove()
        if self.handle('SkeletonVector').get():
            self.show_vector_settings(SkelSettings)
        else:
            self.show_vector_settings(MedianSettings)

    def width_check(self):
        """Enable width estimation related settings."""
        if self.handle('Create_Plots').get():
            if self.handle('measure_width').get():
                self.widthc.configure(state='normal')
            else:
                self.widthc.configure(state='disable')
        else:
            self.handle('plot_width').set(False)


class SkelSettings(tk.Frame):
    """Container for skeleton vector-related settings."""

    def __init__(self, parent, master, handle):
        tk.Frame.__init__(self, parent, bd=2, relief='groove')
        # Container label
        tk.Label(self, text='Vector Parameters:', bd=1, font='TkDefaultFont 10'
                 ).grid(row=0, column=0, columnspan=3, pady=(0, 3))
        # Container widgets
        tk.Label(self, text='Simplify tol.', bd=1).grid(row=1, column=0,
                                                        columnspan=1)
        tk.Entry(self, text=handle('simplifyTol').get(), bd=2,
                 textvariable=handle('simplifyTol')).grid(row=1, column=1)
        tk.Label(self, text='Resize', bd=1).grid(row=2, column=0, columnspan=1)
        tk.Entry(self, text=handle('SkeletonResize').get(), bd=2,
                 textvariable=handle('SkeletonResize')).grid(row=2, column=1)
        tk.Label(self, text='Find distance', bd=1).grid(row=3, column=0,
                                                        columnspan=1)
        tk.Entry(self, text=handle('find_dist').get(), bd=2,
                 textvariable=handle('find_dist')).grid(row=3, column=1)
        tk.Label(self, text='Dilation iter', bd=1).grid(row=4, column=0,
                                                        columnspan=1)
        tk.Entry(self, text=handle('BDiter').get(), bd=2,
                 textvariable=handle('BDiter')).grid(row=4, column=1)
        tk.Label(self, text='Smoothing', bd=1).grid(row=5, column=0,
                                                    columnspan=1, pady=(0, 3))
        tk.Entry(self, text=handle('SigmaGauss').get(), bd=2,
                 textvariable=handle('SigmaGauss')
                 ).grid(row=5, column=1, pady=(0, 3))


class MedianSettings(tk.Frame):
    """Container for median vector-related settings."""

    def __init__(self, parent, master, handle):
        # Container label
        tk.Frame.__init__(self, parent, bd=2, relief='groove')

        tk.Label(self, text='Vector Parameters:', bd=1
                 ).grid(row=0, column=0, columnspan=3, pady=(0, 3))
        # Container widgets
        tk.Label(self, text='Simplify tol.', bd=1).grid(row=1, column=0,
                                                        columnspan=1)
        tk.Entry(self, bd=2, text=handle('simplifyTol').get(),
                 textvariable=handle('simplifyTol')).grid(row=1, column=1)
        tk.Label(self, text='Median bins  ', bd=1
                 ).grid(row=2, column=0, columnspan=1, pady=(0, 63))
        tk.Entry(self, bd=2, text=handle('medianBins').get(),
                 textvariable=handle('medianBins')
                 ).grid(row=2, column=1, pady=(0, 63))


class OtherWin:
    """Container for Other-window settings."""

    def __init__(self, master, check, ref):
        self.handle = SettingHandler(check, ref)

        self.window = tk.Toplevel(master)
        self.window.grab_set()
        self.window.title("Additional Data Settings")
        self.window.bind('<Escape>', self.func_destroy)
        self.window.bind('<Return>', self.window.destroy)
        self.window.protocol("WM_DELETE_WINDOW", self.func_destroy)
        # Frames
        self.frame = tk.Frame(self.window)
        self.dframe = tk.Frame(self.window, relief='groove')
        self.bframe = tk.Frame(self.window)
        self.frame.grid(row=0, rowspan=11, columnspan=9, sticky="new")
        self.dframe.grid(row=8, rowspan=5, columnspan=9, sticky="new")
        self.bframe.grid(row=13, rowspan=2, columnspan=9, sticky="ne")
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
        tk.Label(self.frame, text='Column label', bd=1).grid(row=0, column=0,
                                                             columnspan=2)
        tk.Entry(self.frame, text=self.insert[0].get(),
                 textvariable=self.insert[0], bd=2,
                 ).grid(row=1, column=0, columnspan=2, pady=(0, 10))

        # Entry for data file name:
        tk.Label(self.frame, text='csv-file', bd=1).grid(row=0, column=3,
                                                         columnspan=2)
        tk.Entry(self.frame, text=self.insert[1].get(),
                 textvariable=self.insert[1], bd=2,
                 ).grid(row=1, column=3, columnspan=2, pady=(0, 10))

        # Unit of data type:
        tk.Label(self.frame, text='Unit', bd=1).grid(row=0, column=5,
                                                     columnspan=2)
        tk.Entry(self.frame, text=self.insert[2].get(),
                 textvariable=self.insert[2], bd=2,
                 ).grid(row=1, column=5, columnspan=2, pady=(0, 10))

        # BUTTONS
        self.add_b = tk.Button(self.frame, text='Add', command=self.add_data)
        self.add_b.configure(bg='lightgreen', fg="darkgreen")
        self.save_b = tk.Button(self.bframe, text='Save & Return\n<Enter>',
                                command=self.window.destroy)
        self.save_b.configure(height=2, width=12, bg='lightgreen',
                              fg="darkgreen")
        self.exit_b = tk.Button(self.bframe, text="Return",
                                command=self.func_destroy)
        self.exit_b.configure(height=1, width=5, fg="red")

        self.add_b.grid(row=0, column=7, rowspan=2, padx=(5, 10), pady=(0, 10))
        self.save_b.grid(row=11, column=5, rowspan=2, columnspan=2,
                         padx=(0, 9))
        self.exit_b.grid(row=11, column=7)

        # additional data labels and removal buttons:
        self.rwn = len(self.handle('AddData'))
        self.btns = []

        for i, (key, vals) in enumerate(self.handle('AddData').items()):
            row = i+2
            tk.Label(self.frame, text=key, bd=2, bg='lightgrey',
                     relief='groove').grid(row=row, column=0, columnspan=2)
            tk.Label(self.frame, text=vals[0].get(), bd=2, bg='lightgrey',
                     relief='groove').grid(row=row, column=3, columnspan=2)
            tk.Label(self.frame, text=vals[1].get(), bd=2, bg='lightgrey',
                     relief='groove').grid(row=row, column=5, columnspan=2)
            self.btns.append(tk.Button(self.frame, text='x',
                                       font=('Arial', 10), relief='raised',
                                       command=lambda i=i: self.rmv_data(i)))
            self.btns[i].grid(row=row, column=7, sticky='w')

        # additional data ID-replacement
        tk.Checkbutton(self.dframe, text="Replace file ID",
                       variable=self.handle('replaceID'), relief='groove',
                       bd=4, command=self.replace_check
                       ).grid(row=0, column=0, columnspan=4)
        tk.Label(self.dframe, text='File descriptor:', bd=1
                 ).grid(row=1, column=0, columnspan=3)
        tk.Label(self.dframe, text='Change to:', bd=1
                 ).grid(row=1, column=3, columnspan=3)
        for i, (key, val) in enumerate(self.handle('channelID').items()):
            row = i+2
            file_id = tk.StringVar(value=key)
            tk.Entry(self.dframe, text=file_id.get(), textvariable=file_id,
                     bd=2,).grid(row=row, column=0, columnspan=3)
            tk.Entry(self.dframe, text=val.get(), textvariable=val, bd=2
                     ).grid(row=row, column=3, columnspan=3)
        self.replace_check()

    def replace_check(self):
        """Change relevant settings when replaceID is checked."""
        if not self.handle('replaceID').get():
            for child in self.dframe.winfo_children():
                if isinstance(child, tk.Entry):
                    child.configure(state='disable')
        else:
            for child in self.dframe.winfo_children():
                if isinstance(child, tk.Entry):
                    child.configure(state='normal')

    def add_data(self):
        """Addition of data input to the additional data table."""
        if self.insert[0].get() not in self.handle('AddData').keys():
            row = self.rwn+2
            tk.Label(self.frame, text=self.insert[0].get(), bd=2,
                     bg='lightgrey', relief='groove').grid(row=row, column=0,
                                                           columnspan=2)
            tk.Label(self.frame, text=self.insert[1].get(), bd=2,
                     bg='lightgrey', relief='groove').grid(row=row, column=3,
                                                           columnspan=2)
            tk.Label(self.frame, text=self.insert[2].get(), bd=2,
                     bg='lightgrey', relief='groove').grid(row=row, column=5,
                                                           columnspan=2)
            self.btns.append(tk.Button(self.frame, text='x', relief='raised',
                                       command=lambda i=self.rwn:
                                       self.rmv_data(i)))
            self.btns[self.rwn].grid(row=row, column=7, sticky='w')
            var = [get_tkvar(self.insert[1]), get_tkvar(self.insert[2])]
            self.handle('AddData').update({self.insert[0].get(): var})
            self.rwn = self.rwn + 1
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


class PlotWin:
    """Container for Other-window settings."""

    def __init__(self, master, check, ref):
        self.handle = SettingHandler(check, ref)
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
        self.bframe = tk.Frame(self.window)
        self.bframe.grid(row=13, column=0, rowspan=2, columnspan=4, pady=5,
                         sticky="ne")
        # General settings:
        # Outliers
        tk.Label(self.frame, text="General Settings:",
                 font='TkDefaultFont 10 bold'
                 ).grid(row=0, column=0, columnspan=2)
        tk.Checkbutton(self.frame, text="Drop outliers", bd=1, relief='groove',
                       variable=self.handle('Drop_Outliers'),
                       command=self.drop_check
                       ).grid(row=1, column=0, columnspan=2)
        tk.Label(self.frame, text='Std dev.:').grid(row=2, column=0,
                                                    columnspan=2)
        self.stdin = tk.Entry(self.frame, text=self.handle('dropSTD').get(),
                              textvariable=self.handle('dropSTD'), bd=2)
        self.stdin.grid(row=2, column=2, columnspan=2)
        self.drop_check()
        # Pairplot jitter
        tk.Checkbutton(self.frame, text="Pair plot jitter",
                       variable=self.handle('plot_jitter')
                       ).grid(row=3, column=0, columnspan=2)
        # Save format
        tk.Label(self.frame, text='Save format:'
                 ).grid(row=4, column=0, columnspan=1)
        tk.Entry(self.frame, bd=2, text=self.handle('saveformat'),
                 textvariable=self.handle('saveformat')
                 ).grid(row=4, column=1, columnspan=2)
        comment = "Supported formats:\n\
        eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba,\nsvg, svgz, tif, tiff"
        tk.Label(self.frame, text=comment, fg='dimgrey'
                 ).grid(row=5, column=0, columnspan=4, pady=(0, 10))
        # versus
        tk.Label(self.frame, text='Versus plots:'
                 ).grid(row=6, column=0, columnspan=3)
        tk.Label(self.frame, text='Plotted channels:'
                 ).grid(row=7, column=0, columnspan=2)
        tk.Entry(self.frame, bd=2, text=self.handle('vs_channels'),
                 textvariable=self.handle('vs_channels')
                 ).grid(row=7, column=2, columnspan=2)
        tk.Entry(self.frame, bd=2, text=self.handle('vs_adds'),
                 textvariable=self.handle('vs_adds')
                 ).grid(row=8, column=2, columnspan=2)
        tk.Label(self.frame, text='Plotted add. data:'
                 ).grid(row=8, column=0, columnspan=2)
        # Statistics settings:
        tk.Label(self.statframe, text='Statistical Plotting:'
                 ).grid(row=0, column=0, columnspan=3)
        tk.Checkbutton(self.statframe, text="Sign. color",
                       variable=self.handle('fill')
                       ).grid(row=1, column=2, columnspan=2)
        tk.Checkbutton(self.statframe, text="Neg. log2",
                       variable=self.handle('negLog2'),
                       command=self.sign_check
                       ).grid(row=2, column=0, columnspan=1)
        self.starc = tk.Checkbutton(self.statframe, text="Sign. stars",
                                    variable=self.handle('stars'),
                                    command=self.sign_check)
        self.ylimlbl = tk.Label(self.statframe, text='y-limit:')
        self.ylim_in = tk.Entry(self.statframe, text=self.handle('ylim').get(),
                                textvariable=self.handle('ylim'), bd=2)
        self.starc.grid(row=1, column=0, columnspan=2)
        self.ylimlbl.grid(row=2, column=1, columnspan=1)
        self.ylim_in.grid(row=2, column=2, columnspan=2)

        self.sign_check()
        # BUTTONS
        stl = 'TkDefaultFont 10 bold'
        self.save_b = tk.Button(self.bframe, text='Save & Return\n<Enter>',
                                font=stl, command=self.window.destroy)
        self.save_b.configure(height=2, width=12, bg='lightgreen', fg="black")
        self.save_b.grid(row=0, column=2, rowspan=2, columnspan=2, padx=(0, 9))
        self.exit_b = tk.Button(self.bframe, text="Return", font=stl,
                                command=self.func_destroy)
        self.exit_b.configure(height=1, width=5, fg="red")
        self.exit_b.grid(row=0, column=4)

    def drop_check(self):
        """Relevant changes when dropping of outliers is selected."""
        if not self.handle('Drop_Outliers').get():
            self.stdin.configure(state='disable')
        else:
            self.stdin.configure(state='normal')

    def sign_check(self):
        """Relevant changes when neglog2-setting is selected."""
        if not self.handle('negLog2').get():
            self.starc.configure(state='normal')
            self.ylimlbl.configure(state='disable')
            self.ylim_in.configure(state='disable')
        else:
            self.handle('stars').set(False)
            self.starc.configure(state='disable')
            self.ylimlbl.configure(state='normal')
            self.ylim_in.configure(state='normal')

    def func_destroy(self, event=None):
        """Destroy window without saving."""
        self.window.destroy()
        self.handle.vars.loc[:, 'check'] = False


class StatWin():
    """Container for statistics-window settings."""

    def __init__(self, master, check, ref):
        self.handle = SettingHandler(check, ref)
        self.window = tk.Toplevel(master)
        self.window.grab_set()
        self.window.title("Statistics Settings")
        self.window.bind('<Escape>', self.func_destroy)
        self.window.bind('<Return>', self.window.destroy)
        self.window.protocol("WM_DELETE_WINDOW", self.func_destroy)
        self.frame = tk.Frame(self.window)
        self.frame.grid(row=0, rowspan=4, columnspan=4, sticky="new")
        self.wframe = tk.Frame(self.window, bd=1, relief='groove')
        self.wframe.grid(row=5, rowspan=2, columnspan=4, sticky="n")
        self.bframe = tk.Frame(self.window)
        self.bframe.grid(row=7, rowspan=2, columnspan=4, sticky="ne")
        # Control group
        tk.Label(self.frame, text='Control Group:').grid(row=0, column=0,
                                                         columnspan=2, pady=4)
        tk.Entry(self.frame, text=self.handle('cntrlGroup').get(),
                 textvariable=self.handle('cntrlGroup'), bd=2
                 ).grid(row=0, column=2, columnspan=2, pady=4)
        # Statistic types
        tk.Checkbutton(self.frame, text="Total Statistics", relief='groove',
                       variable=self.handle('stat_total'), bd=1
                       ).grid(row=1, column=0, columnspan=2, pady=4)
        tk.Checkbutton(self.frame, text="Group vs Group", relief='groove',
                       variable=self.handle('stat_versus'), bd=1
                       ).grid(row=1, column=2, columnspan=2, pady=4)
        # Alpha
        tk.Label(self.frame, text='Alpha:').grid(row=2, column=0, columnspan=2,
                                                 pady=4)
        tk.Entry(self.frame, text=self.handle('alpha').get(),
                 textvariable=self.handle('alpha'), bd=2,
                 ).grid(row=2, column=2, columnspan=2, pady=4)
        # Stat window
        tk.Checkbutton(self.frame, text="Windowed statistics",
                       variable=self.handle('windowed'), relief='raised', bd=1,
                       command=self.window_check
                       ).grid(row=3, column=0, columnspan=3, pady=(10, 0))
        tk.Label(self.wframe, text='Trailing window:'
                 ).grid(row=0, column=0, columnspan=2, pady=1)
        tk.Entry(self.wframe, text=self.handle('trail').get(),
                 textvariable=self.handle('trail'), bd=2,
                 ).grid(row=0, column=2, columnspan=2, pady=1)
        tk.Label(self.wframe, text='Leading window:'
                 ).grid(row=1, column=0, columnspan=2, pady=(1, 5))
        tk.Entry(self.wframe, text=self.handle('lead').get(),
                 textvariable=self.handle('lead'), bd=2,
                 ).grid(row=1, column=2, columnspan=2, pady=(1, 5))
        self.window_check()
        # Buttons
        self.save_b = tk.Button(self.bframe, text='Save & Return\n<Enter>',
                                font=('TkDefaultFont 10 bold'),
                                command=self.window.destroy)
        self.save_b.configure(height=2, width=12, bg='lightgreen',
                              fg="darkgreen")
        self.exit_b = tk.Button(self.bframe, text="Return",
                                font=('TkDefaultFont 9 bold'),
                                command=self.func_destroy)
        self.exit_b.configure(height=1, width=5, fg="red")
        self.save_b.grid(row=0, column=2, rowspan=2, columnspan=2,
                         padx=(0, 10))
        self.exit_b.grid(row=0, column=4)

    def window_check(self):
        """Relevant changes when windowed statistics is selected."""
        if not self.handle('windowed').get():
            for widget in self.wframe.winfo_children():
                widget.configure(state='disable')
        else:
            for widget in self.wframe.winfo_children():
                widget.configure(state='normal')

    def func_destroy(self, event=None):
        """Destroy stats-window."""
        self.window.destroy()
        self.handle.vars.loc[:, 'check'] = False


class SettingHandler:
    """Store and transform settings between tk variables and original."""
    # Fetch defaults from settings.py (after parsing cmdl arguments)
    default_settings = {k: v for k, v in vars(Sett).items() if "__" not in k}
    setting_names = sorted(default_settings.keys())

    def __init__(self, check=False, refs=None):
        self.vars = pd.DataFrame(index=SettingHandler.setting_names)
        self.vars = self.vars.assign(check=check, ref=refs)

    def __call__(self, variable_name):
        """Transform to tk variable and get reference."""
        if self.vars.at[variable_name, 'check']:
            return self.vars.at[variable_name, 'ref']
        vref = get_ref(variable_name)
        self.vars.loc[variable_name, :] = [True, vref]
        return self.vars.at[variable_name, 'ref']

    def translate(self):
        """Translate tk variables to original format."""
        mods = self.vars.loc[self.vars.check, 'ref'].to_dict()
        for key, value in mods.items():
            if key in ('Cluster_Channels', 'Distance_Channels', 'vs_channels',
                       'vs_adds'):
                var = mods[key].get().split(',')
                mods[key] = [v.strip() for v in var]
            elif isinstance(value, dict):
                mods[key].update({kn: lst_get(kv) for kn, kv in value.items()})
            else:
                mods[key] = value.get()
        out = deepcopy(SettingHandler.default_settings)
        out.update(mods)
        return out

    def change_settings(self, options):
        """Change run-settings based on given dict variables."""
        if Sett.border_channel is Sett.vectChannel and (options['vectChannel']
                                                        != Sett.vectChannel):
            rename_scoring_vars(options)
        for (key, value) in options.items():
            setattr(Sett, key, value)


def lst_get(key_value):
    """Get tk variables from list or list of lists."""
    if isinstance(key_value, list):
        if len(key_value) == 1:
            try:
                return key_value.get().split(', ')
            except AttributeError:
                print(key_value)
                return key_value
        return [v.get() for v in key_value]
    return key_value.get()


def get_ref(name):
    """Get copy of a original setting."""
    try:
        value = SettingHandler.default_settings[name].copy()
    except AttributeError:
        value = SettingHandler.default_settings[name]
    if isinstance(value, dict):
        for (kname, kval) in value.items():
            if isinstance(kval, str):
                dvar = tk.StringVar(value=kval)
            else:
                dvar = [tk.StringVar(value=v) for v in kval]
            value.update({kname: dvar})
        var = value
    else:
        var = get_tkvar(value)
    return var


def get_tkvar(value):
    """Transform a variable into tk variable."""
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
    """Perform checks for enabling/disabling GUI elements."""
    for func in checks:
        func()


def configure(state, widgets):
    """Set state for all widgets in list."""
    for widget in widgets:
        widget.configure(state=state)

def rename_scoring_vars(options):
    keys = [k for k in options['scoring_vars'].keys() if Sett.border_channel
            in k]
    for key in keys:
        nkey = key.replace(Sett.border_channel, options['vectChannel'])
        options['scoring_vars'][nkey] = options['scoring_vars'].pop(key)
