# -*- coding: utf-8 -*-
"""
Create window for creating vectors for all samples in loops with different
settings.

Created on Tue Sep 15 11:35:28 2020
@author: Arto I. Viitanen
"""
import pathlib as pl
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkFont

# LAM imports
from src.settings import Settings as Sett
import src.system as system
import src.process as process
import src.plotfuncs as pfunc


class VectorWin:
    """Open window for vector creation in loops."""
    keys = ['workdir', 'vectChannel', 'project', 'projBins', 'SkeletonVector', 'SkeletonResize', 'find_dist', 'BDiter',
            'SigmaGauss', 'simplifyTol', 'medianBins', 'process_samples']
    head = ('sample', 'group', 'path')

    def __init__(self, master, handle):
        # The tkinter selection coloring is broken, these fix it:
        ttk.Style().map("Treeview", foreground=fixed_map("foreground"), background=fixed_map("background"))
        ttk.Style().map('Treeview', background=[('selected', 'lightgreen')], foreground=[('selected', 'darkgreen')])

        # Create window
        self.window = tk.Toplevel()
        self.window.lift()
        self.window.title("Vector Creation")
        self.window.protocol("WM_DELETE_WINDOW", self.func_destroy)
        self.window.bind('<Return>', self.keep_vectors)

        # Assign variables
        self.handle = handle
        self.variables = handle.vars.loc[VectorWin.keys, :]
        self.workdir = pl.Path(self.variables.at['workdir', 'ref'].get())
        self.samples = [p for p in self.workdir.iterdir() if p.is_dir() and 'Analysis Data' not in p.name]
        self.sample_vars = [(p.name, str(p.name).split('_')[0], str(p)) for p in self.samples]
        self.tree = None

        # Create widgets
        self._setup()
        self._build()

    def _setup(self):
        """Create window elements."""
        # CREATE TREEVIEW
        self.tree = ttk.Treeview(self.window, columns=VectorWin.head, show="headings")
        self.tree.grid(column=0, row=0, rowspan=25, columnspan=3, sticky='nsw')
        # scroll bars
        vsb = ttk.Scrollbar(self.window, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.window, orient="horizontal", command=self.tree.xview)
        vsb.grid(column=3, row=0, rowspan=25, sticky='ns')
        hsb.grid(column=0, row=27, columnspan=3, sticky='ew')

        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set, height=25)

        # BUTTONS
        style = 'TkDefaultFont 9 bold'
        # Loop
        run_b = tk.Button(self.window, text='Loop', font=style, command=self.creation_loop)
        run_b.configure(height=2, width=7, bg='lightgreen', fg="black")
        run_b.grid(row=29, column=1, columnspan=1, sticky='n')
        # Done
        quit_b = tk.Button(self.window, text="Done", command=self._done, font=style,)
        quit_b.configure(height=1, width=5, fg="red")
        quit_b.grid(row=29, column=3, sticky='ne')
        # Keep
        keep_b = tk.Button(self.window, text="Keep\n<Enter>", font=style, command=self.keep_vectors)
        keep_b.configure(height=2, width=7, fg="green")
        keep_b.grid(row=29, column=2, sticky='nw')
        # Help
        help_b = tk.Button(self.window, text='?', font=style, command=self.open_help)
        help_b.configure(height=1, width=2, bd=3)
        help_b.grid(row=29, column=0, columnspan=1, sticky='nw')

    def _build(self):
        """Assign sample information in to treeview."""
        headers = VectorWin.head
        for col in headers:
            self.tree.heading(col, text=col.title())
            # adjust the column's width to the header string
            self.tree.column(col, width=tkFont.Font().measure(col.title()))

        for item in self.sample_vars:
            self.tree.insert('', 'end', values=item)
            # adjust column's width if necessary to fit each value
            for ind, val in enumerate(item):
                col_w = tkFont.Font().measure(val)
                if self.tree.column(headers[ind], width=None) < col_w:
                    self.tree.column(headers[ind], width=col_w)

    def _done(self):
        """Exit window and set vector creation-setting off."""
        self.variables.at['process_samples', 'ref'].set(False)
        self.window.destroy()

    def func_destroy(self):
        """Exit window."""
        self.window.destroy()

    def creation_loop(self):
        """Get settings from main window and perform vector creation."""
        from src.interface import change_settings

        # Translate tkinter variables of main window
        options = self.handle.translate()
        options['workdir'] = pl.Path(options['workdir'])
        # Adjust settings with the translated variables
        change_settings(options)
        # Get necessary paths for vector creation
        PATHS = system.start(test_vectors=False)
        process.check_resize_step(Sett.SkeletonResize)  # Test resize setting
        print_settings()  # Print used creation settings
        # Loop all samples that don't have a valid vector
        for sample in self.sample_vars:
            path = Sett.workdir.joinpath(sample[0])  # Sample's data path
            # Collect sample data:
            sample = process.GetSample(path, PATHS)
            print("{}  ...".format(sample.name))
            sample.vect_data = sample.get_vect_data(Sett.vectChannel)
            if sample.vect_data is None:
                continue
            # Creation of vector for projection
            if Sett.SkeletonVector:
                sample.create_skeleton()
            else:
                sample.create_median()
        print("Creation loop done. Select samples to keep.\n")
        # Creation of vector plot of all samples
        sample_dirs = [p for p in PATHS.samplesdir.iterdir() if p.is_dir()]
        pfunc.create_vector_plots(Sett.workdir, PATHS.samplesdir, sample_dirs)

    def keep_vectors(self):
        """Remove selected samples from creation loop and refresh table."""
        # Find selected samples
        valids = self.tree.selection()
        valid_vars = [tuple(self.tree.item(s, 'values')) for s in valids]
        # Remove selected items
        for item in valid_vars:
            self.sample_vars.remove(item)
        # Refresh table
        self._setup()
        self._build()

    def open_help(self):
        """Open help window."""
        self.help = tk.Toplevel(self.window)
        self.help.lift()
        self.help.title("Vector Creation Help")
        text = ("""
Creation of vectors for samples in loops with different creation settings.

USAGE:
---------------------
    1) Adjust vector creation settings in the LAM-main window

    2) Press 'Loop' to create vectors with chosen settings

    3) See vector plot file in 'Analysis Data\\Samples'

    4) 'Ctrl + click' to select all samples with valid vectors

    5) Press 'Keep selected' to remove samples from further loops

    6) Repeat steps 1-5 until all samples have vectors

    7) Press 'Done' and run LAM-analysis
""")
        tk.Label(self.help, text=text, font='TkDefaultFont 9 bold', anchor='w').grid(row=0, column=0, sticky='w')


def fixed_map(option):
    """Repair broken tkinter coloring."""
    return [elm for elm in ttk.Style().map("Treeview", query_opt=option) if elm[:2] != ("!disabled", "!selected")]


def print_settings():
    """Print settings of creation loop."""
    if Sett.SkeletonVector:
        sett_dict = {'Type': 'Skeleton', 'Simplif.': Sett.simplifyTol, 'Resize': Sett.SkeletonResize,
                     'Distance': Sett.find_dist, 'Dilation': Sett.BDiter, 'Smooth': Sett.SigmaGauss}
    else:
        sett_dict = {'Type': 'Median', 'Simplif.': Sett.simplifyTol, 'Bins': Sett.medianBins}
    print(f'Settings: {sett_dict}')
