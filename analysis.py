# -*- coding: utf-8 -*-
from settings import settings
from plot import plotter
import system
from system import store as store
import pandas as pd
import numpy as np
import pathlib as pl
import seaborn as sns
import re

class Samplegroups(object):
    # TODO create gathering of all samplegroup data
    _instance = None
    _groups, _chanPaths, _samplePaths, _addData, _channels = [], [], [], [], []
    _grpPalette, _chanPalette = {}, {}
    _plotDir = pl.Path("./")
    
    def __new__(cls, groups = None, channels = None, PATHS = None):
        if not cls._instance:
            cls._instance = super(Samplegroups, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, groups, channels, PATHS):
        Samplegroups._groups = groups
        Samplegroups._channels = channels
        Samplegroups._chanPaths = list(PATHS.datadir.glob("Norm_*"))
        Samplegroups._samplePaths = [p for p in PATHS.samplesdir.iterdir() if p.is_dir()]
        Samplegroups._addData = list(PATHS.datadir.glob("Avg_*"))
        Samplegroups._plotDir = PATHS.plotdir
        groupcolors = sns.color_palette("colorblind", len(groups))
        for i, grp in enumerate(groups):
            Samplegroups._grpPalette.update({grp: groupcolors[i]})
        chancolors = sns.color_palette("colorblind", len(self._chanPaths))
        for i, chan in enumerate(self._chanPaths):
            Samplegroups._chanPalette.update({chan: chancolors[i]})
        
    def create_plots(self):
        for chanPath in self._chanPaths:
            plotData = self.read_channel(chanPath, self._groups)
            make_plots = plotter(plotData, self._plotDir, palette=self._grpPalette)
            kws = {'id_str': 'Sample Group', 'hue': 'Sample Group', 
                   'row': 'Sample Group'}
            make_plots.plot_Data(plotter.boxPlot, **kws)
        for addPath in self._addData:
            # TODO create facetgrid for addData plots
            continue
    
    def read_channel(self, path, groups):
        Data = system.read_data(path, header=0, test=False)
        plotData = pd.DataFrame()
        for grp in groups:
            namer = str(grp+'_')
            namerreg = re.compile(namer, re.I)
            temp = Data.loc[:, Data.columns.str.contains(namerreg)].T
            temp.loc[:,"Sample Group"] = grp
            if plotData.empty: plotData = temp
            else: plotData = pd.concat([plotData, temp])
        plotData.name = str(path.stem).split("_")[1]
        return plotData
        
class Group(Samplegroups):
    # TODO create gathering of data for one samplegroup    
    def __init__(self, group):
        self.group = group
        namer = group+"_"
        namerreg = re.compile(namer, re.I)
        self._samplePaths = [p for p in self._samplePaths if namerreg.search(p.name)]
        self.color = self.grpPalette.get(group)
        
#    def __getattr__(self, name):
#        try:
#            return getattr(self.parent, name)
#        except AttributeError:
#            raise AttributeError("Group object {} has no attribute {}".format(self.name, name))