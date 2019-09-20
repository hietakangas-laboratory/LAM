# -*- coding: utf-8 -*-
from settings import settings
from plot import plotter
import system
from system import store as store
import pandas as pd
import numpy as np
import pathlib as pl
import seaborn as sns
import re, warnings
from itertools import product, combinations

class Samplegroups:
    _instance = None
    _groups, _chanPaths, _samplePaths, _addData, _channels = [], [], [], [], []
    _grpPalette, _chanPalette = {}, {}
    _plotDir = pl.Path("./")
    _AllMPs = None
    
#    def __new__(cls, groups = None, channels = None, PATHS = None, child = True):
#        if not cls._instance:
#            cls._instance = super().__new__(cls)
#        return cls._instance
    
    def __init__(self, groups=None, channels=None, PATHS=None, child=True):
        if not child:
            Samplegroups._groups = groups
            Samplegroups._channels = channels
            Samplegroups._chanPaths = list(PATHS.datadir.glob("Norm_*"))
            Samplegroups._samplePaths = [p for p in PATHS.samplesdir.iterdir() if p.is_dir()]
            Samplegroups._addData = list(PATHS.datadir.glob("Avg_*"))
            Samplegroups._plotDir = PATHS.plotdir
            MPpath = PATHS.datadir.joinpath("MPs.csv")
            Samplegroups._AllMPs = system.read_data(MPpath, header=0, test=False)
            groupcolors = sns.color_palette("colorblind", len(groups), desat=.5)
            for i, grp in enumerate(groups):
                Samplegroups._grpPalette.update({grp: groupcolors[i]})
            chancolors = sns.color_palette("colorblind", len(self._chanPaths))
            for i, chan in enumerate(self._chanPaths):
                Samplegroups._chanPalette.update({chan: chancolors[i]})
        
    def create_plots(self):
        # Creation of boxplots of channel-specific cell counts
        if settings.Create_Channel_Plots:
            print("\nPlotting channels  ...")
            for chanPath in self._chanPaths:
                plotData = self.read_channel(chanPath, self._groups, drop = True)
                plot_maker = plotter(plotData, self._plotDir, palette=self._grpPalette)
                self.title = plotData.name
                kws = {'id_str': 'Sample Group', 'hue': 'Sample Group', 
                       'row': 'Sample Group', 'centerline': plot_maker.MPbin,
                       'xlabel':"Longitudinal Position", 'ylabel': 'Cell Count', 
                       'title': self.title, 'height':5, 'aspect':3}
                savepath = self._plotDir
                plot_maker.plot_Data(plotter.boxPlot, savepath, **kws)
        # Creation of lineplots for additional data
        if settings.Create_AddData_Plots:
            print("\nPlotting additional data  ...")
            for addPath in self._addData:
                plotData = self.read_channel(addPath, self._groups, drop = True)
                plot_maker = plotter(plotData, self._plotDir, palette=self._grpPalette)
                ylabel = settings.AddData.get(plotData.name.split('_')[0])[1]
                self.title = plotData.name
                kws = {'id_str': 'Sample Group', 'hue': 'Sample Group', 
                       'row': 'Sample Group', 'centerline': plot_maker.MPbin, 
                       'xlabel':"Longitudinal Position", 'ylabel': ylabel, 
                       'title': self.title, 'height':5, 'aspect':3}
                savepath = self._plotDir
                plot_maker.plot_Data(plotter.linePlot, savepath, **kws)
        # Creation of channel vs. channel jointplots
        if settings.Create_ChanVSChan_Plots:
            savepath = self._plotDir.joinpath("Chan VS Chan")
            savepath.mkdir(exist_ok=True)
            print("\nPlotting channel VS channel data  ...")
            self.Joint_looper(self._chanPaths, savepath)
        # Creation of channel vs. additional data jointplots
        if settings.Create_ChanVSAdd_Plots:
            savepath = self._plotDir.joinpath("Chan VS AddData")
            savepath.mkdir(exist_ok=True)
            print("\nPlotting channel VS additional data  ...")
            self.Joint_looper(self._chanPaths, savepath, self._addData, addit = True)
        if settings.Create_AddVSAdd_Plots:
            savepath = self._plotDir.joinpath("AddData VS AddData")
            savepath.mkdir(exist_ok=True)
            print("\nPlotting additional data VS additional data  ...")
            self.Joint_looper(self._addData, savepath, addit = True)
    
    def read_channel(self, path, groups, drop = False):
        def __Drop(Data):
            Mean = np.nanmean(Data.values)
            std = np.nanstd(Data.values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                Data = np.abs(Data.values-Mean) <= (settings.dropSTD*std)
            return Data
        
        Data = system.read_data(path, header=0, test=False)
        plotData = pd.DataFrame()
        for grp in groups:
            namer = str(grp+'_')
            namerreg = re.compile(namer, re.I)
            temp = Data.loc[:, Data.columns.str.contains(namerreg)].T
            if settings.Drop_Outliers and drop:
                temp = temp.where(__Drop, np.nan)
            temp["Sample Group"] = grp
            if plotData.empty: plotData = temp
            else: plotData = pd.concat([plotData, temp])
        plotData.name = '_'.join(str(path.stem).split("_")[1:])
        return plotData
    
    def Joint_looper(self, paths1, savepath, paths2 = None, addit = False):
        # Loop through all combinations of data sources:
        inputPaths = paths1
        if paths2 is not None: 
            inputPaths = [paths1, paths2]
            pairs = product(*inputPaths)
        else: pairs = combinations(inputPaths, 2)
        for pair in pairs:
            Path1, Path2 = pair[0], pair[1]
            # Find channel-data and add sample/channel-specific names for plotting
            Data1 = self.read_channel(Path1, self._groups)
            Data2 = self.read_channel(Path2, self._groups)
            Data1["Sample"] = Data1.index
            Data2["Sample"] = Data2.index
            namer1 = Data1.name
            namer2 = Data2.name
            if addit: # Find unit of additional data from settings
                ylabel = settings.AddData.get(Data2.name.split('_')[0])[1]
            else: ylabel = namer2
            # Melt data to long-form, and then merge to have one obs per row.
            Data1 = Data1.melt(id_vars = ["Sample Group", "Sample"], 
                               value_name = Data1.name, var_name = "Bin")
            Data2 = Data2.melt(id_vars = ["Sample Group", "Sample"], 
                                   value_name = Data2.name, var_name = "Bin")
            fullData = Data1.merge(Data2, on = ["Sample Group", 
                                                     "Sample", "Bin"])
            for group in self._groups:
                # Designate names for plotting and create a plotting object
                self.title = "{}_{} VS {}".format(group, namer1, namer2)
                grpData = fullData.where(fullData["Sample Group"]==group).dropna()
                grpData.name = self.title
                plot_maker = plotter(grpData, self._plotDir, title=self.title,
                                     palette=self._grpPalette)
                # Keyword arguments for plotting operations.
                kws = {'X': namer1, 'Y': namer2, 'row': 'Sample Group', 
                       'hue':'Sample Group', 'xlabel':namer1, 'ylabel':ylabel, 
                       'title': self.title, 'height':5, 'aspect':1}
                # Create plot
                plot_maker.plot_Data(plotter.jointPlot, savepath, 
                                 palette = self._grpPalette, **kws)
        
        
class Group(Samplegroups):
    _name = None
    _color = 'b'
    _groupPaths = None
    _MPs = None
    
    def __init__(self, group):
        super().__init__()
        self.group = list(group)
        Group._name = group
        self.namer = "{}_".format(group)
    
    def get_info(self):
        namerreg = re.compile(self.namer, re.I)
        Group._groupPaths = [p for p in self._samplePaths if namerreg.search(p.name)]
        Group._color = self._grpPalette.get(self._name)
        Group._MPs = self._AllMPs.loc[:, self._AllMPs.columns.str.contains(self.namer)]

