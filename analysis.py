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
        if settings.Create_Channel_Plots:
            print("\nPlotting channels  ...")
            for chanPath in self._chanPaths:
                plotData = self.read_channel(chanPath, self._groups)
                make_plots = plotter(plotData, self._plotDir, palette=self._grpPalette)
                kws = {'id_str': 'Sample Group', 'hue': 'Sample Group', 
                       'row': 'Sample Group', 'centerline': make_plots.MPbin,
                       'xlabel':"Longitudinal Position", 'ylabel': 'Cell Count'}
                make_plots.plot_Data(plotter.boxPlot, **kws)
        if settings.Create_AddData_Plots:
            print("\nPlotting additional data  ...")
            for addPath in self._addData:
                plotData = self.read_channel(addPath, self._groups)
                make_plots = plotter(plotData, self._plotDir, palette=self._grpPalette)
                ylabel = settings.AddData.get(plotData.name.split('_')[0])[1]
                kws = {'id_str': 'Sample Group', 'hue': 'Sample Group', 
                       'row': 'Sample Group', 'centerline': make_plots.MPbin, 
                       'xlabel':"Longitudinal Position", 'ylabel': ylabel}
                make_plots.plot_Data(plotter.linePlot, **kws)
        if settings.Create_ChanVSChan_Plots:    
            # TODO make joinplots of chan vs. chan     
            pass
        if settings.Create_ChanVSAdd_Plots:    
            # TODO make joinplots of chan vs. Add
            print("\nPlotting channel VS additional data  ...")
            for chanPath in self._chanPaths:
                chanData = self.read_channel(chanPath, self._groups)
                chanData.loc[:, "Name"] = chanData.name
                for addPath in self._addData:
                    addData = self.read_channel(addPath, self._groups)
                    addData.name = settings.AddData.get(addData.name.split('_')[0])[1]
                    addData.loc[:, "Name"] = addData.name
                    plotData = pd.concat([chanData, addData])
                    plotData.name = "{} VS {}".format(chanData.name, addData.name)
                    print(plotData)
                    make_plots = plotter(plotData, self._plotDir, palette=self._grpPalette)
                    kws = {'id_str': ['Sample Group', 'Name'], 'hue': 'Sample Group', 
                           'row': 'Sample Group', 'xlabel':chanData.name, 
                           'ylabel': addData.name}
                    make_plots.plot_Data(plotter.jointPlot, **kws)
    
    def read_channel(self, path, groups):
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
            if settings.Drop_Outliers:
                temp = temp.where(__Drop, np.nan)
            temp.loc[:,"Sample Group"] = grp
            if plotData.empty: plotData = temp
            else: plotData = pd.concat([plotData, temp])
        plotData.name = '_'.join(str(path.stem).split("_")[1:])
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