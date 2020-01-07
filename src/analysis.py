# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""
# LAM imports
import system
import process
from settings import settings as Sett
from statsMWW import statistics, Total_Stats
from system import store
from plot import plotter
import logger as lg
# Standard libraries
import re
import warnings
from itertools import product, combinations, chain
# Other packages
import numpy as np
import pathlib as pl
from pycg3d.cg3d_point import CG3dPoint
from pycg3d import utils
import seaborn as sns
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)
    import pandas as pd
try:
    LAM_logger = lg.get_logger(__name__)
except AttributeError:
    print('Cannot get logger')


class Samplegroups:
    """Class for holding and handling all sample groups, i.e. every sample of
    analysis."""
    # Initiation of variables shared by all samples.
    _groups, _chanPaths, _samplePaths, _addData, _channels = [], [], [], [], []
    _plotDir, _dataDir, _statsDir = pl.Path('./'), pl.Path('./'), pl.Path('./')
    _grpPalette = {}
    _AllMPs = None
    _AllStarts = None
    _length = 0
    _center = None

    def __init__(self, groups=None, channels=None, length=0, center=None,
                 PATHS=None, child=False):
        # Creation of variables related to all samples, that are later passed
        # on to child classes.
        if not child:
            Samplegroups._groups = sorted(groups)
            Samplegroups._channels = channels
            Samplegroups._chanPaths = list(PATHS.datadir.glob('Norm_*'))
            Samplegroups._samplePaths = [p for p in PATHS.samplesdir.iterdir()
                                         if p.is_dir()]
            Samplegroups._addData = list(PATHS.datadir.glob('Avg_*'))
            Samplegroups._plotDir = PATHS.plotdir
            Samplegroups._dataDir = PATHS.datadir
            Samplegroups._statsDir = PATHS.statsdir
            Samplegroups._length = length
            MPpath = PATHS.datadir.joinpath('MPs.csv')
            Samplegroups._AllMPs = system.read_data(MPpath, header=0,
                                                    test=False)
            if center is not None:
                Samplegroups._center = center
                Samplegroups._AllStarts = Samplegroups._AllMPs.applymap(
                                                    lambda x: int(center - x))
            groupcolors = sns.xkcd_palette(Sett.palette_colors)
            for i, grp in enumerate(groups):
                Samplegroups._grpPalette.update({grp: groupcolors[i]})
            lg.logprint(LAM_logger, 'Sample groups established.', 'i')

    def create_plots(self):
        """For handling data for the creation of most plots, excluding stat
        plots. Passes data to plot.py for plotting functions."""
        # Base keywords utilized in plots.
        basekws = {'id_str': 'Sample Group',  'hue': 'Sample Group',
                   'row': 'Sample Group', 'height': 5, 'aspect': 3,
                   'var_str': 'Longitudinal Position', 'flierS': 2,
                   'title_y': 0.95, 'sharey': True,
                   'gridspec': {'hspace': 0.3}}

        def _select(paths, adds=True):
            """Selects varying data for the versus plots, e.g. a channel and
            additional data, such as area"""
            retPaths = []
            # Find target names from settings
            if adds:
                targets = Sett.vs_adds
            else:
                targets = Sett.vs_channels
            for trgt in targets:  # For each target, find corresponding file
                if adds:
                    namer = "^Avg_.*_{}.*".format(trgt)
                else:
                    namer = "^Norm_{}.*".format(trgt)
                reg = re.compile(namer, re.I)
                selected = [p for p in paths if reg.search(str(p.stem))]
                retPaths.extend(selected)
            return retPaths

        def __base(paths, func, ylabel='Cell Count', title=None, name_sep=1,
                   **kws):
            """Basic plotting for LAM, i.e. data of interest on y-axis, and
            bins on x-axis. Name_sep defines the start of data's name when file
            name is split by '_', e.g. name_sep=1 would name the data as DAPI
            when file name is 'Avg_DAPI'."""
            savepath = self._plotDir
            # For each data file to be plotted:
            for path in paths:
                # Find whether outliers are to be dropped
                dropB = Sett.Drop_Outliers
                # Read data from file and the pass it on to the plotter-class
                plotData, name, cntr = self.read_channel(path, self._groups,
                                                         drop=dropB,
                                                         name_sep=name_sep)
                plot_maker = plotter(plotData, self._plotDir, center=cntr,
                                     title=name, palette=self._grpPalette)
                # Give additional keywords for plotting
                kws2 = {'centerline': plot_maker.MPbin, 'value_str': ylabel,
                        'title': plot_maker.title, 'xlen': self._length,
                        'ylabel': ylabel}
                kws2.update(basekws)  # update to include the base keywords
                # If no given y-label, get name from file name / settings:
                if ylabel is None:
                    addName = plot_maker.title.split('-')[0].split('_')[1]
                    if "DistanceMeans" in addName:
                        newlabel = "Distance"
                    else:
                        temp = Sett.AddData.get(addName)
                        if temp is None:
                            newlabel = 'Cell Count'
                        else:
                            newlabel = temp[1]
                    kws2.update({'ylabel': newlabel, 'value_str': newlabel})
                kws2.update(kws)  # Update with kws passed to this function
                plot_maker.plot_Data(func, savepath, **kws2)  # Plotting

        def __heat(paths, samples=False):
            """Creation of heatmaps of cell counts on each channel for each
            sample group."""
            savepath = self._plotDir
            fullData = pd.DataFrame()
            # Loop through the given channels and read the data file. Each
            # channel is concatenated to one dataframe for easier plotting.
            for path in paths:
                plotData, name, cntr = self.read_channel(path, self._groups)
                # change each sample's group to be contained in its index
                # within the dataframe.
                if not samples:
                    plotData.index = plotData.loc[:, 'Sample Group']
                plotData.drop(labels='Sample Group', axis=1, inplace=True)
                # Channel-variable is added for identification
                plotData.loc[:, 'Channel'] = name
                fullData = pd.concat([fullData, plotData], axis=0, copy=False)
            # The plotting can't handle NaN's, so they are changed to zero.
            fullData = fullData.replace(np.nan, 0)
            if not samples:
                name = "All Channels Heatmaps"
            else:
                name = "All Samples Channel Heatmaps"
            # Initialize plotting-class, create kws, and plot all channel data.
            plot_maker = plotter(fullData, self._plotDir, center=cntr,
                                 title=name, palette=None)
            kws = {'height': 3, 'aspect': 5, 'sharey': False, 'row': 'Channel',
                   'title_y': 0.93, 'center': plot_maker.MPbin,
                   'xlen': self._length, 'gridspec': {'hspace': 0.5}}
            if samples:
                val = fullData.index.unique().size / 2
                kws.update({'height': val})
            plot_maker.plot_Data(plotter.Heatmap, savepath, **kws)

        def __versus(paths1, paths2=None, folder=None):
            """Creation of bivariant jointplots."""
            if folder:
                savepath = self._plotDir.joinpath(folder)
            else:
                savepath = self._plotDir
            savepath.mkdir(exist_ok=True)
            # Pass given paths to a function that pairs each variable
            self.Joint_looper(paths1, paths2, savepath)

        def __pair():
            """Creation of pairplot-grids, where each channel is paired with
            each other to show relations in counts."""
            allData = pd.DataFrame()
            # Loop through all channels.
            for path in self._dataDir.glob('ChanAvg_*'):
                # Find whether to drop outliers and then read data
                dropB = Sett.Drop_Outliers
                plotData, __, cntr = self.read_channel(path, self._groups,
                                                       drop=dropB)
                # get channel name from path, and add identification ('Sample')
                channel = str(path.stem).split('_')[1]
#                plotData['Sample'] = plotData.index
                # Change data into long form (one observation per row):
                plotData = pd.melt(plotData, id_vars=['Sample Group'],
                                   var_name='Longitudinal Position',
                                   value_name=channel)
                if allData.empty:
                    allData = plotData
                else:  # Merge data so that each row contains all channel
                    # counts from one bin of one sample
                    allData = allData.merge(plotData, how='outer', copy=False,
                                            on=['Sample Group',
                                                'Longitudinal Position'])
            name = 'All Channels Pairplots'
            # Initialize plotter, create plot keywords, and then create plots
            plot_maker = plotter(allData, self._plotDir, title=name,
                                 center=cntr, palette=self._grpPalette)
            kws = {'hue': 'Sample Group', 'kind': 'reg', 'diag_kind': 'kde',
                   'height': 3.5, 'aspect': 1, 'title_y': 1}
            plot_maker.plot_Data(plotter.pairPlot, self._plotDir, **kws)

        def __nearestDist():
            """Creation of plots regarding average distances to nearest cell.
            """
            # Find files containing calculated average distances
            paths = self._dataDir.glob('Avg_DistanceMeans_*')
            savepath = self._plotDir
            # Loop the found files
            for path in paths:
                # Read data file, create plotter, update keywords, and plot
                plotData, name, cntr = self.read_channel(path, self._groups)
                plot_maker = plotter(plotData, self._plotDir, center=cntr,
                                     title=name, palette=self._grpPalette)
                kws = {'centerline': plot_maker.MPbin,  'ylabel': 'Distance',
                       'title': plot_maker.title, 'xlen': self._length,
                       'title_y': 0.95}
                kws.update(basekws)
                plot_maker.plot_Data(plotter.linePlot, savepath, **kws)

        def __distributions():
            chanPaths = self._dataDir.glob('All_*')
            savepath = self._plotDir.joinpath('Distributions')
            savepath.mkdir(exist_ok=True)
            kws = {'hue': 'Sample Group', 'row': 'Sample Group', 'height': 5,
                   'sharey': True, 'aspect': 1, 'title_y': 0.95,
                   'gridspec': {'hspace': 0.4}}
            ylabel = 'Density'
            # Channels
            print("Channels  ...")
            for path in chanPaths:
                plotData, name, cntr = self.read_channel(path, self._groups)
                plot_maker = plotter(plotData, self._plotDir, center=cntr,
                                     title=name, palette=self._grpPalette)
                xlabel = 'Cell Count'
                kws.update({'id_str': 'Sample Group', 'var_str': xlabel,
                            'ylabel': ylabel, 'value_str': ylabel})
                plot_maker.plot_Data(plotter.distPlot, savepath, **kws)
            # Additional data
            for key in Sett.AddData.keys():
                print("{}  ...".format(key))
                ban = ['Group', 'Channel']
                AllVals = pd.DataFrame(columns=[key, 'Group', 'Channel'])
                paths = [p for s in self._samplePaths for p in s.glob('*.csv')
                         if p.stem not in ['MPs', 'Vector']]
                for path in paths:
                    try:
                        data = system.read_data(path, header=0, test=False,
                                                index_col=False)
                        values = data.loc[:, data.columns.str.contains(key)]
                        group = str(path.parent.name).split('_')[0]
                        channel = str(path.stem)
                        values = values.assign(Group=group, Channel=channel)
                        for col in values.loc[:, ~values.columns.isin(ban)
                                              ].columns:
                            if values.loc[:, col].nunique() == 1:
                                values.loc[:, str(col)] = np.nan
                        AllVals = pd.concat([AllVals, values], axis=0,
                                            ignore_index=True, sort=True)
                    except:  # TODO get rid of bare except
                        pass
                xlabel = Sett.AddData.get(key)[1]
                kws.update({'id_str': ['Group', 'Channel'], 'ylabel': ylabel,
                            'var_str': xlabel, 'value_str': ylabel,
                            'hue': 'Group', 'row': 'Group'})
                for col in AllVals.loc[:, ~AllVals.columns.isin(ban)].columns:
                    allData = AllVals.loc[:, ['Group', 'Channel',
                                              col]].dropna()
                    for plotChan in allData.Channel.unique():
                        name = "{}_{}".format(plotChan, col)
                        plotData = allData.loc[allData.Channel == plotChan, :]
                        plot_maker = plotter(plotData, self._plotDir,
                                             title=name,
                                             palette=self._grpPalette)
                        plot_maker.plot_Data(plotter.distPlot, savepath, **kws)

        def __clusters():
            # Find paths to sample-specific data based on found cluster data:
            chanpaths = [c for p in Samplegroups._samplePaths for c in
                         p.glob('*.csv')]
            clchans = [str(p.stem).split('-')[1] for p in
                       self._dataDir.glob('Clusters-*.csv')]
            if clchans:
                clpaths = [p for c in clchans for p in chanpaths if
                           p.name == "{}.csv".format(c)]
                # Create directory for cluster plots
                savepath = self._plotDir.joinpath('Clusters')
                savepath.mkdir(exist_ok=True)
                # Create sample-specific position plots:
                for path in clpaths:
                    data = system.read_data(path, header=0, test=False)
                    if 'ClusterID' in data.columns:
                        name = "{} clusters {}".format(path.parts[-2],
                                                       path.stem)
                        plot_maker = plotter(data, savepath, title=name)
                        plot_maker.clustPlot()
            else:
                msg = 'No cluster count files found (Clusters_*)'
                print('WARNING: {}'.format(msg))
                lg.logprint(LAM_logger, msg, 'w')
            fullData = pd.DataFrame()
            paths = list(self._dataDir.glob('ClNorm_*.csv'))
            if paths:
                for path in paths:
                    channel = str(path.stem).split('_')[1]
                    data = system.read_data(path, header=0, test=False).T
                    data.index = data.index.map(lambda x: str(x).split('_')[0])
                    groups = data.index.unique()
                    for grp in groups:
                        temp = data.loc[data.index == grp, :]
                        avgs = temp.mean(axis=0, numeric_only=True,
                                         skipna=True)
                        avgs['Channel'] = channel
                        avgs.rename(grp, inplace=True)
                        fullData = fullData.append(avgs.to_frame().T)
                # The plotting can't handle NaN's, so they are changed to zero.
                fullData = fullData.replace(np.nan, 0)
                # Initialize plotting-class and plot all channel data.
                name = "All Cluster Heatmaps"
                cntr = Samplegroups._center
                plot_maker = plotter(fullData, self._plotDir, center=cntr,
                                     title=name, palette=None)
                kws = {'height': 3, 'aspect': 5, 'gridspec': {'hspace': 0.5},
                       'row': 'Channel', 'title_y': 1.05, 'sharey': False,
                       'center': plot_maker.MPbin, 'xlen': self._length}
                plot_maker.plot_Data(plotter.Heatmap, savepath.parent, **kws)
            else:
                msg = 'No normalized cluster count files found (ClNorm_*)'
                print('WARNING: {}'.format(msg))
                lg.logprint(LAM_logger, msg, 'w')

        # Conditional function calls to create each of the plots.
        lg.logprint(LAM_logger, 'Begin plotting.', 'i')
        print("\n---Creating plots---")
        # Update addData variable to contain newly created average-files
        self._addData = list(self._dataDir.glob('Avg_*'))
        if Sett.Create_Channel_Plots:
            lg.logprint(LAM_logger, 'Plotting channels', 'i')
            print('Plotting channels  ...')
            __base(self._chanPaths, plotter.boxPlot)
            lg.logprint(LAM_logger, 'Channel plots done.', 'i')
        if Sett.Create_AddData_Plots:
            lg.logprint(LAM_logger, 'Plotting additional data', 'i')
            print('Plotting additional data  ...')
            __base(self._addData, plotter.linePlot, ylabel=None)
            lg.logprint(LAM_logger, 'Additional data plots done.', 'i')
        if Sett.Create_Channel_PairPlots:
            lg.logprint(LAM_logger, 'Plotting channel pairs', 'i')
            print('Plotting channel pairs  ...')
            __pair()
            lg.logprint(LAM_logger, 'Channel pairs done.', 'i')
        if Sett.Create_Heatmaps:
            lg.logprint(LAM_logger, 'Plotting heatmaps', 'i')
            print('Plotting heatmaps  ...')
            HMpaths = self._dataDir.glob("ChanAvg_*")
            __heat(HMpaths)
            HMpaths = self._dataDir.glob("Norm_*")
            __heat(HMpaths, samples=True)
            lg.logprint(LAM_logger, 'Heatmaps done.', 'i')
        if Sett.Create_ChanVSAdd_Plots:
            lg.logprint(LAM_logger, 'Plotting channel VS additional data', 'i')
            print('Plotting channel VS additional data  ...')
            paths1 = _select(self._chanPaths, adds=False)
            paths2 = _select(self._addData)
            __versus(paths1, paths2, 'Chan VS AddData')
            lg.logprint(LAM_logger, 'Channel VS additional data done.', 'i')
        if Sett.Create_AddVSAdd_Plots:
            lg.logprint(LAM_logger, 'Plotting add. data vs add. data', 'i')
            print('Plotting additional data VS additional data  ...')
            paths = _select(self._addData)
            __versus(paths, folder='AddData VS AddData')
            lg.logprint(LAM_logger, 'additional data VS additional data done',
                        'i')
        if Sett.Create_Distribution_Plots:
            lg.logprint(LAM_logger, 'Plotting distributions', 'i')
            print('-Distributions-')
            __distributions()
            lg.logprint(LAM_logger, 'Distributions done', 'i')
        # TODO nearest dist plots ???
#        if Sett.Create_NearestDist_Plots:
#            print('Plotting average distances  ...')
#            __nearestDist()
        if Sett.Create_Cluster_Plots:
            lg.logprint(LAM_logger, 'Plotting clusters', 'i')
            print('Plotting clusters  ...')
            __clusters()
            kws = {'ylabel': 'Clustered Cells'}
            paths = list(self._dataDir.glob('ClNorm_*'))
            if paths:
                lg.logprint(LAM_logger, 'Plotting cluster counts', 'i')
                __base(paths, plotter.boxPlot, **kws)
                lg.logprint(LAM_logger, 'Clusters done', 'i')
            else:
                print('No cluster files found')
                lg.logprint(LAM_logger, 'No cluster files found', 'e')
        lg.logprint(LAM_logger, 'Plotting completed', 'i')

    def read_channel(self, path, groups, drop=False, name_sep=1):
        """Reading of channel data, and concatenation of sample group info
        into the resulting dataframe."""
        Data = system.read_data(path, header=0, test=False)
        plotData = pd.DataFrame()
        # Loop through given groups and give an identification variable for
        # each sample belonging to the group.
        for grp in groups:
            namerreg = re.compile('^{}_'.format(grp), re.I)
            # Get only the samples that belong to the loop's current group
            temp = Data.loc[:, Data.columns.str.contains(namerreg)].T
            if Sett.Drop_Outliers and drop:  # conditionally drop outliers
                temp = DropOutlier(temp)
            temp['Sample Group'] = grp  # Giving of sample group identification
            if plotData.empty:
                plotData = temp
            else:
                plotData = pd.concat([plotData, temp])
        # Finding the name of the data under analysis from its filepath
        name = '_'.join(str(path.stem).split('_')[name_sep:])
        center = self._center  # Getting the bin to which samples are centered
        return plotData, name, center

    def Joint_looper(self, paths1, paths2=None, savepath=pl.PurePath()):
        """Creates joint-plots of channels and additional data, e.g. channel
        vs. channel."""
        def __label(name):
            # Gets unit of data from settings based on file name, if found.
            test = name.split('-')
            try:
                label = Sett.AddData.get(test[0].split('_')[1])[1]
                return label
            except:  # TODO get rid of bare except
                return

        # Create all possible pairs from the given lists of paths:
        if paths1 == paths2 or paths2 is None:
            pairs = combinations(paths1, 2)
        else:
            inputPaths = [paths1, paths2]
            pairs = product(*inputPaths)
        # Loop through the pairs of variables.
        for pair in pairs:
            (Path1, Path2) = pair
            # Find channel-data and add specific names for plotting
            Data1, name, cntr = self.read_channel(Path1, self._groups)
            Data2, name2, __ = self.read_channel(Path2, self._groups)
            Data1['Sample'], Data2['Sample'] = Data1.index, Data2.index
            name = ' '.join(name.split('_'))
            name2 = ' '.join(name2.split('_'))
            # Find unit of additional data from settings
            ylabel = __label(name)
            xlabel = __label(name2)
            # Melt data to long-form, and then merge to have one obs per row.
            Data1 = Data1.melt(id_vars=['Sample Group', 'Sample'],
                               value_name=name, var_name='Bin')
            Data2 = Data2.melt(id_vars=['Sample Group', 'Sample'],
                               value_name=name2, var_name='Bin')
            fullData = Data1.merge(Data2, on=['Sample Group', 'Sample', 'Bin'])
            # Get one sample group at a time for plotting:
            for group in self._groups:
                title = '{} {} VS {}'.format(group, name, name2)
                grpData = fullData.where(fullData['Sample Group'] == group
                                         ).dropna()
                plot_maker = plotter(grpData, self._plotDir, title=title,
                                     palette=self._grpPalette)
                kws = {'x': name, 'y': name2, 'hue': 'Sample Group',
                       'xlabel': xlabel, 'ylabel': ylabel, 'title': title,
                       'height': 5,  'aspect': 1, 'title_y': 1}
                plot_maker.plot_Data(plotter.jointPlot, savepath,
                                     palette=self._grpPalette, **kws)

    def subset_data(self, Data, compare, volIncl):
        """Get indexes of cells based on volume."""
        if not isinstance(Data, pd.DataFrame):
            lg.logprint(LAM_logger, 'Wrong data type for subset_data()', 'e')
            C = 'Wrong datatype for find_distance, Has to be pandas DataFrame.'
            print(C)
            return None
        ErrorM = "Volume not found in {}_{}'s {}".format(self.group,
                                                         self.name, Data.name)
        if compare.lower() == 'greater':
            try:  # Get only cells that are of greater volume
                subInd = Data[(Data['Volume'] >= volIncl)].index
            except KeyError:
                print(ErrorM)
        else:
            try:  # Get only cells that are of lesser volume
                subInd = Data[(Data['Volume'] <= volIncl)].index
            except KeyError:
                print(ErrorM)
        return subInd

    def Get_DistanceMean(self):
        """Gathers sample-data and passes it for calculation of average
        distances between cells."""
        lg.logprint(LAM_logger, 'Finding cell-to-cell distances', 'i')
        for grp in self._groups:  # Get one sample group
            lg.logprint(LAM_logger, '-> Distances for group {}'.format(grp),
                        'i')
            print('\n---Finding nearest cells for group {}---'.format(grp))
            SampleGroup = Group(grp)
            for path in SampleGroup._groupPaths:  # Get one sample of the group
                Smpl = Sample(path, SampleGroup.group)
                print('{}  ...'.format(Smpl.name))
                # Find distances between nuclei within the sample
                Smpl.DistanceMean(Sett.maxDist)
        lg.logprint(LAM_logger, 'Distances calculated', 'i')

    def Get_Clusters(self):
        """Gathers sample-data to compute cell clusters."""
        lg.logprint(LAM_logger, 'Finding clusters', 'i')
#        allpaths = [] ???
        for grp in self._groups:  # Get one sample group
            lg.logprint(LAM_logger, '-> clusters for group {}'.format(grp),
                        'i')
            print('\n---Finding clusters for group {}---'.format(grp))
            SampleGroup = Group(grp)
            for path in SampleGroup._groupPaths:  # Get one sample of the group
                Smpl = Sample(path, SampleGroup.group)
                print('{}  ...'.format(Smpl.name))
                paths = Smpl.Clusters(Sett.Cl_maxDist)  # Find clusters
                del paths
#                allpaths.append(paths)
#        paths = pd.DataFrame(allpaths)
#        paths.to_csv(self._plotDir.parent.joinpath("ClusterPaths.csv"),
#                     index=False)
        lg.logprint(LAM_logger, 'Clusters calculated', 'i')

    def Read_Clusters(self):  # ???
        for grp in store.samplegroups:
            namer = "{}_".format(grp)
            paths = [store.clusterPaths.pop(p) for p in store.clusterPaths if
                     p.parts[-2].startswith(namer)]
            group = Group(child=True)
            for path in paths:
                sample = Sample(path.parent, group)
                Data = system.read_data(path, header=0)
                sample.Count_clusters(Data, path.stem)

    def Get_Statistics(self):
        """Handling of data that is to be passed on to group-wise statistical
        analysis of cell counts on each channel, and additional data."""

        def _test_control():
            if Sett.cntrlGroup not in store.samplegroups:
                lg.logprint(LAM_logger, 'Set control group not found', 'c')
                test = 0
                namer = re.compile(r"{}$".format(re.escape(Sett.cntrlGroup)),
                                   re.I)
                for group in store.samplegroups:
                    if re.match(namer, group):
                        msg = "Control group-setting is case-sensitive!"
                        print("WARNING: {}".format(msg))
                        Sett.cntrlGroup = group
                        msg = "Control group has been changed to"
                        print("{} '{}'\n".format(msg, group))
                        lg.logprint(LAM_logger, '-> Changed to group {}'
                                    .format(group), 'i')
                        test += 1
                if test == 0:
                    msg = "control group NOT found in sample groups!"
                    print("WARNING: {}\n".format(msg))
                    flag = 1
                    while flag:
                        print('Found groups:')
                        for i, grp in enumerate(store.samplegroups):
                            print('{}: {}'.format(i, grp))
                        msg = "Select the number of control group: "
                        ans = int(input(msg))
                        if 0 <= ans <= len(store.samplegroups):
                            Sett.cntrlGroup = store.samplegroups[ans]
                            print("Control group set as '{}'.\n".format(
                                                            Sett.cntrlGroup))
                            flag = 0
                        else:
                            print('Command not understood.')
                    msg = "-> Changed to group '{}' by user".format(group)
                    lg.logprint(LAM_logger, msg, 'i')

        def _get_ylabel():
            if 'Clusters' in addChan_name[1]:
                ylabel = 'Clustered Cells'
            # If name is longer, the data is not cell counts, but
            # e.g. intensities, and requires different naming
            elif len(addChan_name) >= 3:
                if 'Distance Means' in addChan_name[2]:
                    ylabel = 'Cell-to-cell distance'
                else:
                    datakey = addChan_name[2].split('-')[0]
                    ylabel = Sett.AddData.get(datakey)[1]
            else:
                ylabel = "Count"
            return ylabel

        lg.logprint(LAM_logger, 'Calculation of statistics', 'i')
        if Sett.Create_Plots and Sett.Create_Statistics_Plots:
            print('\n---Calculating and plotting statistics---')
        else:
            print('\n---Calculating statistics---')
        _test_control()  # Test if given control group is found
        # Create stats of control vs. other groups if stat_versus set to True
        if Sett.stat_versus:
            lg.logprint(LAM_logger, 'Calculating versus statistics', 'i')
            print('-Versus-')
            # Finding control and other groups
            control = Sett.cntrlGroup
            cntrlName = re.compile(control, re.I)
            others = [g for g in self._groups if not cntrlName.fullmatch(g)]
            # Create all possible combinations of control versus other groups
            grouping = [[control], others]
            pairs = product(*grouping)
            # Loop through all the possible group pairs
            for pair in pairs:
                (temp, testgroup) = pair
                # Initiate group-class for both groups
                Cntrl = Group(control)
                Grp = Group(testgroup)
                # Print names of groups under statistical analysis
                print("{} Vs. {}  ...".format(Cntrl.group, Grp.group))
                # Initiate statistics-class with the two groups
                Stats = statistics(Cntrl, Grp)
                # Find stats of cell counts and additional data by looping
                # through each.
                for path in chain(Stats.chanPaths, Stats.avgPaths,
                                  Stats.clPaths):
                    Stats = Stats.MWW_test(path)
                    if Stats.error or "Distance Means" not in path.stem:
                        continue
                    # If plotting set to True, make plots of current stats
                    if Sett.Create_Statistics_Plots and Sett.Create_Plots:
                        # Find name of data and make title and y-label
                        addChan_name = str(path.stem).split('_')
                        titlep = '-'.join(addChan_name[1:])
                        Stats.plottitle = "{} = {}".format(Stats.title, titlep)
                        ylabel = _get_ylabel()
                        # Create statistical plots
                        Stats.Create_Plots(Stats.statData, ylabel,
                                           palette=self._grpPalette)
            lg.logprint(LAM_logger, 'Versus statistics done', 'i')
        # Create stats of total cell numbers if stat_total set to True
        if Sett.stat_total:
            lg.logprint(LAM_logger, 'Calculating total statistics', 'i')
            print('-Totals-')
            # Find the data file, initialize class, and count stats
            datapaths = self._dataDir.glob('Total*.csv')
            for path in datapaths:
                TCounts = Total_Stats(path, self._groups, self._plotDir,
                                      self._statsDir, self._grpPalette)
                TCounts.stats()
                # If wanted, create plots of the stats
                if Sett.Create_Plots and Sett.Create_Statistics_Plots:
                    TCounts.Create_Plots()
            lg.logprint(LAM_logger, 'Total statistics done', 'i')
        lg.logprint(LAM_logger, 'Statistics done', 'i')

    def Get_Totals(self):
        """Counting of sample & channel -specific cell count totals."""

        def _readAndSum(Avg=False):
            ChData, __, __ = self.read_channel(path, self._groups, drop=dropB)
#            ChData = ChData.T.drop('Sample Group')
            ChSum = ChData.sum(axis=1, skipna=True, numeric_only=True)  # Sums
            groups = ChData.loc[:, 'Sample Group']
            ChSum = ChSum.to_frame().assign(group=groups.values)
            ChSum.rename(columns={'group': 'Sample Group'}, inplace=True)
            return ChSum

        lg.logprint(LAM_logger, 'Finding total counts', 'i')
        dropB = Sett.Drop_Outliers
        datadir = self._dataDir
        All = pd.DataFrame()
        # Loop through files containing cell count data, read, and find sums
        for path in datadir.glob('All_*'):
            ChSum = _readAndSum()
            channel = path.stem.split('_')[1]  # Channel name
            ChSum = ChSum.assign(Variable=channel)
            All = pd.concat([All, ChSum], ignore_index=False, sort=False)
        # Save dataframe containing sums of each channel for each sample
        system.saveToFile(All, datadir, 'Total Counts.csv',
                          append=False, w_index=True)
        for channel in [c for c in store.channels if c not in ['MP', 'R45']]:
            All = pd.DataFrame()
            for path in datadir.glob('Avg_{}_*'.format(channel)):
                ChData, __, __ = self.read_channel(path, self._groups,
                                                   drop=dropB)
                add_name = path.stem.split('_')[2:]  # Channel name
                ChData = ChData.assign(Variable='_'.join(add_name))
                All = pd.concat([All, ChData], ignore_index=False, sort=False)
            All = All[All.iloc[:, :-3].nunique(axis=1, dropna=True) > 1]
        # Save dataframe containing sums of each channel for each sample
            filename = 'Total {} AddData.csv'.format(channel)
            system.saveToFile(All, datadir, filename, append=False,
                              w_index=True)
        All = pd.DataFrame()
        for path in chain(datadir.glob('Clusters-*.csv'),
                          datadir.glob('*Distance Means.csv')):
            if 'Clusters-' in path.name:
                name = "{} Clusters".format(path.stem.split('-')[1])
            else:
                name = "{} Distances".format(path.name.split('_')[1])
            ChData, __, __ = self.read_channel(path, self._groups,
                                               drop=dropB)
            ChData = ChData.assign(Variable=name)
            All = pd.concat([All, ChData], ignore_index=False, sort=False)
        if not All.empty:
            filename = 'Total Distance Data.csv'
            system.saveToFile(All, datadir, filename, append=False,
                              w_index=True)
        lg.logprint(LAM_logger, 'Total counts done', 'i')


class Group(Samplegroups):
    """For storing sample group-specific data."""
    _color = 'b'
    _groupPaths = None
    _MPs = None

    def __init__(self, group, child=False):
        super().__init__(child=True)  # Inherit from samplegroups-class
        self.group = group  # group
        # For finding group-specific columns etc.
        self.namer = '{}_'.format(group)
        # When first initialized, create variables inherited by samples:
        if not child:
            self.color = self._grpPalette.get(self.group)
            namerreg = re.compile("^{}".format(self.namer), re.I)
            self.groupPaths = [p for p in self._samplePaths if namerreg.search(
                                p.name)]
            self.MPs = self._AllMPs.loc[:, self._AllMPs.columns.str.contains(
                                        self.namer)]
            Group._color = (self.color)
            Group._groupPaths = self.groupPaths
            Group._MPs = self.MPs


class Sample(Group):
    """For storing sample-specific data and handling sample-related
    functionalities."""

    def __init__(self, path, grp):
        # Inherit variables from the sample's group
        super().__init__(grp, child=True)
        # Sample's name, path to its directory, and paths to data it has
        self.name = str(path.stem)
        self.path = path
        self.channelPaths = [p for p in path.iterdir() if p.suffix == '.csv' if
                             p.stem not in ['vector', 'MPs']]
        # Sample's group-specific color, and it's anchoring bin.
        self.color = Group._color
        self.MP = self._MPs.loc[0, self.name]

    def DistanceMean(self, dist=25):
        """Preparation and data handling for finding nearest distances between
        cells"""
        kws = {'Dist': dist}  # Maximum distance used to find cells
        # List paths of channels where distances are to be found
        distChans = [p for p in self.channelPaths for t in
                     Sett.Distance_Channels if t.lower() == p.stem.lower()]
        if Sett.use_target:  # If distances are found against other channel:
            target = Sett.target_chan  # Get the name of the target channel
            try:  # Find target's data file, read, and update data to keywords
                file = '{}.csv'.format(target)
                tNamer = re.compile(file, re.I)
                targetPath = [p for p in self.channelPaths if
                              tNamer.fullmatch(str(p.name))]
                tData = system.read_data(targetPath[0], header=0)
                kws.update({'tData': tData})
            except (FileNotFoundError, IndexError):
                msg = "No file for channel {}".format(target)
                lg.logprint(LAM_logger, "{}: {}".format(self.name, msg), 'w')
                print("-> {}".format(msg))
                return
        # Loop through the channels, read, and find distances
        for path in distChans:
            try:
                Data = system.read_data(path, header=0)
            except FileNotFoundError:
                msg = "No file for channel {}".format(path.stem)
                lg.logprint(LAM_logger, "{}: {}".format(self.name, msg), 'w')
                print("-> {}".format(msg))
                return
            Data = Data.loc[:, ~Data.columns.str.startswith('Nearest_')]
            Data.name = path.stem
            self.find_distances(Data, volIncl=Sett.Vol_inclusion,
                                compare=Sett.incl_type, **kws)

    def Clusters(self, dist=10):
        """Preparation and data handling for finding clusters of cells."""
        kws = {'Dist': dist}  # Maximum distance for considering clustering
        # Listing of paths of channels on which clusters are to be found
        clustChans = [p for p in self.channelPaths for t in
                      Sett.Cluster_Channels if t.lower() == p.stem.lower()]
        for path in clustChans:  # Loop paths, read file, and find clusters
            try:
                Data = system.read_data(path, header=0)
            except (FileNotFoundError, AttributeError):
                msg = "No file for channel {}".format(path.stem)
                lg.logprint(LAM_logger, "{}: {}".format(self.name, msg), 'w')
                print("-> {}".format(msg))
                return
            Data = Data.loc[:, ~Data.columns.str.contains('ClusterID')]
            Data.name = path.stem  # The name of the clustering channel
            self.find_distances(Data, volIncl=Sett.Cl_Vol_inclusion,
                                compare=Sett.Cl_incl_type, clusters=True,
                                **kws)
        return clustChans

    def find_distances(self, Data, volIncl=200, compare='smaller',
                       clusters=False, **kws):
        """Calculate distances between cells to either find the nearest cell
        and distance means per bin, or to find cell clusters. Argument "Data"
        is channel data from a sample."""

        def __get_nearby(ind, row, target, maxDist, rmv_self=False, **kws):
            """Within an iterator, find all cells near the current cell, to be
            passed either to find nearest cell or to determine clustering."""
            point = CG3dPoint(row.x, row.y, row.z)
            # When finding nearest in the same channel, remove the current
            # cell from the frame, otherwise nearest cell would be itself.
            if rmv_self:
                target = target.loc[target.index.difference([ind]), :]
                # Find cells within the accepted limits (Sett.maxDist)
            near = target[((abs(target.x - row.x) <= maxDist) &
                           (abs(target.y - row.y) <= maxDist) &
                           (abs(target.z - row.z) <= maxDist))].index
            if not near.empty:  # Then get distances to nearby cells:
                cols = ['XYZ', 'Dist', 'ID']
                nearby = pd.DataFrame(columns=cols)
                # Loop through the nearby cells
                for i2, row2 in target.loc[near, :].iterrows():
                    point2 = CG3dPoint(row2.x, row2.y, row2.z)
                    # Distance from the first cell to the second
                    dist = utils.distance(point, point2)
                    if dist <= maxDist:  # If distance is OK, store data
                        temp = pd.Series([(row2.x, row2.y, row2.z), dist,
                                          row2.ID], index=cols, name=i2)
                        nearby = nearby.append(temp, ignore_index=True)
                # if there are cells nearby, return data
                if not nearby.empty:
                    return nearby
            # If no nearby cells, return with None
            return None

        def __find_clusters():
            """Finding of cluster 'seeds', and merging them to create full
            clusters."""
            def __merge(Seeds):
                """Merging of seeds that share cells."""
                r = sum(Seeds, [])  # List of all cells
                # Create map object containing a set for each cell ID:
                r = map(lambda x: set([x]), set(r))
                # Loop through a set of each seed
                for item in map(set, Seeds):
                    # For each seed, find IDs from the set of cell
                    # IDs and merge them
                    out = [x for x in r if not x & item]  # ID-sets not in seed
                    mSeeds = [x for x in r if x & item]  # found ID-sets
                    # make union of the ID sets that are found
                    mSeeds = set([]).union(*mSeeds)
                    # Reassign r to contain the newly merged ID-sets
                    r = out + [mSeeds]
                yield r

            maxDist = kws.get('Dist')  # max distance to consider clustering
            clusterSeed = {}  # For storing cluster 'seeds'
            for i, row in XYpos.iterrows():  # Iterate over all cells
                # Find nearby cells
                nearby = __get_nearby(i, row, XYpos, maxDist, **kws)
                # If nearby cells, make a list of their IDs and add to seeds
                if nearby is not None:
                    if nearby.shape[0] > 1:
                        clusterSeed[i] = nearby.ID.tolist()
            # Make a sorted list of lists of the found cluster seeds
            Cl_lst = [sorted(list(clusterSeed.get(key))) for key in
                      clusterSeed.keys()]
            # Merging of the seeds
            Cl_gen = __merge(Cl_lst)
            # Change the generator into list of lists and drop clusters of size
            # under/over limits
            Clusters = [list(y) for x in Cl_gen for y in x if y and len(y) >=
                        Sett.Cl_min and len(y) <= Sett.Cl_max]
            return Clusters

        def __find_nearest():
            """For iterating the passed data to determine nearby cells."""
            maxDist = kws.get('Dist')  # distance used for subsetting target
            # If distances are found on other channel:
            if 'targetXY' in locals():
                target = targetXY
                comment = Sett.target_chan
                filename = 'Avg_{} VS {}_Distance Means.csv'.format(Data.name,
                                                                    comment)
                rmv = False
            else:  # If using the same channel:
                target = XYpos
                rmv = True
                comment = Data.name
                filename = 'Avg_{}_Distance Means.csv'.format(Data.name)
            cols = ['Nearest_XYZ_{}'.format(comment), 'Nearest_Dist_{}'.format(
                    comment), 'Nearest_ID_{}'.format(comment)]
            pointData = pd.DataFrame(columns=cols, index=XYpos.index)
            # Iterate over each cell (row) in the data
            for i, row in XYpos.iterrows():
                nearby = __get_nearby(i, row, target, maxDist, rmv_self=rmv,
                                      **kws)
                if nearby is not None:
                    nearest = nearby.Dist.idxmin()
                    pointData.loc[i, cols] = nearby.loc[nearest].values
            # Concatenate the obtained data with the read data.
            NewData = pd.concat([Data, pointData], axis=1)
            # Get bin and distance to nearest cell for each cell, calculate
            # average distance within each bin.
            binnedData = NewData.loc[:, 'DistBin']
            distances = NewData.loc[:, cols[1]].astype('float64')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                means = [np.nanmean(distances[binnedData.values == k]) for k in
                         np.arange(0, store.binNum)]
            return NewData, means, filename

        if volIncl > 0:  # Subsetting of data based on cell volume
            dInd = self.subset_data(Data, compare, volIncl)
            if 'tData' in kws.keys():  # Obtain target channel if used
                tData = kws.pop('tData')
                tData.name = Data.name
                tInd = self.subset_data(tData, compare, volIncl)
        elif 'tData' in kws.keys():
            tData = kws.pop('tData')
            tInd = tData.index
            dInd = Data.index
        else:
            dInd = Data.index
        # Accessing the data for the analysis via the indexes taken before.
        # Cells for which the nearest cells will be found:
        XYpos = Data.loc[dInd, ['Position X', 'Position Y', 'Position Z',
                                'ID', 'DistBin']]
        renames = {'Position X': 'x', 'Position Y': 'y', 'Position Z': 'z'}
        XYpos.rename(columns=renames, inplace=True)  # rename for dot notation
        if 'tInd' in locals():  # Get data from target channel, if used
            targetXY = tData.loc[tInd, ['Position X', 'Position Y',
                                        'Position Z', 'ID']]
            targetXY.rename(columns=renames, inplace=True)
        if not clusters:  # Finding nearest distances
            NewData, Means, filename = __find_nearest()
            Means = pd.Series(Means, name=self.name)
            insert, _ = process.relate_data(Means, self.MP, self._center,
                                            self._length)
            SMeans = pd.Series(data=insert, name=self.name)
            system.saveToFile(SMeans, self._dataDir, filename)
        else:  # Finding clusters
            Clusters = __find_clusters()
            # Create dataframe for storing the obtained data
            clustData = pd.DataFrame(index=Data.index, columns=['ID',
                                                                'ClusterID'])
            clustData = clustData.assign(ID=Data.ID)  # Copy ID column
            # Give name from a continuous range to each of the found clusters
            # and add it to cell-specific data (for each belonging cell).
            if Clusters:
                for i, vals in enumerate(Clusters):
                    vals = [int(v) for v in vals]
                    clustData.loc[clustData.ID.isin(vals), 'ClusterID'] = i
            # Merge obtained data with the original data
            NewData = Data.merge(clustData, how='outer', copy=False, on=['ID'])
            self.Count_clusters(NewData, Data.name)
        # Overwrite original sample data with the data containing new columns
        OW_name = '{}.csv'.format(Data.name)
        system.saveToFile(NewData, self.path, OW_name, append=False)

    def Count_clusters(self, Data, name):
        # Find bins of the clustered cells to find counts per bin
        idx = Data.loc[:, 'ClusterID'].notna().index
        binnedData = Data.loc[Data.dropna(subset=['ClusterID']).index,
                              'DistBin']
        # Sort values and then get counts
        bins = binnedData.sort_values().to_numpy()
        unique, counts = np.unique(bins, return_counts=True)
        idx = np.arange(0, store.binNum)
        # Create series to store the cell count data
        binnedCounts = pd.Series(np.full(len(idx), np.nan), index=idx,
                                 name=self.name)
        binnedCounts.loc[unique] = counts
        filename = 'Clusters-{}.csv'.format(name)
        system.saveToFile(binnedCounts, self._dataDir, filename)
        # Relate the counts to context, i.e. anchor them at the MP
        insert, _ = process.relate_data(binnedCounts, self.MP,
                                        self._center, self._length)
        # Save the data
        SCounts = pd.Series(data=insert, name=self.name)
        filename = 'ClNorm_Clusters-{}.csv'.format(name)
        system.saveToFile(SCounts, self._dataDir, filename)


def DropOutlier(Data):
    with warnings.catch_warnings():  # Ignore warnings regarding empty bins
        warnings.simplefilter('ignore', category=RuntimeWarning)
        Mean = np.nanmean(Data.values)
        std = np.nanstd(Data.values)
        Data = Data.applymap(lambda x: x if np.abs(x - Mean) <=
                             (Sett.dropSTD * std) else np.nan)
    return Data