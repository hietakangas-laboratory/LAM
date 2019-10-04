# -*- coding: utf-8 -*-
from settings import settings
from plot import plotter
import system, process, numpy as np, pathlib as pl, seaborn as sns, re, warnings
from itertools import product, combinations, chain
from pycg3d.cg3d_point import CG3dPoint
from pycg3d import utils
import scipy.stats as ss, statsmodels.stats.multitest as multi
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)
    import pandas as pd

class Samplegroups:
    _instance = None
    _groups, _chanPaths, _samplePaths, _addData, _channels = ([], [], [], [], [])
    _plotDir, _dataDir, _statsDir = pl.Path('./'), pl.Path('./'), pl.Path('./')
    _grpPalette, _chanPalette = {}, {}
    _AllMPs = None
    _AllStarts = None
    _length = 0
    _center = int(len(settings.projBins / 2))
    
#    def __new__(cls, groups = None, channels = None, PATHS = None, child = True):
#        if not cls._instance:
#            cls._instance = super().__new__(cls)
#        return cls._instance

    def __init__(self, groups=None, channels=None, length=0, center=None, 
                 PATHS=None, child=False):
        if not child:
            Samplegroups._groups = groups
            Samplegroups._channels = channels
            Samplegroups._chanPaths = list(PATHS.datadir.glob('Norm_*'))
            Samplegroups._samplePaths = [p for p in PATHS.samplesdir.iterdir() if p.is_dir()]
            Samplegroups._addData = list(PATHS.datadir.glob('Avg_*'))
            Samplegroups._plotDir = PATHS.plotdir
            Samplegroups._dataDir = PATHS.datadir
            Samplegroups._statsDir = PATHS.statsdir
            Samplegroups._length = length
            MPpath = PATHS.datadir.joinpath('MPs.csv')
            Samplegroups._AllMPs = system.read_data(MPpath, header=0, test=False)
            if center is not None:
                Samplegroups._center = center
                Samplegroups._AllStarts = Samplegroups._AllMPs.applymap(
                                                    lambda x: int(center - x))
            groupcolors = sns.xkcd_palette(settings.palette_colors)
            for i, grp in enumerate(groups):
                Samplegroups._grpPalette.update({grp: groupcolors[i]})

    def create_plots(self):
        basekws = {'id_str':'Sample Group',  'hue':'Sample Group',  
                   'row':'Sample Group', 'height':5, 'aspect':3,  
                   'var_str':'Longitudinal Position', 'flierS': 3}
        
        def _select(paths, adds=True):
            retPaths = []
            if adds: targets = settings.vs_adds
            else: targets = settings.vs_channels
            for trgt in targets:
                if adds: namer = "^Avg_.*_{}.*".format(trgt)
                else: namer = "^Norm_{}_.*".format(trgt)
                reg = re.compile(namer, re.I)
                selected = [p for p in paths if reg.search(str(p.stem))]
                retPaths.extend(selected)
            return retPaths

        def __base(paths, func, ylabel=None, **kws):
            savepath = self._plotDir
            for path in paths:
                dropB = settings.Drop_Outliers
                plotData, name, cntr = self.read_channel(path, self._groups, 
                                                           drop=dropB)
                plot_maker = plotter(plotData, self._plotDir, center=cntr,
                             title=name, palette=self._grpPalette)
                kws = {'centerline':plot_maker.MPbin, 'title':plot_maker.title,  
                       'value_str': 'Cell Count', 'xlen': self._length, 
                       'ylabel':ylabel}
                kws.update(basekws)
                if ylabel is None:
                    ylabel = settings.AddData.get(plot_maker.title)
                    kws.update({'ylabel': ylabel})
                plot_maker.plot_Data(func, savepath, **kws)

        def __versus(paths1, paths2=None, folder=None):
            """Creation of bivariant jointplots."""
            if folder:
                savepath = self._plotDir.joinpath(folder)
            else: savepath = self._plotDir
            savepath.mkdir(exist_ok=True)
            self.Joint_looper(paths1, paths2, savepath)
            
        def __pair():
            allData = pd.DataFrame()
            for path in Samplegroups._chanPaths:
                dropB = settings.Drop_Outliers
                plotData, __, cntr = self.read_channel(path, self._groups, 
                                                           drop=dropB)
                channel = str(path.stem).split('_')[1]
                plotData['Sample'] = plotData.index
                plotData = pd.melt(plotData, id_vars=['Sample','Sample Group'],
                                   var_name='Longitudinal Position',
                                   value_name=channel)
                if allData.empty: allData = plotData
                else: 
                    allData = allData.merge(plotData, how='outer', copy=False,
                                            on=['Sample', 'Sample Group', 
                                                'Longitudinal Position'])
            name = 'All Channels Pairplots'
            plot_maker = plotter(allData, self._plotDir, title=name, center=cntr,
                                 palette=self._grpPalette)
            kws = {'hue':'Sample Group', 'kind': 'reg', 'diag_kind': 'kde',
                   'height':3.5, 'aspect':1}
            plot_maker.plot_Data(plotter.pairPlot, self._plotDir, **kws)

        def __nearestDist():
            paths = self._dataDir.glob('Avg_DistanceMeans_*')
            savepath = self._plotDir
            for path in paths:
                plotData, name, cntr = self.read_channel(path, self._groups)
                plot_maker = plotter(plotData, self._plotDir, center=cntr, 
                                     title=name, palette=self._grpPalette)
                kws = {'centerline':plot_maker.MPbin,  'ylabel':'Distance',  
                       'title':plot_maker.title, 'xlen':self._length}
                kws.update(basekws)
                (plot_maker.plot_Data)((plotter.linePlot), savepath, **kws)
        #-------#
        print("\n---Creating plots---")
        if settings.Create_Channel_Plots:
            print('\nPlotting channels  ...')
            __base(self._chanPaths, plotter.boxPlot, ylabel='Cell Count')
        if settings.Create_AddData_Plots:
            print('\nPlotting additional data  ...')
            __base(self._addData, plotter.linePlot)
        if settings.Create_Channel_PairPlots:
            print('\nPlotting channel pairs  ...')
            __pair()
        if settings.Create_ChanVSAdd_Plots:
            print('\nPlotting channel VS additional data  ...')
            paths1 = _select(self._chanPaths, adds=False)
            paths2 = _select(self._addData)
            __versus(paths1, paths2, 'Chan VS AddData')
        if settings.Create_AddVSAdd_Plots:
            print('\nPlotting additional data VS additional data  ...')
            paths = _select(self._addData)
            __versus(paths, folder = 'AddData VS AddData')
        if settings.Create_NearestDist_Plots:
            print('\nPlotting average distances  ...')
            __nearestDist()
        # TODO add sample group total counts / plots & stats

    def read_channel(self, path, groups, drop=False):
        Data = system.read_data(path, header=0, test=False)
        plotData = pd.DataFrame()
        for grp in groups:
            namer = str(grp + '_')
            namerreg = re.compile(namer, re.I)
            temp = Data.loc[:, Data.columns.str.contains(namerreg, regex=True)].T
            if settings.Drop_Outliers and drop:
                temp = DropOutlier(temp)
            temp['Sample Group'] = grp
            if plotData.empty: plotData = temp
            else: plotData = pd.concat([plotData, temp])
        name = '_'.join(str(path.stem).split('_')[1:])
        center = self._center
        return plotData, name, center

    def Joint_looper(self, paths1, paths2 = None, savepath = pl.PurePath()):
        def __label(name):
            """Gets unit of data from settings based on file name, if found."""
            test = name.split('-')
            try:
                label = settings.AddData.get(test[0].split('_')[1])[1]                
                return label
            except: return 
            
        # Loop through all combinations of data sources:
        if paths1 == paths2 or paths2 == None:
            pairs = combinations(paths1, 2)
        else:
            inputPaths = [paths1, paths2]
            pairs = product(*inputPaths)
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
            for group in self._groups:
                title = '{} {} VS {}'.format(group, name, name2)
                grpData = fullData.where(fullData['Sample Group'] == group).dropna()
                plot_maker = plotter(grpData, self._plotDir, title=title, 
                                     palette=self._grpPalette)
                kws = {'x': name, 'y': name2, 'hue':'Sample Group', 
                       'xlabel': xlabel, 'ylabel':ylabel, 'title':title, 
                       'height':5,  'aspect':1}
                plot_maker.plot_Data(plotter.jointPlot, savepath, 
                                     palette=self._grpPalette, **kws)

    def subset_data(self, Data, compare, volIncl):
        """Get indexes of cells based on volume."""
        if not isinstance(Data, pd.DataFrame):
            print('Wrong datatype for find_distance(), has to be pandas DataFrame.')
            return
        ErrorM = "Volume not found in {}_{}'s {}".format(self.group, 
                                                  self.name, Data.name)
        if compare.lower() == 'greater':
            try:
                subInd = Data[(Data['Volume'] >= volIncl)].index
            except KeyError: print(ErrorM)
        else:
            try:
                subInd = Data[(Data['Volume'] <= volIncl)].index
            except KeyError: print(ErrorM)
        return subInd

    def Get_DistanceMean(self):
        for grp in self._groups:
            print('\n---Finding nearest cells for group {}---'.format(grp))
            SampleGroup = Group(grp)
            for path in SampleGroup._groupPaths:
                Smpl = Sample(path, SampleGroup.group)
                print('\n{}  ...'.format(Smpl.name))
                Smpl.DistanceMean(settings.maxDist)

    def Get_Statistics(self):
        if settings.Create_Plots:
            print('\n---Calculating and plotting statistics---')
        else: print('\n---Calculating statistics---')
        control = settings.cntrlGroup
        cntrlName = re.compile(control, re.I)
        others = [g for g in self._groups if not cntrlName.fullmatch(g)]
        grouping = [[control], others]
        pairs = product(*grouping)
        current = ''
        for pair in pairs:
            temp, testgroup = pair
            Cntrl = Group(control)
            Grp = Group(testgroup)
            if current != Grp.group:
                current = Grp.group
                print("{} Vs. {}  ...".format(Cntrl.group, Grp.group))
            Stats = statistics(Cntrl, Grp)
            for path in chain(Stats.chanPaths,Stats.avgPaths):
                Stats = Stats.MWW_test(path)
                if settings.Create_Plots and settings.Create_Statistics_Plots:
                    addChan_name = str(path.stem).split('_')[1:]
                    Stats.plottitle = "{} = {}".format(Stats.title, 
                                       '-'.join(addChan_name))
                    ylabel = "Count" 
                    if len(addChan_name) > 2:
                        temp = addChan_name.split('-')
                        try:
                            ylabel = settings.AddData.get(temp[0].split('_')[1])[1] 
                        except: pass
                    Stats.Create_Plots(Stats.statData, ylabel, 
                                       palette=self._grpPalette)

class Group(Samplegroups):
    _color = 'b'
    _groupPaths = None
    _MPs = None

    def __init__(self, group, child=False):
        super().__init__(child=True)
        self.group = group
        self.namer = '{}_'.format(group)
        if not child:
            self.color = self._grpPalette.get(self.group)
            namerreg = re.compile(self.namer, re.I)
            self.groupPaths = [p for p in self._samplePaths if namerreg.search(
                                                                        p.name)]
            self.MPs = self._AllMPs.loc[:,
            self._AllMPs.columns.str.contains(self.namer)]
            Group._color = (self.color)
            Group._groupPaths = self.groupPaths
            Group._MPs = self.MPs


class Sample(Group):
    def __init__(self, path, grp):
        super().__init__(grp, child=True)
        self.name = str(path.stem)
        self.path = path
        self.channelPaths = [p for p in path.iterdir() if p.suffix == '.csv' if 
                             p.stem not in ('vector','MPs')]
        self.color = Group._color
        self.MP = self._MPs.loc[(0, self.name)]

    def DistanceMean(self, dist=10):
        kws = {'Dist': dist}
        distChans = [p for p in self.channelPaths for t in 
                     settings.Distance_Channels if t == p.stem]
        if settings.use_target:
            target = settings.target_chan
            try:
                file = '{}.csv'.format(target)
                tNamer = re.compile(file, re.I)
                targetPath = [p for p in self.channelPaths if 
                              tNamer.fullmatch(str(p.name))]
                tData = system.read_data(targetPath[0], header=0)
                kws.update({'tData': tData})
            except:
                print("Sample doesn't have file for channel {}".format(target))
                return
        for path in distChans:
            try:
                Data = system.read_data(path, header=0)
            except:
                print("Sample doesn't have file for channel {}".format(path.stem))
                return
            Data = Data.loc[:, ~Data.columns.str.contains('Nearest_')]
            Data.name = path.stem
            Data = (self.find_distances)(Data, volIncl=settings.Vol_inclusion, 
                   compare=settings.incl_type, **kws)

    def Clusters(self):
        pass

    def find_distances(self, Data, volIncl=200, compare='smaller', clusters=False, 
                       **kws):
        """Calculate distances between cells to either find the nearest cell 
        and distance means per bin, or to find cell clusters. Argument "Data" 
        is channel data from a sample."""

        def __get_nearby(ind, row, target, rmv_self=False, **kws):
            """Within an iterator, find all cells near the current cell."""
            maxDist = kws.get('Dist') # the distance used for subsetting target
            point = CG3dPoint(row.x, row.y, row.z)
            # When finding nearest in the same channel, remove the current
            # cell from the frame, otherwise nearest cell would be itself.
            if rmv_self == True:
                target = target.loc[target.index.difference([ind]), :]
                # Find cells within the accepted limits (settings.maxDist)
            near = target[((abs(target.x - row.x) <= maxDist) & 
                          (abs(target.y - row.y) <= maxDist) & 
                          (abs(target.z - row.z) <= maxDist))].index
            if not near.empty: # Then get distances to nearby cells:
                cols = [
                 'XYZ', 'Dist', 'ID']
                nearby = pd.DataFrame(columns=cols)
                for i2, row2 in target.loc[near, :].iterrows():
                    point2 = CG3dPoint(row2.x, row2.y, row2.z)
                    dist = utils.distance(point, point2)
                    if dist <= maxDist: # If distance is acceptable, store data
                        temp = pd.Series([(row2.x, row2.y, row2.z), dist, row2.ID], 
                                          index=cols, name=i2)
                        nearby = nearby.append(temp, ignore_index=True)
                if not nearby.empty: return nearby
            return None

        def __find_clusters():
            pass

        def __find_nearest():
            """For iterating the passed data to determine the nearest cells."""
            if 'targetXY' in locals():
                target = targetXY
                comment = settings.target_chan
                filename = 'Avg_DistanceMeans_{}_vs_{}.csv'.format(Data.name, 
                                                                  comment)
                rmv = False
            else:
                target = XYpos
                rmv = True
                comment = Data.name
                filename = 'Avg_DistanceMeans_{}.csv'.format(Data.name)
            cols = ['Nearest_XYZ_{}'.format(comment),'Nearest_Dist_{}'.format(
                    comment), 'Nearest_ID_{}'.format(comment)]
            pointData = pd.DataFrame(columns=cols, index=XYpos.index)
            # Iterate over each cell (row) in the data
            for i, row in XYpos.iterrows():
                nearby = __get_nearby(i, row, target, rmv_self=rmv, **kws)
                if nearby is not None:
                    nearest = nearby.Dist.idxmin()
                    pointData.loc[(i, cols)] = nearby.loc[nearest].values
            # Concatenate the obtained data with the read data.        
            NewData = pd.concat([Data, pointData], axis=1)
            # Get bin and distance to nearest cell for each cell, then calculate
            # average distance within each bin.
            binnedData = NewData.loc[:, 'DistBin']
            distances = NewData.loc[:, cols[1]].astype('float64')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                means = [np.nanmean(distances[binnedData.values==k]) for k in 
                         np.arange(0, len(settings.projBins))]
            return NewData, means, filename
        #--------#
        if volIncl > 0: # Subsetting of data based on cell volume
            dInd = self.subset_data(Data, compare, volIncl)
            if 'tData' in kws.keys(): # Obtain target channel if used.
                tData = kws.pop('tData')
                print(tData)
                tInd = self.subset_data(tData, compare, volIncl)
        elif 'tData' in kws.keys():
            tData = kws.pop('tData')
            tInd = tData.index
            dInd = Data.index
        else: dInd = Data.index
        # Accessing the data for the analysis via the indexes taken before.
        # Cells for which the nearest cells will be found:
        XYpos = Data.loc[dInd,['Position X', 'Position Y', 'Position Z', 
                               'ID','DistBin']]
        renames = {'Position X':'x', 'Position Y':'y', 'Position Z':'z'}
        XYpos.rename(columns=renames, inplace=True) # renaming for dot notation
        if 'tInd' in locals():  # Get data from target channel, if used
            targetXY = tData.loc[(tInd,
             ['Position X', 'Position Y', 'Position Z',
              'ID'])]
            targetXY.rename(columns=renames, inplace=True)
        if clusters == False:
            NewData, Means, filename = __find_nearest()
            Means = pd.Series(Means, name=(self.name))
            insert, _ = process.relate_data(Means, self.MP, self._center, self._length)
            SMeans = pd.Series(data=insert, name=(self.name))
            system.saveToFile(SMeans, self._dataDir, filename)
            OW_name = '{}.csv'.format(Data.name)
        else: # TODO add cluster finding
#               __find_clusters()
            pass
            # Overwrite the original data with the data containing new columns.
        system.saveToFile(NewData, (self.path), OW_name, append=False)


class statistics:
    def __init__(self, control, group2):
        """Takes two Group-objects and creates statistics based on their 
        normalized channel counts and additional data."""
        # Control group variables
        self.cntrlGroup = control.group
        self.cntrlNamer = control.namer
        self.cntrlSamples = control.groupPaths
        # Test group variables
        self.tstGroup = group2.group
        self.tstNamer = group2.namer
        self.tstSamples = control.groupPaths
        # Common / Stat object variables
        self.center = control._center
        self.length = control._length
        self.title = '{} VS. {}'.format(self.cntrlGroup, self.tstGroup)
        self.dataDir = control._dataDir
        self.statsDir = control._statsDir
        self.plotDir = control._plotDir.joinpath("Stat Plots")
        self.plotDir.mkdir(exist_ok=True)
        self.chanPaths = self.dataDir.glob('Norm_*')
        self.avgPaths = self.dataDir.glob('Avg_*')
        self.palette = {control.group: control.color, group2.group: group2.color}
        # Statistics and data
        self.statData = None
        self.cntrlData = None
        self.tstData = None

    def MWW_test(self, Path):
#        def __get_stats():
                  
        self.channel = ' '.join(str(Path.stem).split('_')[1:])
        Data = system.read_data(Path, header=0, test=False)
        cntrlData = Data.loc[:, Data.columns.str.contains(self.cntrlNamer, regex=True)]
        tstData = Data.loc[:, Data.columns.str.contains(self.tstNamer)]
        statData = pd.DataFrame(index=Data.index, columns=['Score Greater',
         'Corrected Greater', 'P Greater', 'Reject Greater',
         'Score Lesser', 'Corrected Lesser', 'P Lesser',
         'Reject Lesser'])
        # TODO try implementing rollin window
#        if settings.windowed:
#            for i, __ in cntrlData.loc[settings.trail-1:-(settings.lead+1), 
#                                       :].iterrows():
#                sInd = i - settings.trail
#                eInd = i + settings.lead
#                cntrlVals = cntrlData.loc[sInd:eInd, :].values().flatten()
#                tstVals = tstData.loc[sInd:eInd, :].values().flatten()
                
        for ind, row in cntrlData.iterrows():
            row2 = tstData.loc[ind, :]
            if row.any() == True or row2.any() == True:
                if np.array_equal(row.values, row2.values) == False:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        stat, pval = ss.mannwhitneyu(row, row2, 
                                                     alternative='greater')
                        stat2, pval2 = ss.mannwhitneyu(row, row2, 
                                                       alternative='less')
                    statData.iat[ind, 0], statData.iat[ind, 2] = stat, pval
                    statData.iat[ind, 4], statData.iat[ind, 6] = stat2, pval2
                else:
                    statData.iat[ind, 0], statData.iat[ind, 2] = 0, 0
                    statData.iat[ind, 4], statData.iat[ind, 6] = 0, 0
        # ??? Setting for multiple test correction or not?
        GrP = statData.iloc[:, 2].values
        LsP = statData.iloc[:, 6].values
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            RejectGr, CorrPGr, _, _ = multi.multipletests(GrP, method='fdr_bh', 
                                                      alpha=settings.alpha)
            RejectLs, CorrPLs, _, _ = multi.multipletests(LsP, method='fdr_bh', 
                                                      alpha=settings.alpha)
        statData.iloc[:, 3], statData.iloc[:, 1] = RejectGr, CorrPGr
        statData.iloc[:, 7], statData.iloc[:, 5] = RejectLs, CorrPLs
        filename = 'Stats_{} = {}.csv'.format(self.title, self.channel)
        system.saveToFile(statData, self.statsDir, filename, append=False)
        self.statData, self.cntrlData, self.tstData = statData, cntrlData, tstData
        return self
    
    def Create_Plots(self, stats, unit="Count", palette = None):
        if settings.Drop_Outliers:
            cntrlData = DropOutlier(self.cntrlData)
            tstData = DropOutlier(self.tstData)
        else:
            cntrlData = self.cntrlData
            tstData = self.tstData
        cntrlData.loc['Sample Group', :] = self.cntrlGroup
        tstData.loc['Sample Group', :] =  self.tstGroup
        plotData = pd.concat([cntrlData.T, tstData.T], ignore_index=True)
        plot_maker = plotter(plotData, savepath=self.plotDir, 
                     title=self.plottitle, palette=palette, center=self.center)
        kws = {'id_str':'Sample Group', 'hue':'Sample Group', 'height':5, 
               'aspect':4, 'var_str':'Longitudinal Position', 'value_str':unit, 
               'centerline':plot_maker.MPbin, 'xlen':self.length,
               'title':plot_maker.title, 'Stats': stats, 
               'fliersize': {'fliersize':'4'}}
        plot_maker.plot_Data(plotter.catPlot, plot_maker.savepath, **kws)
        
def DropOutlier(Data):
    with warnings.catch_warnings(): # Ignore warnings regarding empty bins
        warnings.simplefilter('ignore', category=RuntimeWarning)
        Mean = np.nanmean(Data.values)
        std = np.nanstd(Data.values)
        Data = Data.applymap(lambda x: x if np.abs(x - Mean) <= \
                             (settings.dropSTD * std) else np.nan)
    return Data
        
        
