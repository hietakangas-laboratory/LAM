# -*- coding: utf-8 -*-
from settings import settings
from plot import plotter
import system, analysis
import numpy as np, pandas as pd
import scipy.stats as ss, statsmodels.stats.multitest as multi
import copy
import warnings

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
        def __get_stats(row, row2, ind, statData):
            if row.any() != False or row2.any() != False:
                if np.array_equal(row, row2) == False:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        stat, pval = ss.mannwhitneyu(row, row2, 
                                                     alternative='greater')
                        stat2, pval2 = ss.mannwhitneyu(row, row2, 
                                                       alternative='less')
                        stat3, pval3 = ss.mannwhitneyu(row, row2, 
                                                       alternative='two-sided')
                    statData.iat[ind, 0], statData.iat[ind, 2] = stat, pval
                    statData.iat[ind, 5] = pval2
                    statData.iat[ind, 8] = pval3
                else:
                    statData.iat[ind, 0], statData.iat[ind, 2] = 0, 0
                    statData.iat[ind, 5] = 0
                    statData.iat[ind, 8] = 0
            return statData
        
        def __correct(Pvals, corrInd, rejInd):
            vals = Pvals.values
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                Reject, CorrP, _, _ = multi.multipletests(vals, method='fdr_bh', 
                                                      alpha=settings.alpha)
            statData.iloc[:,corrInd], statData.iloc[:, rejInd] = CorrP, Reject
            return statData
                  
        self.channel = ' '.join(str(Path.stem).split('_')[1:])
        Data = system.read_data(Path, header=0, test=False)
        Data = Data.replace(np.nan, 0)
        cntrlData = Data.loc[:, Data.columns.str.contains(self.cntrlNamer, regex=True)]
        tstData = Data.loc[:, Data.columns.str.contains(self.tstNamer, regex=True)]
        statData = pd.DataFrame(index=Data.index, columns=['U Score',
         'Corr. Greater', 'P Greater', 'Reject Greater', 'Corr. Lesser', 
         'P Lesser', 'Reject Lesser', 'Corr. Two-sided', 'P Two-sided', 
         'Reject Two-sided'])
        if settings.windowed:
            for ind, __ in cntrlData.iloc[settings.trail:-(settings.lead+1), 
                                       :].iterrows():
                sInd = ind - settings.trail
                eInd = ind + settings.lead
                cntrlVals = cntrlData.iloc[sInd:eInd, :].values.flatten()
                tstVals = tstData.iloc[sInd:eInd, :].values.flatten()
                statData = __get_stats(cntrlVals, tstVals, ind, statData)
        else:        
            for ind, row in cntrlData.iterrows():
                cntrlVals = row.values
                tstVals = tstData.loc[ind, :].values
                statData = __get_stats(cntrlVals, tstVals, ind, statData)
        statData = __correct(statData.iloc[:, 2], 1, 3)
        statData = __correct(statData.iloc[:, 5], 4, 6)
        statData = __correct(statData.iloc[:, 8], 7, 9)
        filename = 'Stats_{} = {}.csv'.format(self.title, self.channel)
        system.saveToFile(statData, self.statsDir, filename, append=False)
        self.statData, self.cntrlData, self.tstData = statData, cntrlData, tstData
        return self
    
    def Create_Plots(self, stats, unit="Count", palette = None):
        if settings.Drop_Outliers:
            cntrlData = analysis.DropOutlier(self.cntrlData)
            tstData = analysis.DropOutlier(self.tstData)
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
               'title':plot_maker.title, 'Stats': stats, 'title_y':1, 
               'fliersize': {'fliersize':'2'}}
        if settings.windowed: kws.update({'windowed': True})
        plot_maker.plot_Data(plotter.catPlot, plot_maker.savepath, **kws)
        
class Total_Stats:
    def __init__(self, path, groups, plotDir, statsdir, palette=None):
        self.plotDir = plotDir
        self.statsDir = statsdir
        self.data = system.read_data(path, header=0, test=False, index_col=0)
        self.groups = copy.deepcopy(groups)
        self.channels = self.data.index.tolist()
        self.cntrlGrp = settings.cntrlGroup
        groups.remove(self.cntrlGrp)
        self.tstGroups = groups
        self.palette = palette
        
    def stats(self):
        namer = "{}_".format(self.cntrlGrp)
        cntrlData = self.data.loc[:, self.data.columns.str.contains(namer)]
        cols = ['U Score', 'P Two-sided', 'Reject Two-sided']
        mcols = pd.MultiIndex.from_product([self.tstGroups, cols], 
                                           names=['Sample Group', 'Statistics'])
        TotalStats = pd.DataFrame(index=cntrlData.index, columns=mcols)
        for grp in self.tstGroups:
            namer = "{}_".format(grp)
            tstData = self.data.loc[:, self.data.columns.str.contains(namer)]
            for i, row in cntrlData.iterrows():
                row2 = tstData.loc[i, :]
                stat, pval = ss.mannwhitneyu(row, row2,alternative='two-sided')
                if pval < settings.alpha:
                    reject = True
                else:
                    reject = False
                TotalStats.loc[i, (grp, cols)] = [stat, pval, reject]
        system.saveToFile(TotalStats, self.statsDir, "Total Count Stats.csv", 
                          append=False, w_index=True)
        return TotalStats            
    
    def Create_Plots(self, stats):
        namers = ['{}_'.format(g) for g in self.groups]
        plotData = self.data
        cntrlN = int(len(self.groups) /2)
        order = self.tstGroups
        order.insert(cntrlN, self.cntrlGrp)
        for namer in namers:
            plotData.loc['Sample Group', plotData.columns.str.contains(
                                                namer)] = namer.split('_')[0]
        plot_maker = plotter(plotData, self.plotDir, center=0, title='Total Counts', 
                 palette=self.palette, color='b')
        plot_maker.total_plot(stats, order)