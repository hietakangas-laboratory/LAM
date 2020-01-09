# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""
# LAM modules
from settings import settings as Sett
from plot import plotter
import system
import analysis
# Standard libraries
import warnings
import re
# Other packages
import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.stats.multitest as multi


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
        self.tstSamples = group2.groupPaths
        # Common / Stat object variables
        self.center = control._center
        self.length = control._length
        self.title = '{} VS. {}'.format(self.cntrlGroup, self.tstGroup)
        self.dataDir = control._dataDir
        self.statsDir = control._statsDir
        self.plotDir = control._plotDir.joinpath("Stat Plots")
        self.plotDir.mkdir(exist_ok=True)
        self.chanPaths = self.dataDir.glob('Norm_*')  # Cell counts
        self.avgPaths = self.dataDir.glob('Avg_*')  # Additional data avgs
        self.clPaths = self.dataDir.glob('ClNorm_*')  # Cluster data
        self.palette = {control: control.color, group2.group: group2.color}
        # Statistics and data
        self.statData = None
        self.cntrlData = None
        self.tstData = None
        self.order = [self.cntrlGroup, self.tstGroup]
        self.error = False

    def MWW_test(self, Path):
        def __get_stats(row, row2, ind, statData):
            if row.any() or row2.any():
                if not (np.array_equal(np.unique(row), np.unique(row2)) and
                        len(np.unique(row)) > 1 and len(np.unique(row2)) > 1):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',
                                              category=RuntimeWarning)
                        stat, pval = ss.mannwhitneyu(row, row2,
                                                     alternative='greater')
                        __, pval2 = ss.mannwhitneyu(row, row2,
                                                    alternative='less')
                        __, pval3 = ss.mannwhitneyu(row, row2,
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
                Reject, CorrP, _, _ = multi.multipletests(vals,
                                                          method='fdr_bh',
                                                          alpha=Sett.alpha)
            statData.iloc[:, corrInd], statData.iloc[:, rejInd] = CorrP, Reject
            return statData

        self.error = False
        self.channel = ' '.join(str(Path.stem).split('_')[1:])
        Data = system.read_data(Path, header=0, test=False)
        # NaN-values changed to zero in order to allow statistical comparison
        nulData = Data.replace(np.nan, 0)
        if nulData.nunique().nunique() == 1:
            print("-> {}: No data, passed.".format(self.channel))
            self.error = True
            return self
        Cntrlreg = re.compile("^{}".format(self.cntrlNamer), re.I)
        tstreg = re.compile("^{}".format(self.tstNamer), re.I)
        cntrlData = nulData.loc[:, Data.columns.str.contains(Cntrlreg,
                                                             regex=True)]
        tstData = nulData.loc[:, Data.columns.str.contains(tstreg, regex=True)]
        statData = pd.DataFrame(
            index=Data.index, columns=['U Score', 'Corr. Greater', 'P Greater',
                                       'Reject Greater', 'Corr. Lesser',
                                       'P Lesser', 'Reject Lesser',
                                       'Corr. Two-sided', 'P Two-sided',
                                       'Reject Two-sided'])
        if Sett.windowed:
            for ind, __ in cntrlData.iloc[Sett.trail:-(Sett.lead+1),
                                          :].iterrows():
                sInd = ind - Sett.trail
                eInd = ind + Sett.lead
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
        # Slice data again to have NaN-values where data doesn't exist
        cntrlData = Data.loc[:, Data.columns.str.contains(Cntrlreg,
                                                          regex=True)]
        tstData = Data.loc[:, Data.columns.str.contains(tstreg, regex=True)]
        self.statData = statData
        self.cntrlData, self.tstData = cntrlData, tstData
        return self

    def Create_Plots(self, stats, unit="Count", palette=None):
        if Sett.Drop_Outliers:
            cntrlData = analysis.DropOutlier(self.cntrlData)
            tstData = analysis.DropOutlier(self.tstData)
        else:
            cntrlData = self.cntrlData
            tstData = self.tstData
        cntrlData.loc['Sample Group', :] = self.cntrlGroup
        tstData.loc['Sample Group', :] = self.tstGroup
        plotData = pd.concat([cntrlData.T, tstData.T], ignore_index=True)
        plot_maker = plotter(plotData, savepath=self.plotDir,
                             title=self.plottitle, palette=palette,
                             center=self.center)
        kws = {'id_str': 'Sample Group', 'hue': 'Sample Group', 'height': 5,
               'aspect': 4, 'var_str': 'Longitudinal Position',
               'value_str': unit, 'centerline': plot_maker.MPbin,
               'xlen': self.length, 'title': plot_maker.title, 'Stats': stats,
               'title_y': 1, 'fliersize': {'fliersize': '1'}}
        if Sett.windowed:
            kws.update({'windowed': True})
        plot_maker.order = self.order
        plot_maker.plot_Data(plotter.catPlot, plot_maker.savepath, **kws)


class Total_Stats:
    def __init__(self, path, groups, plotDir, statsdir, palette=None):
        self.dataerror = False
        self.plotDir = plotDir
        self.statsDir = statsdir
        self.filename = path.stem
        self.data = system.read_data(path, header=0, test=False, index_col=0)
        if self.data.empty:
            self.dataerror = True
            return
        self.groups = groups
        self.channels = self.data.index.tolist()
        self.cntrlGrp = Sett.cntrlGroup
        self.tstGroups = [g for g in groups if g != self.cntrlGrp]
        self.palette = palette

    def stats(self):
        # TODO handle new data structure
        grpData = self.data.groupby(['Sample Group'])
        cntrlData = grpData.get_group(self.cntrlGrp)
        # cntrlData = self.data.loc[:, ('Sample Group' == namer)]
        cols = ['U Score', 'P Two-sided', 'Reject Two-sided']
        mcols = pd.MultiIndex.from_product([self.tstGroups, cols],
                                           names=['Sample Group',
                                                  'Statistics'])
        variables = self.data.Variable.unique()
        TotalStats = pd.DataFrame(index=variables, columns=mcols)
        TotalStats.sort_index(level=['Sample Group', 'Statistics'],
                              inplace=True)
        for grp in self.tstGroups:
            tstData = grpData.get_group(grp)
            # tstData = self.data.loc[:, self.data.columns.str.contains(namer)]
            for var in variables:
                cVals = cntrlData.loc[(cntrlData.Variable == var),
                                      cntrlData.columns.difference(
                                          ['Sample Group', 'Variable'])]
                tVals = tstData.loc[(tstData.Variable == var),
                                    tstData.columns.difference(
                                        ['Sample Group', 'Variable'])]
                stat, pval = ss.mannwhitneyu(cVals.to_numpy().flatten(),
                                             tVals.to_numpy().flatten(),
                                             alternative='two-sided')
                if pval < Sett.alpha:
                    reject = True
                else:
                    reject = False
                TotalStats.loc[var, (grp, cols)] = [stat, pval, reject]
        self.savename = self.filename + ' Stats.csv'
        system.saveToFile(TotalStats, self.statsDir, self.savename,
                          append=False, w_index=True)
        self.statData = TotalStats

    def Create_Plots(self):
        plotData = self.data
        cntrlN = int(len(self.groups) / 2)
        order = self.tstGroups
        order.insert(cntrlN, self.cntrlGrp)
        # for namer in namers:
        #     plotData.loc['Sample Group', plotData.columns.str.startswith(
        #                                         namer)] = namer.split('_')[0]
        plot_maker = plotter(plotData, self.plotDir, center=0,
                             title=self.filename, palette=self.palette,
                             color='b')
        plot_maker.total_plot(self.statData, order)
