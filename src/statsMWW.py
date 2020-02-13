# -*- coding: utf-8 -*-
"""
LAM-module for handling all statistical calculations.

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""
# Standard libraries
import warnings
# Other packages
import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.stats.multitest as multi
# LAM modules
from settings import settings as Sett
from plot import plotter
import system
import analysis


class statistics:
    """Find bin-wise MWW statistics between sample groups."""

    def __init__(self, control, group2):
        """Create statistics for two Group-objects, i.e. sample groups."""
        # Sample groups
        self.ctrlGroup = control.group
        self.tstGroup = group2.group
        # Common / Stat object variables
        self.center = control._center
        self.length = control._length
        self.title = '{} VS. {}'.format(self.ctrlGroup, self.tstGroup)
        self.statsDir = control._statsDir
        self.plotDir = control._plotDir.joinpath("Stat Plots")
        self.plotDir.mkdir(exist_ok=True)
        self.chanPaths = control._dataDir.glob('Norm_*')  # Cell counts
        self.avgPaths = control._dataDir.glob('Avg_*')  # Additional data avgs
        self.clPaths = control._dataDir.glob('ClNorm_*')  # Cluster data
        self.palette = {control: control.color, group2.group: group2.color}
        # Statistics and data
        self.statData = None
        self.ctrlData = None
        self.tstData = None
        self.error = False
        self.channel = ""

    def MWW_test(self, Path):
        """Perform MWW-test for a data set of two groups."""
        def __get_stats(row, row2, ind, statData):
            """Compare respective bins of both groups."""
            unqs = np.unique(np.hstack((row, row2))).size
            if ((row.any() or row2.any()) and not np.array_equal(
                    np.unique(row), np.unique(row2)) and unqs > 1):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    # Whether ctrl is greater
                    stat, pval = ss.mannwhitneyu(row, row2,
                                                 alternative='greater')
                    statData.iat[ind, 0], statData.iat[ind, 2] = stat, pval
                    # Whether ctrl is lesser
                    __, pval = ss.mannwhitneyu(row, row2, alternative='less')
                    statData.iat[ind, 5] = pval
                    # Whether significant difference exists
                    __, pval = ss.mannwhitneyu(row, row2,
                                               alternative='two-sided')
                    statData.iat[ind, 8] = pval
            else:
                statData.iat[ind, 0], statData.iat[ind, 2] = 0, 0
                statData.iat[ind, 5] = 0
                statData.iat[ind, 8] = 0
            return statData

        def __correct(Pvals, corrInd, rejInd):
            """Perform multipletest correction."""
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
        # Test that data exists and has non-zero, numeric values
        cols = Data.any().index
        validData = Data.loc[:, cols]
        validGrpN = cols.map(lambda x: str(x).split('_')[0]).unique().size
        if validData.empty or validGrpN < 2:
            print("-> {}: Insufficient data, passed.".format(self.channel))
            self.error = True
            return self
        # Find group-specific data
        grpData = validData.T.groupby(lambda x: str(x).split('_')[0])
        self.ctrlData = grpData.get_group(self.ctrlGroup).T
        self.tstData = grpData.get_group(self.tstGroup).T
        statCols = ['U Score', 'Corr. Greater', 'P Greater', 'Reject Greater',
                    'Corr. Lesser', 'P Lesser', 'Reject Lesser',
                    'Corr. Two-sided', 'P Two-sided', 'Reject Two-sided']
        statData = pd.DataFrame(index=Data.index, columns=statCols)
        if Sett.windowed:
            for ind, __ in self.ctrlData.iloc[Sett.trail:-Sett.lead,
                                              :].iterrows():
                sInd = ind - Sett.trail
                eInd = ind + Sett.lead
                ctrlVals = self.ctrlData.iloc[sInd:eInd, :].values.flatten()
                ctrlVals = ctrlVals[~np.isnan(ctrlVals)]
                tstVals = self.tstData.iloc[sInd:eInd, :].values.flatten()
                tstVals = tstVals[~np.isnan(tstVals)]
                statData = __get_stats(ctrlVals, tstVals, ind, statData)
        else:
            for ind, row in self.ctrlData.iterrows():
                ctrlVals = row.dropna().values
                tstVals = self.tstData.loc[ind, :].dropna().values
                statData = __get_stats(ctrlVals, tstVals, ind, statData)
        statData = __correct(statData.iloc[:, 2], 1, 3)
        statData = __correct(statData.iloc[:, 5], 4, 6)
        statData = __correct(statData.iloc[:, 8], 7, 9)
        filename = 'Stats_{} = {}.csv'.format(self.title, self.channel)
        system.saveToFile(statData, self.statsDir, filename, append=False)
        self.statData = statData
        return self

    def Create_Plots(self, stats, unit="Count", palette=None):
        """Handle statistical data for plots."""
        if Sett.Drop_Outliers:
            ctrlData = analysis.DropOutlier(self.ctrlData)
            tstData = analysis.DropOutlier(self.tstData)
        else:
            ctrlData = self.ctrlData
            tstData = self.tstData
        ctrlData.loc['Sample Group', :] = self.ctrlGroup
        tstData.loc['Sample Group', :] = self.tstGroup
        plotData = pd.concat([ctrlData.T, tstData.T], ignore_index=True)
        plot_maker = plotter(plotData, savepath=self.plotDir,
                             title=self.plottitle, palette=palette,
                             center=self.center)
        kws = {'id_str': 'Sample Group', 'hue': 'Sample Group', 'height': 4,
               'aspect': 3, 'var_str': 'Longitudinal Position',
               'value_str': unit, 'centerline': plot_maker.MPbin,
               'xlen': self.length, 'title': plot_maker.title, 'Stats': stats,
               'title_y': 1, 'fliersize': {'fliersize': '1'}}
        if Sett.windowed:
            kws.update({'windowed': True})
        plot_maker.order = [self.ctrlGroup, self.tstGroup]
        plot_maker.plot_Data(plotter.catPlot, plot_maker.savepath, **kws)


class Total_Stats:
    """Find statistics based on sample-specific totals."""

    def __init__(self, path, groups, plotDir, statsdir, palette=None):
        self.dataerror = False
        self.errorVars = {}
        self.plotDir = plotDir
        self.statsDir = statsdir
        self.filename = path.stem
        self.data = system.read_data(path, header=0, test=False, index_col=0)
        if self.data is None or self.data.empty:  # Test that data is fine
            self.dataerror = True
        self.groups = groups
        self.tstGroups = [g for g in groups if g != Sett.cntrlGroup]
        self.palette = palette
        self.savename = ""
        self.statData = None

    def stats(self):
        """Calculate statistics of one variable between two groups."""
        grpData = self.data.groupby(['Sample Group'])
        ctrlData = grpData.get_group(Sett.cntrlGroup)
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
            for var in variables:
                cVals = ctrlData.loc[(ctrlData.Variable == var),
                                     ctrlData.columns.difference(
                                         ['Sample Group', 'Variable'])]
                tVals = tstData.loc[(tstData.Variable == var),
                                    tstData.columns.difference(
                                        ['Sample Group', 'Variable'])]
                # if np.unique(np.vstack((cVals.to_numpy(),
                #                         tVals.to_numpy()))).size > 1:
                try:
                    stat, pval = ss.mannwhitneyu(cVals.to_numpy().flatten(),
                                                 tVals.to_numpy().flatten(),
                                                 alternative='two-sided')
                    reject = bool(pval < Sett.alpha)
                # else:
                except ValueError as e:
                    if str(e) == 'All numbers are identical in mannwhitneyu':
                        msg = 'Identical {}-values between control and {}'\
                            .format(var, grp)
                    else:
                        msg = 'ValueError for {}'.format(var)
                    print('WARNING: {}'.format(msg))
                    if grp not in self.errorVars.keys():
                        self.errorVars.update({grp: [var]})
                    else:
                        self.errorVars[grp].append(var)
                    continue
                    # stat, pval, reject = np.nan, np.nan, False
                TotalStats.loc[var, (grp, cols)] = [stat, pval, reject]
        self.savename = self.filename + ' Stats.csv'
        system.saveToFile(TotalStats, self.statsDir, self.savename,
                          append=False, w_index=True)
        self.statData = TotalStats

    def Create_Plots(self):
        """Handle statistical data for plotting."""
        plotData = self.data
        ctrlN = int(len(self.groups) / 2)
        order = self.tstGroups
        order.insert(ctrlN, Sett.cntrlGroup)
        plot_maker = plotter(plotData, self.plotDir, center=0,
                             title=self.filename, palette=self.palette,
                             color='b')
        plot_maker.total_plot(self.statData, order)
