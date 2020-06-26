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
# from plot import plotter
import system


class statistics:
    """Find bin-wise MWW statistics between sample groups."""

    def __init__(self, control, group2):
        """Create statistics for two Group-objects, i.e. sample groups."""
        # Sample groups
        self.ctrlGroup = control.group
        self.tstGroup = group2.group
        # Common / Stat object variables
        self.title = '{} VS. {}'.format(self.ctrlGroup, self.tstGroup)
        self.statsDir = control.paths.statsdir
        self.plotDir = control.paths.plotdir.joinpath("Stats")
        self.plotDir.mkdir(exist_ok=True)
        # Statistics and data
        self.statData = None
        self.ctrlData = None
        self.tstData = None
        self.error = False
        self.channel = ""

    def MWW_test(self, Path):
        """Perform MWW-test for a data set of two groups."""
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
        if Sett.windowed:  # If doing rolling window stats
            for ind, __ in self.ctrlData.iloc[Sett.trail:-Sett.lead,
                                              :].iterrows():
                sInd = ind - Sett.trail # Window edges
                eInd = ind + Sett.lead
                # Get values from both sample groups:
                ctrlVals = self.ctrlData.iloc[sInd:eInd, :].values.flatten()
                ctrlVals = ctrlVals[~np.isnan(ctrlVals)]
                tstVals = self.tstData.iloc[sInd:eInd, :].values.flatten()
                tstVals = tstVals[~np.isnan(tstVals)]
                # Compare values
                statData = get_stats(ctrlVals, tstVals, ind, statData)
        else:  # Bin-by-bin stats:
            for ind, row in self.ctrlData.iterrows():
                ctrlVals = row.dropna().values
                tstVals = self.tstData.loc[ind, :].dropna().values
                statData = get_stats(ctrlVals, tstVals, ind, statData)
        # Correct for multiple testing:
        statData = correct(statData, statData.iloc[:, 2], 1, 3)  # greater
        statData = correct(statData, statData.iloc[:, 5], 4, 6)  # lesser
        statData = correct(statData, statData.iloc[:, 8], 7, 9)  # 2-sided
        # Save statistics
        filename = 'Stats_{} = {}.csv'.format(self.title, self.channel)
        system.saveToFile(statData, self.statsDir, filename, append=False)
        self.statData = statData
        return self


class Total_Stats:
    """Find statistics based on sample-specific totals."""

    def __init__(self, path, groups, plotDir, statsdir):
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
        self.statData = None

    def stats(self):
        """Calculate statistics of one variable between two groups."""
        # Group all data by sample groups
        grpData = self.data.groupby(['Sample Group'])
        # Find data of control group
        ctrlData = grpData.get_group(Sett.cntrlGroup)
        # Make a DataFrame for results
        cols = ['U Score', 'P Two-sided', 'Reject Two-sided']  # Needed columns
        mcols = pd.MultiIndex.from_product([self.tstGroups, cols],
                                           names=['Sample Group',
                                                  'Statistics'])
        variables = self.data.Variable.unique()  # Find analyzable variables
        TotalStats = pd.DataFrame(index=variables, columns=mcols)  # create DF
        TotalStats.sort_index(level=['Sample Group', 'Statistics'],
                              inplace=True)
        # Test each group against the control:
        for grp in self.tstGroups:
            tstData = grpData.get_group(grp)
            for var in variables:  # Test all found variables:
                # Get data of both groups
                cVals = ctrlData.loc[(ctrlData.Variable == var),
                                     ctrlData.columns.difference(
                                         ['Sample Group', 'Variable'])]
                tVals = tstData.loc[(tstData.Variable == var),
                                    tstData.columns.difference(
                                        ['Sample Group', 'Variable'])]
                try:  # MWW test:
                    stat, pval = ss.mannwhitneyu(cVals.to_numpy().flatten(),
                                                 tVals.to_numpy().flatten(),
                                                 alternative='two-sided')
                    reject = bool(pval < Sett.alpha)
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
                # Insert values to result DF
                TotalStats.loc[var, (grp, cols)] = [stat, pval, reject]
        # Save statistics
        savename = self.filename + ' Stats.csv'
        system.saveToFile(TotalStats, self.statsDir, savename,
                          append=False, w_index=True)
        self.statData = TotalStats


def get_stats(row, row2, ind, statData):
    """Compare respective bins of both groups."""
    unqs = np.unique(np.hstack((row, row2))).size
    if ((row.any() or row2.any()) and not np.array_equal(
            np.unique(row), np.unique(row2)) and unqs > 1):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            # Whether ctrl is greater
            stat, pval = ss.mannwhitneyu(row, row2, alternative='greater')
            statData.iat[ind, 0], statData.iat[ind, 2] = stat, pval
            # Whether ctrl is lesser
            __, pval = ss.mannwhitneyu(row, row2, alternative='less')
            statData.iat[ind, 5] = pval
            # Whether significant difference exists
            __, pval = ss.mannwhitneyu(row, row2, alternative='two-sided')
            statData.iat[ind, 8] = pval
    else:
        statData.iat[ind, 0], statData.iat[ind, 2] = 0, 0
        statData.iat[ind, 5] = 0
        statData.iat[ind, 8] = 0
    return statData


def correct(statData, Pvals, corrInd, rejInd):
    """Perform multipletest correction."""
    vals = Pvals.values
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        Reject, CorrP, _, _ = multi.multipletests(vals, method='fdr_bh',
                                                  alpha=Sett.alpha)
    statData.iloc[:, corrInd], statData.iloc[:, rejInd] = CorrP, Reject
    return statData
