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
from src.settings import Settings as Sett
import src.system as system


class VersusStats:
    """Find bin-wise MWW statistics between sample groups."""

    def __init__(self, control, group2):
        """Create statistics for two Group-objects, i.e. sample groups."""
        # Sample groups
        self.ctrl_grp = control.group
        self.test_grp = group2.group

        # Common / Stat object variables
        self.title = '{} VS. {}'.format(self.ctrl_grp, self.test_grp)
        self.stat_dir = control.paths.statsdir
        self.plot_dir = control.paths.plotdir.joinpath("Stats")
        self.plot_dir.mkdir(exist_ok=True)

        # Statistics and data
        self.stat_data = None
        self.ctrl_data = None
        self.test_data = None
        self.error = False
        self.channel = ""

    def mww_test(self, channel_path):
        """Perform MWW-test for a data set of two groups."""
        self.error = False
        self.channel = ' '.join(str(channel_path.stem).split('_')[1:])
        data = system.read_data(channel_path, header=0, test=False)

        # Test that data exists and has non-zero numeric values
        cols = data.any().index
        valid_data = data.loc[:, cols]
        valid_grp_n = cols.map(lambda x: str(x).split('_')[0]).unique().size

        if not valid_data.any().any() or valid_grp_n < 2:
            self.error = True

        # Find group-specific data
        grp_data = valid_data.T.groupby(lambda x: str(x).split('_')[0])
        try:
            self.ctrl_data = grp_data.get_group(self.ctrl_grp).T
            self.test_data = grp_data.get_group(self.test_grp).T
        except KeyError:  # If sample group not found, i.e. no sample has data
            self.error = True

        if self.error:
            print(f"WARNING: {self.channel} - Insufficient data, skipped.")

        stat_cols = ['U Score', 'Corr. Greater', 'P Greater', 'Reject Greater', 'Corr. Lesser', 'P Lesser',
                     'Reject Lesser', 'Corr. Two-sided', 'P Two-sided', 'Reject Two-sided']
        stat_data = pd.DataFrame(index=data.index, columns=stat_cols)

        if Sett.windowed:  # If doing rolling window stats
            stat_data = self.windowed_test(stat_data)

        else:  # Bin-by-bin stats:
            stat_data = self.bin_test(stat_data)

        # Correct for multiple testing:
        stat_data = correct(stat_data, stat_data.iloc[:, 2], 1, 3)  # greater
        stat_data = correct(stat_data, stat_data.iloc[:, 5], 4, 6)  # lesser
        stat_data = correct(stat_data, stat_data.iloc[:, 8], 7, 9)  # 2-sided

        # Save statistics
        filename = f'Stats_{self.title} = {self.channel}.csv'
        system.save_to_file(stat_data, self.stat_dir, filename, append=False)
        self.stat_data = stat_data

    def bin_test(self, stat_data):
        """Perform MWW test bin-to-bin."""
        for ind, row in self.ctrl_data.iterrows():  # Loop bins
            # Drop missing values from data
            ctrl_vals = row.dropna().values
            tst_vals = self.test_data.loc[ind, :].dropna().values
            # Test values at current bin
            stat_data = get_stats(ctrl_vals, tst_vals, ind, stat_data)
        return stat_data

    def windowed_test(self, stat_data):
        """Perform windowed MWW test."""
        trail, lead = Sett.trail, Sett.lead
        for idx, __ in self.ctrl_data.iloc[trail:-lead, :].iterrows():

            # Define window edges
            s_ind = idx - trail
            e_ind = idx + lead

            # Get values from both sample groups:
            ctrl_vals = self.ctrl_data.iloc[s_ind:e_ind, :].values.flatten()
            ctrl_vals = ctrl_vals[~np.isnan(ctrl_vals)]
            tst_vals = self.test_data.iloc[s_ind:e_ind, :].values.flatten()
            tst_vals = tst_vals[~np.isnan(tst_vals)]

            # Compare values
            stat_data = get_stats(ctrl_vals, tst_vals, idx, stat_data)
        return stat_data


class TotalStats:
    """Find statistics based on sample-specific totals."""

    def __init__(self, path, groups, plot_dir, stat_dir):
        self.dataerror = False
        self.error_vars = {}
        self.plot_dir = plot_dir
        self.stat_dir = stat_dir
        self.filename = path.stem
        self.data = system.read_data(path, header=0, test=False, index_col=0)

        # Test that data exists
        if self.data is None or self.data.empty:
            self.dataerror = True

        self.groups = groups
        self.test_grps = [g for g in groups if g != Sett.cntrlGroup]
        self.stat_data = None

    def stats(self):
        """Calculate statistics of one variable between two groups."""
        # Group all data by sample groups
        grp_data = self.data.groupby(['Sample Group'])

        # Find data of control group
        ctrl_data = grp_data.get_group(Sett.cntrlGroup)

        # Make indices for DataFrame
        cols = ['U Score', 'P Two-sided', 'Reject Two-sided']  # Needed columns
        mcol = pd.MultiIndex.from_product([self.test_grps, cols], names=['Sample Group', 'Statistics'])
        variables = self.data.Variable.unique()  # Index

        # Create the DataFrame
        total_stats = pd.DataFrame(index=variables, columns=mcol)
        total_stats.sort_index(level=['Sample Group', 'Statistics'], inplace=True)

        # Test each group against the control:
        for grp in self.test_grps:
            test_data = grp_data.get_group(grp)

            # Loop all variables to test
            for variable in variables:
                # Get data of both groups
                c_vals = ctrl_data.loc[(ctrl_data.Variable == variable),
                                       ctrl_data.columns.difference(['Sample Group', 'Variable'])]
                t_vals = test_data.loc[(test_data.Variable == variable),
                                       test_data.columns.difference(['Sample Group', 'Variable'])]

                # Perform test
                test_values = self.total_mww(grp, c_vals, t_vals, variable)

                # Insert values to result DF
                total_stats.loc[variable, (grp, cols)] = test_values

            if grp in self.error_vars.keys():
                print(f"WARNING: {self.filename} - No data on {', '.join(self.error_vars[grp])}")

        # Save statistics
        savename = self.filename + ' Stats.csv'
        system.save_to_file(total_stats, self.stat_dir, savename, append=False, w_index=True)

        # Store to object
        self.stat_data = total_stats

    def total_mww(self, grp, c_vals, t_vals, variable):
        """Perform MWW-test for group totals of a variable."""

        # Flatten matrix and drop missing values
        c_vals = c_vals.to_numpy().flatten()
        t_vals = t_vals.to_numpy().flatten()
        c_vals = c_vals[~np.isnan(c_vals)]
        t_vals = t_vals[~np.isnan(t_vals)]

        try:  # MWW test:
            stat, pval = ss.mannwhitneyu(c_vals, t_vals, alternative='two-sided')
            reject = bool(pval < Sett.alpha)

        except ValueError:
            if grp not in self.error_vars.keys():
                self.error_vars.update({grp: [variable]})
            else:
                self.error_vars[grp].append(variable)
            return [0, 0, 0]

        return [stat, pval, reject]


def get_stats(row, row2, ind, stat_data):
    """Compare respective bins of both groups."""
    unqs = np.unique(np.hstack((row, row2))).size

    # If data rows are different, get stats
    if (row.any() and row2.any()) and not np.array_equal(np.unique(row), np.unique(row2)) and unqs > 1:

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            # Whether ctrl is greater
            stat, pval = ss.mannwhitneyu(row, row2, alternative='greater')
            stat_data.iat[ind, 0], stat_data.iat[ind, 2] = stat, pval
            # Whether ctrl is lesser
            __, pval = ss.mannwhitneyu(row, row2, alternative='less')
            stat_data.iat[ind, 5] = pval
            # Whether significant difference exists
            __, pval = ss.mannwhitneyu(row, row2, alternative='two-sided')
            stat_data.iat[ind, 8] = pval

    else:  # If rows are same. input zeros
        stat_data.iat[ind, 0], stat_data.iat[ind, 2] = 0, 0
        stat_data.iat[ind, 5] = 0
        stat_data.iat[ind, 8] = 0
    return stat_data


def correct(stat_data, p_vals, corr_ind, rej_ind):
    """Correct for multipletesting."""

    vals = p_vals.values  # Get P-values

    with warnings.catch_warnings():  # Correct
        warnings.simplefilter('ignore', category=RuntimeWarning)
        reject, corr_p, _, _ = multi.multipletests(vals, method='fdr_bh', alpha=Sett.alpha)

    # Add corrected values to DF
    stat_data.iloc[:, corr_ind], stat_data.iloc[:, rej_ind] = corr_p, reject
    return stat_data
