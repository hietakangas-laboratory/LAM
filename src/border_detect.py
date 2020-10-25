# -*- coding: utf-8 -*-
"""
Detect region borders for for all sample groups through variable scoring.

Requires LAM-produced width data! On default settings, the functionality
also expects to find nearest distance estimates for the used channel, i.e.
'find distances' in LAM's GUI.
Created on Fri May  8 19:03:36 2020

@author: Arto Viitanen
"""

# Other packages
import numpy as np
from numpy.polynomial import Chebyshev
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import warnings

import tkinter as tk
import tkinter.simpledialog as tks

# LAM modules
from src.settings import Settings as Sett, store
import src.logger as lg
import src.system as system

LAM_logger = None


def detect_borders(paths, all_samples, palette, anchor, variables, scoring, threshold=0.5, channel='DAPI'):
    """
    Midgut border detection by weighted scoring of binned variables.

    Args:
    ----
        paths - LAM system.Paths-object that contains directory paths
        all_samples - Paths to sample folders
        palette - Color palette dict with sample groups as keys
        anchor - Anchoring bin of the samples in the full data matrix
        threshold - Minimum score for peak detection, i.e. borders
        variables - List of column names to collect from sample's channel data
        scoring - Dict of variable names with their scoring weight
        channel - The name of the data channel that is used, e.g. 'DAPI' data
    """
    print('\n---Finding border regions---')
    lg.logprint(LAM_logger, 'Finding border regions.', 'i')
    b_dirpath = plotting_directory(paths.plotdir)

    # Get widths and if not found, abort
    widths = read_widths(paths.datadir)
    if widths is None:
        return

    # Establish object to store scores of individual samples
    border_data = FullBorders(all_samples, widths, anchor, palette)
    print('  Scoring samples  ...')

    # Collect and score variables for each sample in the sample list
    for path in all_samples:
        sample = GetSampleBorders(path, channel, scoring, anchor, variables)
        # If expected variables are found, calculate sample scores
        if not sample.error:
            sample(border_data, b_dirpath)

    # If no data, return without finding borders
    if border_data.scores.isnull().values.all():
        print('\nERROR: Missing data, border detection cancelled.')
        lg.logprint(LAM_logger, 'Border detection variables not found.', 'e')
        return

    # Once sample scores have been collected, find peaks
    print('  Finding peaks  ...')
    flat, peaks = border_data(b_dirpath, threshold)

    # Add the locations of border peaks in each sample's individual binning
    binned_peaks = append_binning(border_data.sample_starts, peaks)

    # Save data
    flat.T.to_csv(paths.datadir.joinpath('Borders_scores.csv'), index=False)
    binned_peaks.to_csv(paths.datadir.joinpath('Borders_peaks.csv'), index=False)
    lg.logprint(LAM_logger, 'Border detection done.', 'i')


class FullBorders:
    """Store sample scores from GetSampleBorders and find group peaks."""

    def __init__(self, samples, widths, anchor, palette):
        self.samples = samples
        self.width_data = widths
        self.anchor = anchor * 2
        self.sample_starts = pd.Series()
        self.palette = palette
        self.scores = pd.DataFrame(columns=[p.name for p in samples], index=widths.index)

    def __call__(self, dirpath, threshold):
        # Fit a curve to sample data and get divergence of values
        flattened = self.flatten_scores()

        # Compute total scores of sample groups
        s_sums = get_group_total(flattened)
        s_sums.value = s_sums.groupby(s_sums.group).apply(lambda x: x.assign(value=norm_func(x.value)))

        # Find group peaks
        peaks = get_peak_data(s_sums, threshold)

        # Create plots
        if Sett.Create_Border_Plots & Sett.Create_Plots:
            print('  Creating border plot  ...')
            # Transform data to plottable format
            scores = prepare_data(self.scores.T)
            # Plot
            self.group_plots(scores, s_sums, peaks, dirpath)

        # Readjust peak locations and indexing to original bins
        peaks.peak = peaks.peak.divide(2)
        self.sample_starts = self.sample_starts.divide(2)
        return self.scores.T, peaks

    def flatten_scores(self):
        """Subtract fitted curve from values of each group."""
        # Add identifiers to data
        groups = [s[0] for s in self.scores.columns.str.split('_')]
        vals = self.scores.T.copy().assign(group=groups)
        # Group data based on sample groups
        grouped = vals.groupby('group')
        # Fit curve to values in each sample group
        curves = grouped.apply(lambda x: get_fit(x, x.name, id_var=['group']))
        # re-index
        curves = curves.droplevel(1)
        # Subtract curve from each sample
        devs = vals.apply(lambda x, c=curves: x[:-1].subtract(c.loc[x.group, :]), axis=1)
        return devs

    def group_plots(self, scores, s_sums, peaks, dirpath):
        """Create plots of sample group border scores."""
        # Create canvas
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={'height_ratios': [5, 5]})
        plt.subplots_adjust(hspace=0.75, top=0.85, bottom=0.1, left=0.1, right=0.85)

        # Plot smoothed summed scores
        sns.lineplot(data=s_sums.infer_objects(), x='variable', y='value', hue='group', palette=self.palette, alpha=0.7,
                     legend=False, ax=ax1)

        # Add peaks to sum score plot
        for peak in peaks.iterrows():
            loc = peak[1]['peak']
            prom = peak[1]['prominence']
            grp = peak[1]['group']
            c_val = s_sums.loc[(grp, loc)].value
            color = self.palette[grp]
            # add peak location lines with prominence
            ax1.vlines(x=loc, ymin=c_val-prom, ymax=c_val, color=color, linewidth=1, zorder=2)
            # Add peak location annotation
            ax1.annotate(int(loc/2), (loc + 0.03, c_val * 1.03), color=color, alpha=0.7)

        # Plot raw group scores
        sns.lineplot(data=scores, x='variable', y='value', hue='group', palette=self.palette, alpha=0.7, ax=ax2)

        # Readjust all plots' x-axis labels to original binning
        left, right = plt.xlim()
        locs, _ = plt.xticks()
        lbls = [int(n/2) for n in locs]
        for axis in (ax1, ax2):
            ybot, ytop = axis.get_ylim()
            # Add line to indicate anchoring bin
            axis.vlines(self.anchor, ybot, ytop, 'firebrick', zorder=0, linestyles='dashed')
            # Add peak detection threshold line
            axis.hlines(Sett.peak_thresh, xmin=left, xmax=right, linewidth=1, zorder=0, linestyles='dashed',
                        color='dimgrey')
            # Define labels and ticks
            axis.set_xticks(locs)
            axis.set_xticklabels(labels=lbls)
            axis.set_xlabel('')
            axis.set_ylabel('Score')

        # Change titles and legend location
        ax1.set_title('Scaled mean score')
        ax2.set_title('Raw group score')
        ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1)
        ax2.set_xlabel('Linear Position')
        plt.suptitle('GROUP BORDER SCORES')

        # Save figure
        plt.savefig(dirpath.joinpath(f'All-Border_Scores.{Sett.saveformat}'))
        plt.close()


class GetSampleBorders:
    """Create sample-specific border scores."""

    def __init__(self, samplepath, channel, scoring, anchor, variables):
        self.name = samplepath.name
        # Get anchoring point
        self.ind_start = 0
        # Select necessary columns from the data
        id_cols = ['NormDist', 'DistBin']
        self.var_cols = variables

        # Find data of border detection channel
        filepath = samplepath.joinpath(f'{channel}.csv')
        data = None
        try:
            data = pd.read_csv(filepath, index_col=False)
            self.data = data.loc[:, id_cols + self.var_cols]
            self.error = False
        except FileNotFoundError:
            print(f'ERROR: {filepath.name} not found for {self.name}.')
            self.error = True
        except KeyError:  # If all required columns are not found
            cols = [v for v in self.var_cols if v not in data.columns]
            print(f'{self.name}: Variables not found - {", ".join(cols)}')
            self.error = True

        # Create series with the scoring weights
        self.scoring = pd.Series(scoring)
        # Adjust anchor bin to the detection resolution (bins x 2)
        self.anchor = anchor * 2
        self.var_data = pd.DataFrame()

    def __call__(self, borders, dirpath):
        """Score sample."""
        # Collect variables and calculate diffs and stds
        self.get_variables(borders)

        # Normalize all variables between 0 and 1
        normalized = self.normalize_data()

        # Fit curve to variables and get deviations
        curve = get_fit(normalized.T)
        devs = deviate_data(normalized, curve)

        # Score the deviations
        scores = self.score_data(devs)
        sum_score = get_sum_score(scores)

        # Create plots if needed
        if Sett.plot_samples & Sett.Create_Border_Plots & Sett.Create_Plots:
            self.sample_plot(normalized, curve, scores, sum_score, dirpath)

        # Insert scores and sample's start index to full data set
        borders.scores.loc[scores.index, self.name] = sum_score
        borders.sample_starts.at[self.name] = self.ind_start

    def score_data(self, devs):
        """Score deciation values based on weights."""
        # Minor smoothing to get rid of excess variance
        devs = smooth_data(devs, win=3, tau=5)
        # Multiply variable values by the scoring weights
        scores = devs.multiply(self.scoring, axis=1)
        return scores

    def sample_plot(self, normalized, curve, scores, sum_score, dirpath):
        """Create plots of the sample variable values and scores."""
        # Assign identifiers to data
        norm = normalized.T.assign(var_name=normalized.T.index, stype='norm')
        score = scores.T.assign(var_name=scores.T.index, stype='score')

        # Melt data to plottable form
        score = score.melt(id_vars=['var_name', 'stype'])
        norm = norm.melt(id_vars=['var_name', 'stype'])

        # Create canvas
        _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 5), gridspec_kw={'height_ratios': [3.5, 5, 5]})

        # Plot final score of sample
        sum_score = sum_score.dropna()  # Drop missing values > plot cont. line
        ax1.plot(sum_score.index, sum_score.values, 'dimgrey')

        # Plot peak detection threshold line
        left, right = ax1.get_xlim()
        ax1.hlines(Sett.peak_thresh, xmin=left, xmax=right, linewidth=1, zorder=0, linestyles='dashed',
                   color='firebrick')

        # Plot raw scores of sample
        sns.lineplot(data=score.infer_objects(), x='variable', y='value', hue='var_name', alpha=0.7, ax=ax2)

        # Adjust legend location
        ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1)

        # Variable values plot
        sns.lineplot(data=norm.infer_objects(), x='variable', y='value', hue='var_name', alpha=0.7, legend=False,
                     ax=ax3)
        ax3.plot(curve.columns, curve.values.ravel(), 'r--')

        # Set titles and other labels
        ax1.set_title('Smooth score differential')
        ax2.set_title('Raw scores')
        ax3.set_title('Normalized variables')
        plt.xlabel('Linear Distance')
        ax1.set_ylabel('Score')
        ax2.set_ylabel('Score')
        ax2.set_xlabel('')

        # Re-adjust tick labels to original binning
        for ax in [ax1, ax2, ax3]:
            locs = [int(n/2) for n in ax.get_xticks()]
            ybot, ytop = ax.get_ylim()
            ax.vlines(self.anchor, ybot, ytop, 'firebrick', zorder=0, linestyles='dashed')
            ax.set_xticklabels(labels=locs)

        # Adjust plot space, add title, and save
        plt.subplots_adjust(bottom=0.1, left=0.1, right=0.75, hspace=0.8)
        plt.suptitle(self.name)
        plt.savefig(dirpath.joinpath(f'{self.name}.{Sett.saveformat}'))
        plt.close()

    def get_variables(self, borders):
        """Collect needed variables and calculate required characteristics."""
        # Get sample's width
        width = self.get_width(borders.width_data)
        self.var_data = self.var_data.assign(width=width, width_diff=get_diff(width))

        # Get sample's start index in the full dataset (for determining border locations in the sample's own indexing).
        self.ind_start = self.var_data.index[0]

        # Recalculate binning for bin averages etc
        bins = np.linspace(0, 1, width.index.size + 1)
        self.data = self.data.assign(binning=pd.cut(self.data["NormDist"], bins=bins))

        # get counts and and related:
        if 'Count' in self.scoring.keys():
            col = {'Count': self.get_count()}
            if 'Count_diff' in self.scoring.keys():
                col.update({'Count_diff': get_diff(col.get('Count'))})
            self.var_data = self.var_data.assign(**col)

        # Get variable averages and related:
        for var in self.var_cols:  # Sett.border_vars !!!
            col = {var: self.get_var(var)}
            if f'{var}_std' in self.scoring.keys():
                col.update({f'{var}_std': self.get_std(var)})
            if f'{var}_diff' in self.scoring.keys():
                col.update({f'{var}_diff': get_diff(col.get(var))})
            self.var_data = self.var_data.assign(**col)

        # Keep only necessary variables in DataFrame
        self.var_data = self.var_data.loc[:, Sett.scoring_vars.keys()]

        # Drop outliers (>3 SD)
        self.var_data = self.var_data.apply(drop_outlier)

    def get_count(self):
        """Count binned features."""
        counts = self.data["binning"].value_counts().sort_index()
        counts.index = self.var_data.index
        return counts

    def get_std(self, var):
        """Calculate standard deviations."""
        grouped = self.data.groupby('binning')
        std = grouped[var].std()
        std.index = self.var_data.index
        return std

    def get_var(self, var):
        """Calculate mean values of bins for a variable."""
        grouped = self.data.groupby('binning')
        mean = grouped[var].mean()
        mean.index = self.var_data.index
        return mean

    def get_width(self, width_data):
        """Collect sample's width from full data."""
        width = width_data.loc[:, self.name]
        width = width.rename('width')
        return width.dropna()

    def normalize_data(self):
        """Normalize variables between zero and one."""
        normalized = self.var_data.apply(norm_func)
        return normalized


class PeakDialog:
    """Create user input-window for peak plotting."""

    def __init__(self, data=None, master=None):
        self.master = master
        top = tk.Toplevel(master)
        top.bind('<Return>', self.apply)

        # Labels and button
        tk.Label(top, text="Select peaks for plotting:").grid(row=0)
        tk.Label(top, text="(group, bin, prominence)").grid(row=1)
        tk.Button(top, text="OK", command=self.apply).grid(row=1, column=4)

        data.prominence = data.prominence.round(decimals=2)
        self.values = []
        self.bools = None

        # Print detected peaks and related data
        for i, row in data.iterrows():
            string = ',  '.join(row[:3].values.astype(str))
            val = tk.IntVar(value=1)
            wg = tk.Checkbutton(top, text=string, variable=val)
            wg.grid(row=i+2, columnspan=2, sticky='W')
            self.values.append(val)

        print('\a')
        self.top = top
        self.top.wait_window()

    def apply(self):
        """Apply peak selection."""
        # Make list of bools for slicing
        self.bools = [bool(v.get()) for v in self.values]
        self.top.destroy()
        self.master.grab_set()
        return self.bools


def append_binning(sample_starts, peaks):
    """Append sample index locations to detected peaks."""

    def _define_sample_bin(row):
        """Find given borders peak's index location on specific samples."""
        # Select samples relevant  to peak at hand
        ind = sample_starts.index.str.contains(f'{row.group}_')
        # Subtract offset of the samples from the peak location
        row[sample_starts.index[ind]] = row.peak - sample_starts[ind]
        return row

    # Reindex DF to have all samples
    peaks = peaks.reindex(columns=peaks.columns.append(sample_starts.index))
    # Define index location of each group's peaks on each sample
    peaks = peaks.apply(_define_sample_bin, axis=1)
    return peaks


def deviate_data(data, curve):
    """Subtract fitted curve from variable values."""
    devs = data.subtract(curve.iloc[0, :], axis=0)
    return devs


def detect_peaks(score_arr, x_dist=6, thresh=0.15, width=1):
    """Find peaks from total scores of sample groups."""
    # Group data by sample group
    grouped = score_arr.groupby(score_arr.loc[:, 'group'])

    # Create DF for peak data
    all_peaks = pd.DataFrame(columns=['group', 'peak', 'prominence', 'sign'])

    # For each sample group:
    for grp, data in grouped:
        # Drop empty values
        total = data.value.dropna()

        # POSITIVE peaks
        peaks, _ = find_peaks(total, distance=x_dist, height=thresh, width=width)  # , threshold=thresh)
        # Find prominence of found peaks
        prom = peak_prominences(total, peaks)[0]
        peaks = total.index[peaks].get_level_values(1).values
        peak_dict = {'group': grp, 'peak': peaks, 'prominence': prom, 'sign': 'pos'}

        # NEGATIVE peaks
        # neg_total = total * -1
        # npeaks, _ = find_peaks(neg_total, distance=4, height=thresh,
        #                           width=2)#, threshold=thresh)
        # nprom = peak_prominences(neg_total, npeaks)[0]
        # npeaks = neg_total.index[npeaks].get_level_values(1).values
        # npeak_dict = {'group': grp, 'peak': npeaks, 'prominence': nprom * -1,
        #               'sign': 'neg'}

        # Add groups data to full peak data
        all_peaks = all_peaks.append(pd.DataFrame(peak_dict), ignore_index=True)
        # all_peaks = all_peaks.append(pd.DataFrame(npeak_dict), ignore_index=True)
    return all_peaks


def get_diff(data):
    """Calculate bin-to-bin differences"""
    diff = data.diff()[1:]
    # Shift data by one index position to match locations
    diff.index = diff.index - 1
    return diff


def get_group_total(data):
    """Get smoothed total scores of groups."""
    # smoothed = smooth_data(data.T, win=7, tau=10).T
    trimmed = prepare_data(data)
    # Get mean score for each bin
    s_sums = trimmed.groupby(['group', 'variable']).apply(lambda x: x.value.mean())

    # Transform to dataframe and assign necessary identifier columns
    s_sums = s_sums.to_frame(name='value')
    s_sums = s_sums.groupby('group').apply(lambda x: x.assign(value=smooth_data(x.value, win=5, tau=10)))

    # Add identifier columns, i.e. groups and bins
    s_sums = s_sums.assign(group=s_sums.index.get_level_values(0), variable=s_sums.index.get_level_values(1))
    return s_sums


def get_sum_score(scores):
    """Find the summed score of the detection variables."""
    # Smooth raw scores and the sum for total
    s_sum = smooth_data(scores, win=7, tau=10).sum(axis=1)
    # trim zeros from array ends
    trimmed_sum = np.trim_zeros(s_sum)
    # Get bin-to-bin score differential and drop outliers
    diffs = np.diff(trimmed_sum)
    diffs = drop_outlier(pd.Series(diffs, index=trimmed_sum.index[1:]))
    # Normalize with end points dropped (they have highly variant values)
    norm_diffs = norm_func(diffs[1:-1])
    sum_score = pd.Series(norm_diffs, index=trimmed_sum.index[2:-1])
    return sum_score


def drop_outlier(arr):
    """Drop bins with outlying values."""
    with warnings.catch_warnings():  # Catch warning from empty bins
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean = np.nanmean(arr.astype('float'))
        std = np.nanstd(arr.astype('float'))

    drop_val = 3 * std

    # Replace values more distant than drop_val with NaN
    arr.where(np.abs(arr - mean) <= drop_val, other=np.nan, inplace=True)
    return arr


def get_peak_data(data, threshold):
    """Find peaks from score data."""
    peaks = detect_peaks(data, thresh=threshold)
    # Determine data type of columns
    peaks = peaks.infer_objects()
    return peaks


def prepare_data(data):
    """Transform data into long form."""
    # Add group identifier
    data = data.assign(group=[s.split('_')[0] for s in data.index])
    # Melt data with group as ID
    data = trim_data(data).melt(id_vars='group').infer_objects()
    return data


def plotting_directory(plotdir):
    """Determine and make plot save directory."""
    dirpath = plotdir.joinpath('Borders')
    dirpath.mkdir(exist_ok=True)
    return dirpath


def read_widths(datadir):
    """Find and read width datafile"""
    filepath = datadir.joinpath('Sample_widths_norm.csv')
    try:
        widths = pd.read_csv(filepath, index_col=False)
    except FileNotFoundError:
        msg = 'Width data not found. Perform analysis with measure_width.'
        print(f'ERROR: {msg}')
        lg.logprint(LAM_logger, f'-> {msg}', 'e')
        return None
    return widths


def smooth_data(data, win=3, tau=10):
    """Perform rolling smoothing to data."""
    smoothed_data = data.rolling(win, center=True, win_type='exponential').mean(tau=tau)
    return smoothed_data


def trim_data(data, grouper='group'):
    """Trim bins where less than half of a sample group has values."""
    grouped = data.groupby(grouper)
    # Create mask
    masks = grouped.apply(lambda x: x.isna().sum() < x.shape[0]/2)
    # Apply mask
    return grouped.apply(lambda x, m=masks: apply_mask(x, m.loc[x.name, :]))


def apply_mask(arr, mask):
    """Apply given mask to dataframe columns."""
    return arr.apply(lambda x, m=mask: x.where(m), axis=1)


def get_fit(data, name='c', id_var=None):
    """Fit a curve to data for variable deviations."""
    # Change to long format
    try:
        data = data.melt(id_vars=id_var)
    except KeyError:
        data = data.melt()

    # Drop missing values
    data = data.dropna()

    # Take all x and y data
    x_data = data.variable.values.astype(np.float)
    y_data = data.value.values.astype(np.float)

    #  #Fit x and y data with 4th degree polynomial
    # z = np.polyfit(x_data.astype(np.float), y_data.astype(np.float), 4)
    # f = np.poly1d(z)
    # x_curve = data.variable.unique()
    # y_curve = f(x_curve)

    # !!!! Chebyshev polynomial
    c_fit = Chebyshev.fit(x_data, y_data, 5, domain=[0, 1])
    x_curve = data.variable.unique()
    y_curve = c_fit(x_curve)
    # !!!! End

    # Create DF from obtained fit
    curve = pd.DataFrame(index=[name], columns=x_curve.astype(int), data=np.reshape(y_curve, (-1, len(y_curve))))
    return curve


def norm_func(arr: pd.Series) -> pd.Series:
    """Normalize array between 0 and 1."""
    return (arr-np.nanmin(arr))/(np.nanmax(arr)-np.nanmin(arr))


def ask_peaks(peaks, gui_root):
    """Ask user input to determine plottable peaks."""

    if gui_root is not None:  # If GUI, make input window
        win = PeakDialog(data=peaks, master=gui_root)
        store.border_peaks = peaks.loc[win.bools, :]

    else:  # Otherwise ask for written input
        print('\aDetected peaks:')
        print(peaks)
        ans = input('\nGive indices of peaks to DROP (e.g. 1, 2): ')
        if ans == '':
            store.border_peaks = peaks
        else:
            nums = [int(v) for v in ans.split(',')]
            if nums:
                store.border_peaks = peaks.loc[peaks.index.difference(nums), :]


def peak_selection(datadir, gui_root=None):
    """Collect detected peaks for plotting."""

    try:
        peaks = pd.read_csv(datadir.joinpath('Borders_peaks.csv'))
    except FileNotFoundError:
        msg = 'Borders NOT added to plots - missing Border_peaks.csv'
        print(f'\nWARNING: {msg}')
        lg.logprint(LAM_logger, msg, 'w')
        return

    if Sett.force_dialog:
        store.border_peaks = peaks
    elif Sett.select_peaks:  # Ask for subset of peaks if needed
        ask_peaks(peaks, gui_root)
    else:
        store.border_peaks = peaks


def test_channel(samplepaths, border_channel):
    """Test existence of required channel data for border detection."""

    not_found = 0  # Number of samples with no data
    samples = []
    for path in samplepaths:  # Test all samples

        # If file found
        if path.joinpath(f'{border_channel}.csv').is_file():
            continue
        # If not found
        not_found += 1
        samples.append(path.parent.name)

    # File not found for any sample
    if not_found == len(samplepaths):
        print('\nBorder detection data not found for any sample.')
        changed = ask_new_channel(border_channel)  # Confirm channel
        if changed:  # If channel is changed
            return True
        return False

    # If data not found for some samples
    elif not_found > 0:
        print('\nBorder detection data not found for some samples.')
        print(f'Missing samples: {samples}')
        _ = ask_new_channel(border_channel)
    return True


def ask_new_channel(border_channel):
    """Ask user input to determine new border detection channel."""

    if Sett.force_dialog:  # If forcing no user input
        msg = "Border detection data not found for all samples."
        lg.logprint(LAM_logger, msg, 'i')
        return False

    flag = True
    print('\a')

    while flag:  # Ask input until satisfied
        dlg = f'Current channel is {border_channel}. Change channel? [y/n]'
        ans = system.ask_user(dlg)  # Ask whether to change channel
        if ans in ('Y', 'y'):
            dlg = "Give name of new border detection channel: "
            new_channel = system.ask_user(dlg)  # Ask channel name
            change_keys(border_channel, new_channel)  # Change variables
            Sett.border_channel = new_channel
            msg = f'Border detection channel changed from {border_channel} to {new_channel}.'
            print('\n' + msg)
            lg.logprint(LAM_logger, msg, 'i')
            return True
        if ans in ('N', 'n'):
            return False
        print('Command not understood.\n')


def change_keys(old_channel, new_channel):
    """Change border detection variables between two channels."""
    old = f'_{old_channel}'
    new = f'_{new_channel}'
    # Change column names
    Sett.border_vars = [col.replace(old, new) for col in Sett.border_vars]
    items = list(Sett.scoring_vars.items())
    # Update scoring dictionary
    for key, value in items:
        if old in key:
            new_key = key.replace(old, new)
            Sett.scoring_vars.update({new_key: value})
            del Sett.scoring_vars[key]
