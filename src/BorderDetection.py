# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:03:36 2020

@author: arska
"""
# LAM modules
from settings import settings as Sett
import logger as lg
# Other
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import warnings

try:
    LAM_logger = lg.get_logger(__name__)
except AttributeError:
    print('Cannot get logger')

# workdir = pl.Path(r'E:\Code_folder\ALLSTATS')
# channel = "DAPI"
# samplesdir = workdir.joinpath('Analysis Data', 'Samples')
# plotdir = workdir.joinpath('Analysis Data', 'Plots', 'Borders')
# plotdir.mkdir(exist_ok=True)
# datadir = workdir.joinpath('Analysis Data', 'Data Files')
# width_path = datadir.joinpath('Sample_widths_norm.csv')
# scoring = {#'Count': 0,
#            # 'Count_diff': 0,
#            'width': -1.5,
#            'width_diff': 1.5,
#            'Area_diff': -1,
#            # 'Area_std': -1,
#            f'Nearest_Dist_{channel}': -1.5,
#            f'Nearest_Dist_{channel}_diff': 1.5,
#            # f'Nearest_Dist_{channel}_std': 1.5
#            }

# path_kws = {'samplesdir': samplesdir, 'plotdir': plotdir, 'datadir': datadir}


def detect_borders(paths, all_samples, palette, anchor,
                   threshold=Sett.peak_thresh, variables=Sett.border_vars,
                   scoring=Sett.scoring_vars, channel=Sett.border_channel):
    print('-Finding border regions-')
    lg.logprint(LAM_logger, 'Finding border regions.', 'i')
    b_dirpath = plotting_directory(paths.plotdir)
    widths = read_widths(paths.datadir)
    if widths is None:
        return
    border_data = FullBorders(all_samples, widths, anchor, palette)
    print('  Scoring samples  ...')
    for path in all_samples:
        sample = GetSampleBorders(path, channel, scoring, anchor, variables)
        sample(border_data, b_dirpath)
    print('  Finding peaks  ...')
    peaks = border_data(b_dirpath, threshold)
    print('Found peaks:')
    print(peaks)
    lg.logprint(LAM_logger, 'Border detection done.', 'i')


class FullBorders:

    def __init__(self, samples, widths, anchor, palette):
        self.samples = samples
        self.groups = sorted(list(set([p.name.split('_')[0] for p in samples if
                           len(p.name.split('_')) > 1])))
        self.width_data = widths
        self.anchor = anchor * 2
        self.palette = palette
        self.scores = pd.DataFrame(columns=[p.name for p in samples],
                                   index=widths.index)

    def __call__(self, dirpath, threshold):
        # group_totals = self.group_scores()
        flattened, curves = self.flatten_scores()
        s_sums = get_group_total(flattened)
        peaks = get_peak_data(s_sums, threshold)
        if (Sett.Create_Border_Plots & Sett.Create_Plots):
            print('  Creating border plot  ...')
            scores = prepare_data(self.scores.T)
            flat = prepare_data(flattened)
            self.group_plots(scores, flat, curves, s_sums, peaks, dirpath)
        # self.group_plots(flattened)
        return peaks

    def flatten_scores(self):
        groups = [s[0] for s in self.scores.columns.str.split('_')]
        vals = self.scores.T.copy().assign(group=groups)
        grouped = vals.groupby('group', as_index=True)
        curves = pd.DataFrame()
        curves = grouped.apply(lambda x: get_fit(x.melt(id_vars='group')))
        devs = vals.apply(lambda x, c=curves: x - c[x.group], axis=1)
        return devs, curves

    def group_plots(self, scores, flat, curves, s_sums, peaks, dirpath):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 5),
                                            gridspec_kw={
                                                'height_ratios': [3, 5, 5]})
        plt.subplots_adjust(hspace=0.75, top=0.85, bottom=0.1, left=0.1,
                            right=0.85)
        sns.lineplot(data=s_sums.infer_objects(), x='variable', y='value',
                     hue='group', palette=self.palette, alpha=0.7,
                     legend=False, ax=ax1)
        for peak in peaks.iterrows():
            loc = peak[1]['peak']
            prom = peak[1]['prominence']
            grp = peak[1]['group']
            c_val = s_sums.loc[(grp, loc)].value
            color = self.palette[grp]
            ax1.vlines(x=loc, ymin=c_val-prom, ymax=c_val, color=color,
                       linewidth=1, zorder=2)
            ax1.annotate(loc, (loc + 0.03, c_val * 1.03), color=color,
                         alpha=0.7)
        ax1.set_title('Smoothed sum scores')

        sns.lineplot(data=flat, x='variable', y='value', hue='group',
                     alpha=0.7, palette=self.palette, ax=ax2)
        ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1)
        ax2.set_title('Flattened scores')
        sns.lineplot(data=scores, x='variable', y='value', hue='group',
                     palette=self.palette, alpha=0.7, legend=False, ax=ax3)
        p_curves = pd.DataFrame(data=curves.values, columns=['value'])
        p_curves = p_curves.assign(variable=curves.index.get_level_values(1),
                                   group=curves.index.get_level_values(0))
        sns.lineplot(data=p_curves.infer_objects(), x='variable', y='value',
                     hue='group', style='group', alpha=0.9, legend=False,
                     dashes=True, linewidth=0.5, palette=self.palette, ax=ax3)
        ax3.set_title('Group scores')
        left, right = plt.xlim()
        locs, _ = plt.xticks()
        lbls = [int(n/2) for n in locs]
        for ax in (ax1, ax2, ax3):
            ybot, ytop = ax.get_ylim()
            ax.vlines(self.anchor, ybot, ytop, 'firebrick', zorder=0,
                      linestyles='dashed')
            ax.hlines(0, xmin=left, xmax=right, linewidth=1, zorder=0,
                      linestyles='dashed', color='dimgrey')
            ax.set_xticklabels(labels=lbls)
            ax.set_xlabel('')
            ax.set_ylabel('Score')
        ax3.set_xlabel('Linear Position')
        plt.suptitle('GROUP BORDER SCORES')
        plt.savefig(dirpath.joinpath(f'All-Border_Scores.pdf'))


class GetSampleBorders:

    def __init__(self, samplepath, channel, scoring, anchor, variables):
        self.sample_name = samplepath.name
        mp_file = samplepath.joinpath('MPs.csv')
        self.MP = pd.read_csv(mp_file, index_col=False).iat[0, 0]
        filepath = samplepath.joinpath(f'{channel}.csv')
        data = pd.read_csv(filepath, index_col=False)
        id_cols = ['NormDist', 'DistBin']
        self.var_cols = variables
        self.data = data.loc[:, id_cols + self.var_cols]
        self.scoring = pd.Series(scoring)
        self.anchor = anchor * 2

    def __call__(self, FullBorders, dirpath):
        # print(f'{self.sample_name}  ...')
        self.get_variables(FullBorders)
        normalized = self.normalize_data()
        curve = get_fit(normalized.T.melt())
        devs = deviate_data(normalized, curve)
        scores = self.score_data(devs)
        if (Sett.plot_samples & Sett.Create_Border_Plots & Sett.Create_Plots):
            self.sample_plot(normalized, curve, scores, dirpath)
        FullBorders.scores.loc[scores.index,
                               self.sample_name] = scores.sum(axis=1)

    def score_data(self, devs):
        scores = devs.multiply(self.scoring, axis=1)
        return scores

    def sample_plot(self, normalized, curve, scores, dirpath):
        norm = normalized.T.assign(var_name=normalized.T.index, stype='norm')
        score = scores.T.assign(var_name=scores.T.index, stype='score')
        norm_mean = norm.mean()

        scurve = get_fit(scores.T.melt())
        fitted_mean = scores.apply(lambda x, c=scurve: x - c, axis=0).mean(axis=1)

        score = score.melt(id_vars=['var_name', 'stype'])
        norm = norm.melt(id_vars=['var_name', 'stype'])
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 5),
                                            gridspec_kw={
                                                'height_ratios': [2, 5, 5]})
        ax1.plot(fitted_mean.index, fitted_mean.values, 'dimgrey')
        ax1.set_title('Flattened score deviance')
        sns.lineplot(data=score.infer_objects(), x='variable', y='value',
                     hue='var_name', alpha=0.7, ax=ax2)
        ax2.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1)
        ax2.set_title('Score deviance')
        sns.lineplot(data=norm.infer_objects(), x='variable', y='value',
                     hue='var_name', alpha=0.7, legend=False, ax=ax3)
        ax3.plot(norm_mean.index, norm_mean.values, 'dimgrey')
        ax3.plot(curve.index, curve.values, 'r')
        ax3.set_title('Normalized variables')
        plt.xlabel('Linear Distance')
        ax1.set_ylabel('Score')
        ax2.set_ylabel('Score')
        ax2.set_xlabel('')
        locs, _ = plt.xticks()
        lbls = [int(n/2) for n in locs]
        for ax in [ax1, ax2, ax3]:
            ybot, ytop = ax.get_ylim()
            ax.vlines(self.anchor, ybot, ytop, 'firebrick', zorder=0,
                      linestyles='dashed')
            ax.set_xticklabels(labels=lbls)

        plt.subplots_adjust(bottom=0.1, left=0.1, right=0.75, hspace=0.8)
        plt.suptitle(self.sample_name)
        plt.savefig(dirpath.joinpath(f'{self.sample_name}.pdf'))
        plt.close()

    def get_variables(self, FullBorders):
        width = self.get_width(FullBorders.width_data)
        self.var_data = pd.DataFrame(index=width.index)
        self.var_data = self.var_data.assign(width=width,
                                             width_diff=self.get_diff(width))
        bin_max = self.data.DistBin.max()
        bins = np.linspace(0, 1, (bin_max + 1) * 2)
        self.data = self.data.assign(binning=pd.cut(self.data["NormDist"],
                                                    bins=bins))
        # get counts and and related:
        if 'Count' in self.scoring.keys():
            col = {'Count': self.get_count()}
            if 'Count_diff' in self.scoring.keys():
                col.update({'Count_diff': self.get_diff(col.get('Count'))})
            self.var_data = self.var_data.assign(**col)
        # Get variable averages and related:
        for var in self.var_cols:  # Sett.border_vars !!!
            col = {var: self.get_var(var)}
            if f'{var}_std' in self.scoring.keys():
                col.update({f'{var}_std': self.get_std(var)})
            if f'{var}_diff' in self.scoring.keys():
                col.update({f'{var}_diff': self.get_diff(col.get(var))})
            self.var_data = self.var_data.assign(**col)

    def get_count(self):
        counts = self.data["binning"].value_counts().sort_index()
        counts.index = self.var_data.index
        return counts

    def get_diff(self, data):
        diff = data.diff()
        return diff

    def get_std(self, var):
        grouped = self.data.groupby('binning')
        std = grouped[var].std()
        std.index = self.var_data.index
        return std

    def get_var(self, var):
        grouped = self.data.groupby('binning')
        mean = grouped[var].mean()
        mean.index = self.var_data.index
        return mean

    def get_width(self, width_data):
        width = width_data.loc[:, self.sample_name]
        width = width.rename('width')
        return width.dropna()

    def normalize_data(self):
        normalized = self.var_data.apply(norm_func)
        return normalized


def assign_fit(x, fit):
    vals = []
    for val in x:
        vals.append(fit.at[val[1].at['NormDist']])
    return vals


def deviate_data(data, curve):
    devs = data.subtract(curve, axis=0)
    return devs


def mean_func(arr):
    mean= arr['value'].mean()
    return mean


def detect_peaks(score_arr, x_dist=6, thresh=0.15, width=1):
    grouped = score_arr.groupby(score_arr.loc[:, 'group'])
    all_peaks = pd.DataFrame(columns=['group', 'peak', 'prominence', 'sign'])
    for grp, data in grouped:
        total = data.value.dropna()
        # POSITIVE peaks
        peaks, _ = find_peaks(total, distance=x_dist, height=thresh,
                              width=width)#, threshold=thresh)
        prom = peak_prominences(total, peaks)[0]
        peaks = total.index[peaks].get_level_values(1).values
        peak_dict = {'group': grp, 'peak': peaks, 'prominence': prom,
                     'sign': 'pos'}
        # NEGATIVE peaks
        # neg_total = total * -1
        # npeaks, _ = find_peaks(neg_total, distance=4, height=thresh,
        #                           width=2)#, threshold=thresh)
        # nprom = peak_prominences(neg_total, npeaks)[0]
        # npeaks = neg_total.index[npeaks].get_level_values(1).values
        # npeak_dict = {'group': grp, 'peak': npeaks, 'prominence': nprom * -1,
        #               'sign': 'neg'}

        all_peaks = all_peaks.append(pd.DataFrame(peak_dict),
                                     ignore_index=True)
        # all_peaks = all_peaks.append(pd.DataFrame(npeak_dict),
        #                              ignore_index=True)
    return all_peaks


def get_group_total(data):
    smoothed = smooth_data(data.T, win=7, tau=10).T
    trimmed = prepare_data(smoothed)
    s_sums = trimmed.groupby(['group', 'variable']
                             ).apply(lambda x: drop_outlier(x.value).sum())
    # s_sums = trimmed.groupby(['group', 'variable']).sum()
    s_sums = s_sums.to_frame(name='value')
    s_sums = s_sums.assign(group=s_sums.index.get_level_values(0),
                           variable=s_sums.index.get_level_values(1))
    return s_sums


def drop_outlier(arr):
    with warnings.catch_warnings():  # Catch warning from empty bins
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean = np.nanmean(arr.astype('float'))
        std = np.nanstd(arr.astype('float'))
    drop_val = 3 * std
    arr.where(np.abs(arr - mean) <= drop_val, other=np.nan, inplace=True)
    # arr.value.where(np.abs(arr.value - mean) <= drop_val, other=np.nan, axis=1)
    return arr


def get_peak_data(data, threshold):
    peaks = detect_peaks(data, thresh=threshold)
    peaks = peaks.infer_objects()
    return peaks


def prepare_data(data):
    data = data.assign(group=[s.split('_')[0] for s in data.index])
    data = trim_data(data).melt(id_vars='group').infer_objects()
    return data


def fit_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d
    # return a * x**2 + b * x + c
    # return a * np.exp(-b * x) + c


def plotting_directory(plotdir):
    dirpath = plotdir.joinpath('Borders')
    dirpath.mkdir(exist_ok=True)
    return dirpath


def read_widths(datadir):
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
    smooth_data = data.rolling(win, center=True, win_type='exponential'
                             ).mean(tau=tau)
    return smooth_data


def trim_data(data, grouper='group'):
    grouped = data.groupby(grouper)
    masks = grouped.apply(lambda x: x.isna().sum() < x.shape[0]/2)
    return grouped.apply(lambda x, m=masks: apply_mask(x, m.loc[x.name, :]))


def apply_mask(arr, mask):
    return arr.apply(lambda x, m=mask: x.where(m), axis=1)


def get_fit(data):
    data = data.dropna()
    x_data = data.variable.values
    y_data = data.value.values
    # popt, pcov = curve_fit(fit_func, x_data, y_data) #, sigma=0.1)
    z = np.polyfit(x_data.astype(np.float), y_data.astype(np.float), 4)
    f = np.poly1d(z)
    x_curve = data.variable.unique()
    # x_curve = np.linspace(0, 1, 100)
    y_curve = f(x_curve)
    curve = pd.Series(y_curve, index=x_curve)
    return curve


def norm_func(arr):
    return (arr-arr.min())/(arr.max()-arr.min())
