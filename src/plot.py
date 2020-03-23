# -*- coding: utf-8 -*-
"""
LAM-module for plot creation.

Created on Tue Mar 10 11:45:48 2020
@author: Arto I. Viitanen
"""
# LAM modules
from settings import settings as Sett, store
import logger as lg
import system
# Standard libraries
import warnings
# from random import shuffle
# Packages
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)
    import pandas as pd
try:
    LAM_logger = lg.get_logger(__name__)
except AttributeError:
    print('Cannot get logger')


class DataHandler:
    """
    Handle data for plotting.

    Data will be passed to MakePlot-class
    """

    def __init__(self, samplegroups, paths, savepath=None):
        if savepath is None:
            self.savepath = samplegroups.paths.plotdir
        else:
            self.savepath = savepath
        self.palette = samplegroups._grpPalette
        self.center = samplegroups._center
        self.total_length = samplegroups._length
        self.MPs = samplegroups._AllMPs
        self.data = pd.DataFrame()
        self.paths = paths


    def get_data(self, *args, **kws):
        melt = False
        all_data = pd.DataFrame()
        for path in self.paths:
            data = system.read_data(path, header=0, test=False)
            if 'IDs' in kws.keys():
                data = identifiers(data, path, kws.get('IDs'))
            if 'melt' in kws.keys():
                m_kws = kws.get('melt')
                if 'path_id' in args:
                    id_sep = kws.get('id_sep')
                    try:
                        id_var = path.stem.split('_')[id_sep]
                        m_kws.update({'value_name': id_var})
                    except IndexError:
                        msg = 'Faulty list index. Incorrect file names?'
                        print('ERROR: {}'.format(msg))
                        lg.logprint(LAM_logger, msg, 'e')
                data = data.T.melt(id_vars=m_kws.get('id_vars'),
                                   value_vars=m_kws.get('value_vars'),
                                   var_name=m_kws.get('var_name'),
                                   value_name=m_kws.get('value_name'))
                melt = True
            else:
                data = data.T
            if 'merge' in args:
                if all_data.empty:
                    all_data = data
                else:
                    all_data = all_data.merge(data, how='outer', copy=False,
                                              on=kws.get('merge_on'))
                continue
            all_data = pd.concat([all_data, data], sort=True)
        all_data.index = pd.RangeIndex(stop=all_data.shape[0])
        if 'drop_outlier' in args and Sett.Drop_Outliers:
            all_data = drop_outliers(all_data, melt, **kws)
        all_data = all_data.infer_objects()
        return all_data


class MakePlot:
    """Create decorated plots."""

    # Base keywords utilized in plots.
    base_kws = {'hue': 'Sample Group', 'row': 'Channel', 'col': 'Sample Group',
                'height': 3, 'aspect': 3, 'flier_size': 2, 'title_y': 0.95,
                'sharex': False, 'sharey': False, 'gridspec': {'hspace': 0.45},
                'xlabel': 'Linear Position', 'ylabel': 'Feature Count'}

    def __init__(self, data, handle, title, sec_data=None):
        self.data = data
        self.sec_data = sec_data
        self.plot_error = False
        self.handle = handle
        self.title = title
        self.format = Sett.saveformat
        self.g = None
        self.filepath = handle.savepath.joinpath(self.title +
                                                 ".{}".format(Sett.saveformat))

    def __call__(self, func, *args, **kws):
        plot_kws = merge_kws(MakePlot.base_kws, kws)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            # Make canvas if needed:
            if 'no_grid' not in args:
                if 'Versus' in args:
                    pass
                else:
                    self.g = self.get_facet(**plot_kws)
            # Plot data
            self.g = func(self, **plot_kws)
        if self.plot_error:
            msg = "Plot not saved"
            print("WARNING: {}".format(msg))
            lg.logprint(LAM_logger, msg, 'w')
            return
        self.add_elements(*args, **plot_kws)
        self.save_plot()

    def add_elements(self, *args, **kws):
        if 'centerline' in args:
            self.centerline()
        if 'ticks' in args:
            self.xticks()
        if 'labels' in args:
            self.collect_labels(kws.get('xlabel'), kws.get('ylabel'))
            self.labels(kws.get('xlabel'), kws.get('ylabel'))
        if 'legend' in args:
            self.g.add_legend()
        if 'title' in args:
            self.set_title(**kws)

    def centerline(self):
        """Plot centerline, i.e. the anchoring point of samples."""
        for ax in self.g.axes.flat:
            __, ytop = ax.get_ylim()
            ax.vlines(self.handle.center, 0, ytop, 'dimgrey', zorder=0,
                      linestyles='dashed')

    def collect_labels(self, xlabel, ylabel):
        if 'collect' not in (xlabel, ylabel):
            return
        for ax in self.g.axes.flat:
            title = ax.get_title()
            var_strs = title.split(' | ')
            label_strs = [l.split(' = ')[1] for l in var_strs]
            if ylabel == 'collect':
                label = get_unit(label_strs[0])
                ax.set_ylabel(label)
            if xlabel == 'collect':
                label = get_unit(label_strs[1])
                ax.set_xlabel(label)
            ax.set_title('')

    def get_facet(self, **kws):
        g = sns.FacetGrid(self.data, row=kws.get('row'),
                          col=kws.get('col'), hue=kws.get('hue'),
                          sharex=kws.get('sharex'), sharey=kws.get('sharey'),
                          gridspec_kws=kws.get('gridspec'),
                          height=kws.get('height'), aspect=kws.get('aspect'),
                          legend_out=True, dropna=False,
                          palette=self.handle.palette)
        return g

    def xticks(self):
        """Set plot xticks & tick labels to be shown every 5 ticks."""
        xticks = np.arange(0, self.handle.total_length, 5)
        plt.setp(self.g.axes, xticks=xticks, xticklabels=xticks)

    def labels(self, xlabel=None, ylabel=None):
        if xlabel not in (None, 'collect'):
            for ax in self.g.axes.flat:
                ax.set_xlabel(xlabel)
        if ylabel not in (None, 'collect'):
            for ax in self.g.axes.flat:
                ax.set_ylabel(ylabel)

    def set_title(self, **kws):
        self.g.fig.suptitle(self.title, weight='bold', y=kws.get('title_y'))

    def stats(self):
        pass

    def save_plot(self):
        fig = plt.gcf()
        fig.savefig(str(self.filepath), format=self.format)
        plt.close()


def drop_func(x, mean, drop_value):
    if np.abs(x - mean) <= drop_value:
        return x
    return np.nan


def drop_outliers(all_data, melted, **kws):  # !!! Finish
    def drop(data, cols):
        """Drop outliers from a dataframe."""
        # Get mean and std of input data
        values = data.loc[:, cols]
        with warnings.catch_warnings():  # Ignore empty bin warnings
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean = np.nanmean(values.astype('float'))
            std = np.nanstd(values.astype('float'))
        drop_val = Sett.dropSTD * std
        if isinstance(values, pd.DataFrame):
            data = values.applymap(lambda x: drop_func(x, mean, drop_val))
        elif isinstance(values, pd.Series):
            data.loc[:, cols] = values.apply(drop_func, args=[mean, drop_val])
        return data

    # Handle data for dropping
    grouper = kws.get('drop_grouper')
    grp_data = all_data.groupby(grouper)
    if melted:
        names = kws.get('melt').get('value_name')
    else:
        names = all_data.loc[:, all_data.columns != grouper].columns
    all_data = grp_data.apply(lambda grp: drop(grp, cols=names))
    return all_data


def identifiers(data, path, ids):
    if 'Channel' in ids:
        data.loc['Channel', :] = path.stem.split('_')[1]
    if 'Sample Group' in ids:
        data.loc['Sample Group', :] = [str(c).split('_')[0] for c in
                                       data.columns]
    if 'Sample' in ids:
        data.loc['Sample', :] = data.columns
    if 'Type' in ids:
        name = str(path.stem).split('_')[2:]
        data.loc['Type', :] = name
    return data


def get_unit(string):
    # If string is a LAM created value name:
    if string in ("Distance Means", "Width"):
        return "Units (coord system)"
    sub_str = string.split('-')
    if len(sub_str) == 3:
        chan, key, key_c = sub_str
    elif len(sub_str) == 2:
        key, key_c = sub_str
    else:
        key = sub_str[0]
    # If not user defined value:
    if key not in Sett.AddData.keys():
        if key in store.channels:
            return '{} Count'.format(string)
        else:
            return 'Value'
    # Otherwise, build label from the sub-units
    label = Sett.AddData.get(key)[1]
    if 'chan' in locals():
        label = '{}, '.format(chan) + label
    if 'key_c' in locals():
        if Sett.replaceID:
            key_c = Sett.channelID.get(key_c)
        label = label + '-{}'.format(key_c)
    return label


def merge_kws(kws1, kws2):
    new_kws = kws1.copy()
    if kws2 is not None:
        new_kws.update(kws2)
    return new_kws


def remove_from_kws(kws, *args):
    new_kws = kws.copy()
    for key in args:
        if isinstance(key, str):
            del new_kws[key]
    return new_kws

class plotter:
    """Fake class"""

    def __init__(self, plotData, savepath, center=0, title=None,
                 palette=None, color='b'):
        pass
