# -*- coding: utf-8 -*-
"""
LAM-module for plot creation.

Created on Tue Mar 10 11:45:48 2020
@author: Arto I. Viitanen
"""
# LAM modules
from settings import settings as Sett
import logger as lg
import system
# Standard libraries
import warnings
from random import shuffle
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


class data_handler:
    """
    Handle data for plotting.

    Data will be passed to plot.make_plot.
    """

    def __init__(self, samplegroups, *args, **kws):
        self.savepath = samplegroups.paths.plotdir
        self.palette = samplegroups._grpPalette
        self.center = samplegroups._center
        self.total_length = samplegroups._length
        self.MPs = samplegroups._AllMPs
        self.data = pd.DataFrame()

    def data_array(self):
        pass

    def get_add_vars(self):
        pass

    def get_data(self, paths, *args, **kws):
        melt = False
        all_data = pd.DataFrame()
        for path in paths:
            # !!! Add samplegroup identifier
            data = system.read_data(path, header=0, test=False)
            if 'IDs' in kws.keys():
                data = identifiers(data, path, kws.get('IDs'))
            if 'melt' in kws.keys():
                melt = True
                melt_kws = kws.get('melt')
                data = self.melt_data(data, **melt_kws)
            elif 'array' in args:
                # data = self.data_array(data)
                pass
            all_data = pd.concat([all_data, data], sort=True)
        all_data.index = pd.RangeIndex(stop=all_data.shape[0])
        if 'drop_outlier' in kws:
            all_data = drop_outliers(all_data, melt, **kws)
        return all_data


    def melt_data(self, data, **kws):
        data = data.T.melt(id_vars=kws.get('id_vars'),
                           value_vars=kws.get('value_vars'),
                           var_name=kws.get('var_name'),
                           value_name=kws.get('value_name'))
        return data

    # def call_plot(self, func, plot_kws=base_kws):  # needed ???
    #     plot.make_plot(func)


class make_plot:
    """Create decorated plots."""

    # Base keywords utilized in plots.
    base_kws = {'hue': 'Sample Group', 'row': 'Channel', 'col': 'Sample Group',
                'height': 5, 'aspect': 3, 'flier_size': 2, 'title_y': 0.95,
                'sharex': False, 'sharey': 'row', 'gridspec': {'hspace': 0.3},
                'xlabel': 'Linear Position', 'ylabel': 'Count'}

    def __init__(self, handle, title, *args, **kws):
        self.plot_error = False
        self.handle = handle
        self.title = title
        self.format = Sett.saveformat
        self.filepath = handle.savepath.joinpath(self.title + 
                                                 ".{}".format(Sett.saveformat))
        facet_kws = make_plot.base_kws.copy()
        facet_kws.update(kws)
        # Make canvas:
        if 'Joint' in args:
            pass
        else:
            self.g = self.get_facet(**facet_kws)

    def __call__(self, func, *args, **kws):
        plot_kws = make_plot.base_kws.copy()
        plot_kws.update(kws)
        self.g = func(self, **plot_kws)
        self.add_elements(*args)
        self.save_plot()

    def add_elements(self, *args, **kws):
        if 'centerline' in args:
            self.centerline(**kws)
        if 'ticks' in args:
            self.xticks(**kws)
        if 'labels' in args:
            self.labels(**kws)
        if 'legend' in args:
            self.g.add_legend()
        # if 'centerline' in args:
        #     pass

    def centerline(self, **kws):
        """Plot centerline, i.e. the anchoring point of samples."""
        for ax in self.g.axes.flat:
            __, ytop = ax.get_ylim()
            ax.vlines(self.handle.center, 0, ytop, 'dimgrey', zorder=0,
                      linestyles='dashed')

    def get_facet(self, **kws):
        g = sns.FacetGrid(self.handle.data, row=kws.get('row'),
                          col=kws.get('col'), hue=kws.get('hue'),
                          sharex=kws.get('sharex'), sharey=kws.get('sharey'),
                          gridspec_kws=kws.get('gridspec'),
                          aspect=kws.get('aspect'), legend_out=True,
                          dropna=False, palette=self.handle.palette)
        return g

    def xticks(self, **kws):#xticks=None, yticks=None):
        """Set plot xticks & tick labels to be shown every 5 ticks."""
        xticks = np.arange(0, self.handle.total_length, 5)
        plt.setp(self.g.axes, xticks=xticks, xticklabels=xticks)

    def labels(self, **kws):#xlabel=None, ylabel=None, title=None):
        labels = kws.get('labels')  # !!! Finish
        if 'xlabel' in kws.keys():
            # for ax in self.g.axes:
            plt.xlabel(kws.get('xlabel'))
        if 'ylabel' in kws.keys():
            # for ax in self.g.axes:
            #     ax.
            plt.ylabel(kws.get('ylabel'))
        if 'title' in kws.keys():
            plt.suptitle(self.handle.title, weight='bold', y=kws.get('title_y'))

    def stats(self):
        pass

    def stats(self):
        pass

    def save_plot(self):
        self.g.savefig(str(self.filepath), format=self.format)
        plt.close()


class pfunc:
    """Choose plotting function for make_plot decorator."""

    def lines(plot, *args, **kws):
        err_dict = {'alpha': 0.4}
        g = (plot.g.map_dataframe(sns.lineplot, data=plot.handle.data,
                       x=kws.get('var_name'), y=kws.get('value_name'), ci='sd',
                       err_style='band', hue=kws.get('hue'), dashes=False,
                       alpha=1, palette=plot.handle.palette, err_kws=err_dict))
        return g

    def heatmap():
        pass

    def joint():
        pass

    def totals():
        pass

    def clusters():
        pass

    def widths():
        pass


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
    grouper = kws.get('drop_outlier')
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
    return data


class plotter:

    def __init__(self):
        pass
