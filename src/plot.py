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
                           var_name=kws.get('var_name'),
                           value_name=kws.get('value_name'))
        return data

    # def call_plot(self, func, plot_kws=base_kws):  # needed ???
    #     plot.make_plot(func)


class make_plot:
    """Create decorated plots."""

    # Base keywords utilized in plots.
    base_kws = {'hue': 'Sample Group', 'row': 'Sample Group', 'height': 5,
                'aspect': 3, 'flier_size': 2, 'title_y': 0.95, 'sharey': True,
                'gridspec': {'hspace': 0.3}}

    def __init__(self, func, plot_data):
        pass

    def __call__(self):
        pass

    def centerline(self):
        pass

    def ticks(self, xticks=None, yticks=None):
        pass

    def labels(self, xlabel=None, ylabel=None, title=None):
        pass

    def x_ticks(self):
        pass

    def stats(self):
        pass

    def save_plot(self):
        pass


class plot_funcs:
    """Choose plotting function for make_plot decorator."""

    def __init__(self, func):
        func()

    # @make_plot
    # def lines():
    #     pass

    # @make_plot
    # def heatmap():
    #     pass

    # @make_plot
    # def joint():
    #     pass

    # @make_plot
    # def totals():
    #     pass

    # @make_plot
    # def clusters():
    #     pass

    # @make_plot
    # def widths():
    #     pass


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
        data.loc['Channel', :] = data.columns
    return data


class plotter:

    def __init__(self):
        pass
