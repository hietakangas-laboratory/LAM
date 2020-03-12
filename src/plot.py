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

    def __init__(self, samplegroups,paths, *args, **kws):
        self.savepath = samplegroups.paths.plotdir
        self.palette = samplegroups._grpPalette
        self.center = samplegroups._center
        self.total_length = samplegroups._length
        self.MPs = samplegroups._AllMPs
        self.data = pd.DataFrame()
        self.paths = paths

    def data_array(self):
        pass

    def get_add_vars(self):
        pass

    def get_data(self, *args, **kws):
        melt = False
        all_data = pd.DataFrame()
        for path in self.paths:
            # !!! Add samplegroup identifier
            data = system.read_data(path, header=0, test=False)
            if 'IDs' in kws.keys():
                data = identifiers(data, path, kws.get('IDs'))
            if 'melt' in kws.keys():
                melt = True
                melt_kws = kws.get('melt')
                if 'Pair' in args:
                    chan = path.stem.split('_')[1]
                    melt_kws.update({'value_name': chan})
                data = self.melt_data(data, **melt_kws)
            elif 'array' in args:
                # data = self.data_array(data)
                pass
            if 'Pair' in args:
                if all_data.empty:
                    all_data = data
                else:
                    all_data = all_data.merge(data, how='outer', copy=False,
                                              on=['Sample Group',
                                                  'Linear Position'])
                continue
            all_data = pd.concat([all_data, data], sort=True)
        all_data.index = pd.RangeIndex(stop=all_data.shape[0])
        if 'drop_outlier' in args and Sett.Drop_Outliers:
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
                'height': 3, 'aspect': 3, 'flier_size': 2, 'title_y': 0.95,
                'sharex': False, 'sharey': False, 'gridspec': {'hspace': 0.45},
                'xlabel': 'Linear Position', 'ylabel': 'Feature Count'}

    def __init__(self, data, handle, title, *args, **kws):
        self.data = data
        self.plot_error = False
        self.handle = handle
        self.title = title
        self.format = Sett.saveformat
        self.filepath = handle.savepath.joinpath(self.title +
                                                 ".{}".format(Sett.saveformat))

    def __call__(self, func, *args, **kws):
        plot_kws = merge_kws(make_plot.base_kws, kws)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            # Make canvas if needed:
            if 'no_grid' not in args:
                if 'Joint' in args:
                    pass
                else:
                    self.g = self.get_facet(**plot_kws)
            # Plot data
            self.g = func(self, **plot_kws)
        if self.g is None and self.plot_error:
            print('Whoops!')
            # !!! HANDLE ERRORS PROPERLY
        self.add_elements(*args, **plot_kws)
        self.save_plot()

    def add_elements(self, *args, **kws):
        if 'centerline' in args:
            self.centerline(**kws)
        if 'ticks' in args:
            self.xticks(**kws)
        if 'collect_labels' in args:
            new_labels = self.get_labels(**kws)
            kws.update({'ylabel': new_labels})
        if 'labels' in args:
            self.labels(**kws)
        if 'legend' in args:
            self.g.add_legend()
        if 'title' in args:
            self.set_title(**kws)

    def centerline(self, **kws):
        """Plot centerline, i.e. the anchoring point of samples."""
        for ax in self.g.axes.flat:
            __, ytop = ax.get_ylim()
            ax.vlines(self.handle.center, 0, ytop, 'dimgrey', zorder=0,
                      linestyles='dashed')

    def get_facet(self, **kws):
        g = sns.FacetGrid(self.data, row=kws.get('row'),
                          col=kws.get('col'), hue=kws.get('hue'),
                          sharex=kws.get('sharex'), sharey=kws.get('sharey'),
                          gridspec_kws=kws.get('gridspec'),
                          height=kws.get('height'), aspect=kws.get('aspect'),
                          legend_out=True, dropna=False,
                          palette=self.handle.palette)
        return g

    def get_labels(self, **kws):  # !!! FIX Y-LABELING
        labels = {}
        for path in self.handle.paths:
            name = '_'.join(str(path.stem).split('_')[1:])
            # Get unit of data
            sub_names = name.split('_')
            key_name = sub_names[1].split('-')[0]
            if "Distance Means" in key_name:
                label = "Distance"
            else:
                temp = Sett.AddData.get(key_name)
                if temp is None:
                    label = 'Value'
                else:
                    label = temp[1]
            labels.update({key_name: label})
        row_order = self.data[kws.get('row')].unique()
        new_labels = [labels.get(k) for k in row_order]
        return new_labels

    def xticks(self, **kws):#xticks=None, yticks=None):
        """Set plot xticks & tick labels to be shown every 5 ticks."""
        xticks = np.arange(0, self.handle.total_length, 5)
        plt.setp(self.g.axes, xticks=xticks, xticklabels=xticks)

    def labels(self, **kws):#xlabel=None, ylabel=None, title=None):
        labels = kws.get('labels')  # !!! Finish
        if 'xlabel' in kws.keys():
            for ax in self.g.axes.flat:
                ax.set_xlabel(kws.get('xlabel'))
        if 'ylabel' in kws.keys():
            labels = kws.get('ylabel')
            ln = bool(isinstance(labels, list) & len(labels) > 1)
            if not ln:
                for ax in self.g.axes.flat:
                    ax.set_ylabel(labels)
            else:
                for ind, ax in enumerate(self.g.axes):
                    ax.set_ylabel(labels[ind])
                # nrows, ncols = self.g.axes.shape
                # index = [[i_1, i_2] for i_1 in np.arange(nrows) for i_2 in
                #           np.arange(ncols)]
                # for ind in index:
                #     self.g.axes[ind[0], ind[1]].set_ylabel(labels[ind[0]])

    def set_title(self, **kws):
        self.g.fig.suptitle(self.title, weight='bold', y=kws.get('title_y'))

    def stats(self):
        pass

    def save_plot(self):
        fig = plt.gcf()
        fig.savefig(str(self.filepath), format=self.format)
        # self.g.savefig(str(self.filepath), format=self.format)
        plt.close()


class pfunc:
    """Choose plotting function for make_plot decorator."""

    def lines(plot, *args, **kws):
        err_dict = {'alpha': 0.3}
        data = plot.data.dropna()
        melt_kws = kws.get('melt')
        g = (plot.g.map_dataframe(sns.lineplot, data=plot.data,
                                  x=data.loc[:, melt_kws.get('var_name')
                                             ].astype(int),
                                  y=data.loc[:, melt_kws.get('value_name')],
                                  ci='sd', err_style='band',
                                  hue=kws.get('hue'), dashes=False, alpha=1,
                                  palette=plot.handle.palette,
                                  err_kws=err_dict))
        return g

    def pairs(plot, *args, **kws):
        """Creation of pair plots."""
        # Settings for plotting:
        pkws = {'x_ci': None, 'truncate': True, 'order': 3,
                'scatter_kws': {'linewidth': 0.05, 's': 10, 'alpha': 0.3},
                'line_kws': {'alpha': 0.6, 'linewidth': 1.5}}
        if Sett.plot_jitter:
            pkws.update({'x_jitter': 0.49, 'y_jitter': 0.49})
        
        # Drop unneeded data and replace NaN with zero (required for plot)
        data = plot.data.sort_values(by="Sample Group").drop(
            'Linear Position', axis=1)
        data = data.dropna(how='all',
                           subset=data.columns[data.columns != 'Sample Group']
                           ).replace(np.nan, 0)
        try:
            g = sns.pairplot(data=data, hue=kws.get('hue'),
                             height=1.5, aspect=1,
                             kind=kws.get('kind'),
                             diag_kind=kws.get('diag_kind'),
                             palette=plot.handle.palette,
                             plot_kws=pkws,
                             diag_kws= {'linewidth': 2})
            # Set bottom values to zero, as no negatives in count data
            for ax in g.axes.flat:
                ax.set_ylim(bottom=0)
                ax.set_xlim(left=0)
        # In case of missing or erroneous data, linalgerror can be raised
        except np.linalg.LinAlgError:  # Then, exit plotting
            msg = '-> Confirm that all samples have proper channel data'
            fullmsg = 'Pairplot singular matrix\n{}'.format(msg)
            lg.logprint(LAM_logger, fullmsg, 'ex')
            print('ERROR: Pairplot singular matrix')
            print(msg)
            plot.plot_error = True
            return None
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
    if 'Type' in ids:
        name = str(path.stem).split('_')[2:]
        data.loc['Type', :] = name
    return data


def merge_kws(kws1, kws2):
    new_kws = kws1.copy()
    if kws2 is not None:
        new_kws.update(kws2)
    return new_kws

class plotter:
    """Fake class"""

    def __init__(self, plotData, savepath, center=0, title=None,
                 palette=None, color='b'):
        pass