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

    def __init__(self, samplegroups, paths):
        self.savepath = samplegroups.paths.plotdir
        self.palette = samplegroups._grpPalette
        self.center = samplegroups._center
        self.total_length = samplegroups._length
        self.MPs = samplegroups._AllMPs
        self.data = pd.DataFrame()
        self.paths = paths

    def data_array_index(self, data, indexer=None):
        if indexer is not None:
            data.index = data.loc[:, indexer]
            data.drop(indexer, axis=1, inplace=True)
        return data

    def get_add_vars(self):
        pass

    def get_data(self, *args, **kws):
        melt = False
        all_data = pd.DataFrame()
        for path in self.paths:
            data = system.read_data(path, header=0, test=False)
            if 'IDs' in kws.keys():
                data = identifiers(data, path, kws.get('IDs'))
            if 'melt' in kws.keys():
                melt = True
                m_kws = kws.get('melt')
                if 'Matrix' in args:
		    id_sep = kws['Matrix'].get('id_sep')
                    id_var = path.stem.split('_')[id_sep:]
                    m_kws.update({'value_name': id_var})
                data = data.T.melt(id_vars=m_kws.get('id_vars'),
                                   value_vars=m_kws.get('value_vars'),
                                   var_name=m_kws.get('var_name'),
                                   value_name=m_kws.get('value_name'))
            else:
                data = data.T
            if 'Matrix' in args:
                if all_data.empty:
                    all_data = data
                else:
                    all_data = all_data.merge(data, how='outer', copy=False,
                                              on=['Sample Group',
                                                  'Linear Position'])
                continue
            all_data = pd.concat([all_data, data], sort=True)
        if 'array' in args:
            all_data = self.data_array_index(all_data, kws.get('array_index'))
        if 'drop_outlier' in args and Sett.Drop_Outliers:
            all_data.index = pd.RangeIndex(stop=all_data.shape[0])
            all_data = drop_outliers(all_data, melt, **kws)
        all_data = all_data.infer_objects()
        return all_data

    def get_add_data(self, *args, **kws):
        all_data = pd.DataFrame()
        for path in self.paths:
            data = system.read_data(path, header=0, test=False)
            data = identifiers(data, path, kws.get('IDs'))
            m_kws = kws.get('melt')
            id_var = path.stem.split('_')[2]
            m_kws.update({'value_name': id_var})
            data = data.T.melt(id_vars=m_kws.get('id_vars'),
                               value_vars=m_kws.get('value_vars'),
                               var_name=m_kws.get('var_name'),
                               value_name=m_kws.get('value_name'))
            all_data = all_data.merge(data, how='outer', copy=False,
                                      on=['Sample Group',
                                          'Linear Position'])


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
        if 'collect_labels' in args:
            new_labels = self.get_labels(**kws)
            kws.update({'ylabel': new_labels})
	    #if 'copy_YtoX' in args:
	        #kws.update({'xlabel': new_labels}
        if 'labels' in args:
            self.labels(**kws)
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
        types = self.data[kws.get('row')].unique()
        row_order = sorted([s.split('-')[0] for s in types])
        new_labels = [labels.get(k) for k in row_order]
        return new_labels

    def xticks(self):#xticks=None, yticks=None, , **kws):
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
            length_test = (isinstance(labels, list) and len(labels) > 1)
            # Test whether multiple keys are present:
            if not length_test:
                for ax in self.g.axes.flat:
                    ax.set_ylabel(labels)
            else:  # Change labels dependently on grid position
                nrows, ncols = self.g.axes.shape
                index = [(i_1, i_2) for i_1 in np.arange(nrows) for i_2 in
                          np.arange(ncols)]
                for ind in index:
                    self.g.axes[ind].set_ylabel(labels[ind[0]])

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
