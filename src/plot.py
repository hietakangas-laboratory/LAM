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
import plotfuncs as pfunc
# Standard libraries
import warnings
from itertools import combinations, chain
# Packages
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
                data = data.dropna(subset=[m_kws.get('value_name')])
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

    def get_sample_data(self, col_ids, *args, **kws):
        """Collect data from channel-specific sample files."""
        all_data = pd.DataFrame()
        for path in self.paths:
            data = system.read_data(path, header=0, test=False)
            col_list = ['DistBin']
            for key in col_ids:
                col_list.extend([c for c in data.columns if key in c])
                # temp = data.loc[:, data.columns.str.contains(key)]
            sub_data = data.loc[:, col_list].sort_values('DistBin')
            # Test for missing variables:
            for col in sub_data.columns:
                # If no variance, drop data
                if sub_data.loc[:, col].nunique() == 1:
                    sub_data.drop(col, axis=1, inplace=True)
            # Add identifier columns and melt data
            sub_data.loc[:, 'Channel'] = path.stem
            sub_data.loc[:, 'Sample Group'] = str(path.parent.name
                                                  ).split('_')[0]
            if 'melt' in kws.keys():
                m_kws = kws.get('melt')
                sub_data = sub_data.melt(id_vars=m_kws.get('id_vars'),
                                         value_vars=m_kws.get('value_vars'),
                                         var_name=m_kws.get('var_name'),
                                         value_name=m_kws.get('value_name'))
            all_data = pd.concat([all_data, sub_data], sort=True)
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
        self.g = None
        self.filepath = handle.savepath.joinpath(self.title +
                                                 ".{}".format(Sett.saveformat))

    def __call__(self, func, *args, **kws):
        plot_kws = merge_kws(MakePlot.base_kws, kws)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            # Make canvas if needed:
            if 'no_grid' not in args:
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
        if kws.get('sharey') == 'row' or kws.get('sharex') == 'col':
            self.visible_labels(**kws)

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
        fig.savefig(str(self.filepath), format=Sett.saveformat)
        plt.close()

    def visible_labels(self, **kws):
        if (kws.get('sharey') == 'row' or kws.get('sharex') == 'col'):
            for ax in self.g.axes.flat:
                ax.yaxis.set_tick_params(which='both', labelleft=True)
                ax.xaxis.set_tick_params(which='both', labelbottom=True)


class plotting:
    """Make operations for different plots."""
    handle_kws = {'IDs': ['Channel', 'Sample Group'],
                  'melt': {'id_vars': ['Sample Group', 'Channel'],
                           'var_name': 'Linear Position',
                           'value_name': 'Value'},
                  'array_index': 'Sample Group',
                  'drop_grouper': 'Sample Group'}

    def __init__(self, samplegroups, **kws):
        self.kws = plotting.handle_kws.copy()
        self.kws.update(kws)
        self.sgroups = samplegroups

    def add_bivariate(self):
        data_vars = ['Channel', 'Sample Group', 'Sample', 'Type']
        m_kws = {'IDs': data_vars, 'ylabel': 'collect',
                 'xlabel': 'collect', 'title_y': 1,
                 'melt': {'id_vars': data_vars,
                          'value_name': 'Value',
                          'var_name': 'Linear Position'},
                 'plot_kws': {'col': 'Type_X', 'row': 'Type_Y'},
                 'drop_grouper': ['Sample Group', 'Channel', 'Type']}
        new_kws = merge_kws(self.kws, m_kws)
        # If required data hasn't been yet collected
        savepath = self.sgroups.paths.plotdir.joinpath('Versus')
        savepath.mkdir(exist_ok=True)
        add_paths = select(self.sgroups._addData)

        # Get Add data
        all_add_data = pd.DataFrame()
        for channel in Sett.vs_channels:
            paths = [p for p in add_paths if channel == str(p.name)
                     .split('_')[1]]
            if not paths:
                print("-> No data found for {}".format(channel))
                continue
            handle = DataHandler(self.sgroups, paths, savepath)
            add_data = handle.get_data('drop_outlier', **new_kws)
            all_add_data = pd.concat([all_add_data, add_data])
        grouped = all_add_data.groupby('Channel')

        # Make plot:
        combined_grps = combinations(grouped.groups, 2)
        against_self = iter(zip(grouped.groups, grouped.groups))
        for grps in chain(combined_grps, against_self):
            grp, grp2 = grps
            data = grouped.get_group(grp)
            data2 = grouped.get_group(grp2)
            print("  {} vs. {}  ...".format(grp, grp2))
            f_tit = 'Versus_Add {} Data - Add {} Data Matrix'.format(grp, grp2)
            # Take only data types present in both channels:
            diff = set(data.Type.unique()).symmetric_difference(set(
                data2.Type.unique()))
            p_d = data[~data.Type.isin(diff)].index
            p_d2 = data2[~data2.Type.isin(diff)].index
            # Define identifier columns that are in plottable format
            data = data.assign(Type_Y=data['Channel'] + '-' + data['Type'])
            data2 = data2.assign(Type_X=data2['Channel'] + '-' + data2['Type'])
            # Make plot
            plotter = MakePlot(data.loc[p_d, :], handle, f_tit,
                               sec_data=data2.loc[p_d2, :])
            plotter(pfunc.bivariate_kde, 'title', 'legend', 'no_grid',
                    'labels', **new_kws)

    def add_data(self):
        # Collect data:
        data_vars = ['Channel', 'Sample Group', 'Type']
        m_kws = {'IDs': data_vars, 'row': 'Type', 'col': 'Sample Group',
                 'melt': {'id_vars': data_vars,
                          'value_name': 'Value',
                          'var_name': 'Linear Position'},
                 'ylabel': 'collect'}
        new_kws = merge_kws(self.kws, m_kws)
        handle = DataHandler(self.sgroups, self.sgroups._addData)
        all_data = handle.get_data('drop_outlier', **new_kws)
        grouped_data = all_data.groupby('Channel')

        # Make plot:
        for grp, data in grouped_data:
            plotter = MakePlot(data, handle,
                               'Additional Data - {}'.format(grp))
            plotter(pfunc.lines, 'centerline', 'ticks', 'title', 'legend',
                    'labels', **new_kws)

    def chan_bivariate(self):
        savepath = self.sgroups.paths.plotdir.joinpath('Versus')
        savepath.mkdir(exist_ok=True)
        paths1 = select(self.sgroups._addData)
        paths2 = select(self.sgroups._chanPaths, adds=False)

        # Get Add data
        all_add_data = pd.DataFrame()
        for channel in Sett.vs_channels:
            paths = [p for p in paths1 if channel == str(p.name).split('_')[1]]
            if not paths:
                print("-> No data found for {}".format(channel))
                continue
            handle = DataHandler(self.sgroups, paths, savepath)
            data_vars = ['Channel', 'Sample Group', 'Sample', 'Type']
            m_kws = {'IDs': data_vars, 'ylabel': 'collect',
                     'xlabel': 'collect', 'title_y': 1,
                     'melt': {'id_vars': data_vars,
                              'var_name': 'Linear Position',
                              'value_name': 'Value'},
                     'plot_kws': {'col': 'Channel', 'row': 'Type'},
                     'drop_grouper': ['Channel', 'Sample Group', 'Type']}
            new_kws = merge_kws(self.kws, m_kws)
            add_data = handle.get_data('drop_outlier', **new_kws)
            all_add_data = pd.concat([all_add_data, add_data])

        # Get Channel data
        data_vars = ['Channel', 'Sample Group', 'Sample']
        new_kws.update({'IDs': data_vars,
                        'drop_grouper': ['Channel', 'Sample Group']})
        new_kws['melt'].update({'id_vars': data_vars})
        ch_handle = DataHandler(self.sgroups, paths2)
        all_chan_data = ch_handle.get_data('drop_outlier', **new_kws)

        # Make plot:
        grouped = all_add_data.groupby('Channel')
        for grp, data in grouped:
            print("  {}  ...".format(grp))
            f_title = 'Versus_Channels - Add {} Data Matrix'.format(grp)
            plotter = MakePlot(data, handle, f_title, sec_data=all_chan_data)
            plotter(pfunc.bivariate_kde, 'title', 'legend', 'no_grid',
                    'labels', **new_kws)

    def channels(self):
        new_kws = merge_kws(self.kws, {'sharey': 'row'})

        # Collect data:
        handle = DataHandler(self.sgroups, self.sgroups._chanPaths)
        all_data = handle.get_data('drop_outlier', **new_kws)

        # Make plot:
        plotter = MakePlot(all_data, handle, 'Channels - All')
        plotter(pfunc.lines, 'centerline', 'ticks', 'title', 'legend',
                'labels', **new_kws)

    def channel_matrix(self):
        # Collect data:
        paths = self.sgroups.paths.datadir.glob('ChanAvg_*')
        handle = DataHandler(self.sgroups, paths)
        m_kws = {'id_sep': 1, 'IDs': ['Sample Group'], 'kind': 'reg',
                 'diag_kind': 'kde', 'title_y': 1,
                 'xlabel': 'Feature Count',
                 'melt': {'id_vars': ['Sample Group'],
                          'var_name': 'Linear Position',
                          'value_name': 'Value'},
                 'merge_on': ['Sample Group', 'Linear Position']}
        new_kws = merge_kws(self.kws, m_kws)
        all_data = handle.get_data('path_id', 'merge', **new_kws)

        # Make plot:
        plotter = MakePlot(all_data, handle, 'Channels - Matrix')
        plotter(pfunc.channel_matrix, 'title', 'legend', 'no_grid', **new_kws)

    def clusters(self):
        # Find cluster channels from existing data
        cl_chans = [str(p.stem).split('-')[1] for p in
                    self.sgroups.paths.datadir.glob('Clusters-*.csv')]
        if not cl_chans:
            msg = 'No cluster count files found (Clusters_*)'
            print('WARNING: {}'.format(msg))
            lg.logprint(LAM_logger, msg, 'w')
            return

        # Create directory for cluster plots
        savepath = self.sgroups.paths.plotdir.joinpath('Clusters')
        savepath.mkdir(exist_ok=True)

        # SAMPLE-SPECIFIC POSITION PLOTS:
        # Find all cluster data files for each sample
        chan_paths = [c for p in self.sgroups._samplePaths for c in
                      p.glob('*.csv') if c.stem in cl_chans]
        cols = ['Position X', 'Position Y', 'ClusterID']
        kws = {'ylabel': 'Y', 'xlabel': 'X', 'height': 5}
        new_kws = merge_kws(self.kws, kws)
        # Find all channel paths relevant to cluster channels
        for sample in store.samples:
            smpl_paths = [p for p in chan_paths if p.parent.name == sample]
            handle = DataHandler(self.sgroups, smpl_paths, savepath)
            all_data = handle.get_sample_data(cols)
            all_data.index = pd.RangeIndex(stop=all_data.shape[0])

            f_title = "Positions - {}".format(sample)
            sub_ind = all_data.loc[all_data.ClusterID.notnull()].index
            plotter = MakePlot(all_data.loc[sub_ind, :], handle, f_title)
            b_data = all_data.loc[all_data.index.difference(sub_ind), :]
            new_kws.update({'b_data': b_data})
            plotter(pfunc.cluster_positions, 'title', 'labels', **new_kws)

        # CLUSTER HEATMAPS
        paths = list(self.sgroups.paths.datadir.glob('ClNorm_*.csv'))
        if not paths:  # Only if cluster data is found
            msg = 'No normalized cluster count files found (ClNorm_*)'
            print('WARNING: {}'.format(msg))
            lg.logprint(LAM_logger, msg, 'w')
            return

        new_kws = remove_from_kws(self.kws, 'melt')
        new_kws.update({'IDs': ['Channel', 'Sample Group', 'Sample'],
                       'col': None, 'hue': None, 'xlabel': 'Linear Position'})

        # Get and plot heatmap with samples
        handle = DataHandler(self.sgroups, paths, savepath)
        all_data = handle.get_data(array=False, **new_kws)

        all_data.index = all_data.loc[:, 'Sample']
        # Drop unneeded identifiers for 'samples' heatmap
        smpl_data = all_data.drop(['Sample Group', 'Sample'], axis=1)
        plotter = MakePlot(smpl_data, handle, 'Cluster Heatmaps - Samples')
        plotter(pfunc.heatmap, 'centerline', 'ticks', 'title', 'labels',
                **new_kws)

        # Plot sample group averages
        grouped = all_data.groupby(['Channel', 'Sample Group'])
        # Construct a dataframe with averages:
        avg_data = pd.DataFrame()
        for grp, data in grouped:
            temp = pd.Series(data.mean(), name=grp[1])
            temp['Channel'] = grp[0]
            avg_data = avg_data.append(temp)
        # Create plot
        plotter = MakePlot(avg_data, handle, 'Cluster Heatmaps - Groups')
        plotter(pfunc.heatmap, 'centerline', 'ticks', 'title', 'labels',
                **new_kws)

        # CLUSTER LINEPLOT
        m_kws = {'ylabel': 'Clustered cells',
                 'melt': {'id_vars': ['Channel', 'Sample Group'],
                          'var_name': 'Linear Position',
                          'value_name': 'Value'}}
        m_data = all_data.drop('Sample', axis=1)
        m_data = m_data.melt(id_vars=['Channel', 'Sample Group'],
                             var_name='Linear Position',
                             value_name='Value')
        plotter = MakePlot(m_data, handle, 'Cluster Lineplots')
        plotter(pfunc.lines, 'centerline', 'ticks', 'title', 'legend',
                'labels', **m_kws)

    def distributions(self):
        # Channels:
        m_kws = {'IDs': ['Sample Group', 'Channel'], 'title_y': 1,
                 'ylabel': 'Probability density', 'xlabel': 'Feature Count',
                 'melt': {'id_vars': ['Sample Group', 'Channel'],
                          'var_name': 'Linear Position',
                          'value_name': 'Value'},
                 'row': 'Channel', 'col': None, 'aspect': 1.75,
                 'drop_grouper': ['Sample Group', 'Channel']}
        print('  Channels  ...')
        new_kws = merge_kws(self.kws, m_kws)
        # Collect data:
        handle = DataHandler(self.sgroups, self.sgroups._chanPaths)
        all_data = handle.get_data('drop_outlier', **new_kws)
        # Make plot:
        plotter = MakePlot(all_data, handle, 'Distributions - Channels')
        plotter(pfunc.distribution, 'title', 'legend', 'labels', **new_kws)

        # Additional data
        print("  Additional Data  ...")
        id_vars = ['Sample Group', 'Channel', 'Type']
        new_kws.update({'drop_grouper': id_vars, 'col': 'Type',
                        'melt': {'id_vars': ['Sample Group', 'Channel',
                                             'DistBin'],
                                 'var_name': 'Type', 'value_name': 'Value'},
                       'gridspec': {'top': 0.85, 'bottom': 0.2}})
        paths = [p for s in self.sgroups._samplePaths for p in s.glob('*.csv')
                 if p.stem not in ['Vector', 'MPs']]
        # Collect and plot each channel separately:
        for channel in store.channels:
            ch_paths = [p for p in paths if p.stem == channel]
            handle = DataHandler(self.sgroups, ch_paths)
            all_data = handle.get_sample_data(Sett.AddData.keys(),
                                              **new_kws)
            # Make plot:
            p_title = 'Distributions - Additional {} Data'.format(channel)
            plotter = MakePlot(all_data, handle, p_title)
            plotter(pfunc.distribution, 'title', 'legend', 'labels', **new_kws)

    def heatmaps(self):
        # Get and plot _sample group averages_
        HMpaths = self.sgroups.paths.datadir.glob("ChanAvg_*")
        handle = DataHandler(self.sgroups, HMpaths)
        new_kws = remove_from_kws(self.kws, 'melt')
        new_kws.update({'IDs': ['Channel']})
        all_data = handle.get_data(array='Sample Group', **new_kws)
        plotter = MakePlot(all_data, handle, 'Heatmaps - Groups')
        p_kws = {'col': None, 'hue': None}
        plotter(pfunc.heatmap, 'centerline', 'ticks', 'title', **p_kws)

        # Get and plot heatmap with _samples_
        HMpaths = self.sgroups.paths.datadir.glob("Norm_*")
        handle = DataHandler(self.sgroups, HMpaths)
        all_data = handle.get_data(array=False, **new_kws)
        plotter = MakePlot(all_data, handle, 'Heatmaps - Samples')
        # val = all_data.index.unique().size / 5  # Plot height = size depende.
        # p_kws.update({'height': val})
        plotter(pfunc.heatmap, 'centerline', 'ticks', 'title', **p_kws)

    def stat_versus(self, Stats, path):  # !!!
        """Handle statistical data for plots."""
        # Restructure data to be plottable:
        ctrlData = Stats.ctrlData
        tstData = Stats.tstData
        if Sett.Drop_Outliers:  # Drop outliers
            ctrlData = drop_outliers(ctrlData.T, raw=True)
            tstData = drop_outliers(tstData.T, raw=True)
        # Add identifier
        ctrlData.loc[:, 'Sample Group'] = Stats.ctrlGroup
        tstData.loc[:, 'Sample Group'] = Stats.tstGroup
        # Combine data in to one frame and melt it to long format
        plot_data = pd.concat([ctrlData, tstData], ignore_index=True)
        plot_data = plot_data.melt(id_vars=['Sample Group'],
                                 var_name='Linear Position',
                                 value_name='Value')
        # Initialize plotting:
        savepath = Stats.plotDir
        handle = DataHandler(self.sgroups, path, savepath)
        # Give title
        data_name = str(path.stem).split('_')[1:]
        titlep = '-'.join(data_name)
        f_title = "{} = {}".format(Stats.title, titlep)
        # Plot variable
        plotter = MakePlot(plot_data, handle, f_title, sec_data=Stats)
        ylabel = get_unit('_'.join(data_name))
        p_kws = {'col': None, 'row': None, 'ylabel': ylabel,
                 'melt': {'id_vars': ['Sample Group'],
                          'var_name': 'Linear Position',
                          'value_name': 'Value'}}
        if Sett.windowed:
            p_kws.update({'windowed': True})

        plotter(pfunc.lines, 'centerline', 'ticks', 'title', 'stats', 'labels',
                **p_kws)

        # kws = {'id_str': 'Sample Group', 'hue': 'Sample Group', 'height': 4,
        #        'aspect': 3, 'var_str': 'Longitudinal Position',
        #        'value_str': unit, 'centerline': plot_maker.MPbin,
        #        'xlen': self.length, 'title': plot_maker.title, 'Stats': stats,
        #        'title_y': 1, 'fliersize': {'fliersize': '1'}}
        # plot_maker.order = [self.ctrlGroup, self.tstGroup]
        # plot_maker.plot_Data(plotter.catPlot, plot_maker.savepath, **kws)

    def stat_total(self, samplegroups, Stats):
        pass


def select(paths, adds=True):
    """Select paths of defined types of data for versus plot."""
    # Find target names from settings
    add_targets = Sett.vs_adds
    ch_targets = Sett.vs_channels
    # If selecting additional data:
    if adds:
        ret_paths = [p for p in paths if
                     str(p.stem).split('_')[1] in ch_targets and
                     str(p.stem).split('_')[2].split('-')[0] in add_targets]
        return ret_paths
    # If selecting channel counts:
    ret_paths = [p for p in paths if str(p.stem).split('_')[1] in ch_targets]
    return ret_paths


def drop_func(x, mean, drop_value):
    if np.abs(x - mean) <= drop_value:
        return x
    return np.nan


def drop_outliers(all_data, melted=False, raw=False, **kws):
    def drop(data, cols):
        """Drop outliers from a dataframe."""
        # Get mean and std of input data
        if raw:
            values = data
        else:
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

    if raw:
        all_data = drop(all_data, cols=None)
        return all_data
    # Handle data for dropping
    if 'drop_grouper' in kws.keys():
        grouper = kws.get('drop_grouper')
    else:
        grouper = 'Sample Group'
    grp_data = all_data.groupby(by=grouper)
    if melted:
        names = kws['melt'].get('value_name')
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
    if string in ("Distance Means", "Width"):  # !!!
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
        return 'Value'
    # Otherwise, build label from the sub-units
    label = Sett.AddData.get(key)[1]
    if 'chan' in locals():
        label = '{}, '.format(chan) + label
    if 'key_c' in locals():
        if Sett.replaceID and key_c in Sett.channelID.keys():
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
