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
        melt = False
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
                melt = True
            all_data = pd.concat([all_data, sub_data], sort=True)
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
    # Colors for fills
    LScolors = sns.color_palette('Reds', n_colors=4)
    GRcolors = sns.color_palette('Blues', n_colors=4)

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
            self.labels(kws.get('xlabel'), kws.get('ylabel'),
                        kws.get('label_first_only'))
        if 'legend' in args:
            self.g.add_legend()
        if 'title' in args:
            self.set_title(**kws)
        if 'stats' in args:
            self.stats(**kws)
        if 'total_stats' in args:
            self.stats_total(**kws)
        if (kws.get('sharey') == 'row' or kws.get('sharex') == 'col'):
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
            ax.set_title(' | '.join(label_strs))

    def get_facet(self, **kws):
        g = sns.FacetGrid(self.data, row=kws.get('row'),
                          col=kws.get('col'), hue=kws.get('hue'),
                          sharex=kws.get('sharex'), sharey=kws.get('sharey'),
                          gridspec_kws=kws.get('gridspec'),
                          height=kws.get('height'), aspect=kws.get('aspect'),
                          legend_out=True, dropna=False,
                          palette=self.handle.palette)
        return g

    def labels(self, xlabel=None, ylabel=None, first=None):
        for ax in self.g.axes.flat:
            if xlabel not in (None, 'collect'):
                ax.set_xlabel(xlabel)
            if ylabel not in (None, 'collect'):
                ax.set_ylabel(ylabel)
            if first:
                return

    def plot_significance(self, ix, row, ax, yaxis, yheight, fill=Sett.fill,
                          stars=Sett.stars):
        # If both hypothesis rejections have same value, continue
        if row[3] == row[6]:
            return
        xaxis = [ix-0.43, ix+0.43]
        if row[3] is True:  # ctrl is greater
            pStr, color = significance_marker(row[1], MakePlot.LScolors)
        elif row[6] is True:  # ctrl is lesser
            pStr, color = significance_marker(row[4], MakePlot.GRcolors)
        if fill:
            ax.fill_between(xaxis, yaxis, color=color, alpha=0.35, zorder=0)
        if stars:
            ax.annotate(pStr, (ix, yheight), fontsize=8, ha='center')

    def set_title(self, **kws):
        self.g.fig.suptitle(self.title, weight='bold', y=kws.get('title_y'))

    def stats(self, **kws):
        stats = self.sec_data.statData
        __, ytop = plt.ylim()
        tytop = ytop*1.35
        ax = plt.gca()
        ax.set_ylim(top=tytop)
        yaxis = [tytop, tytop]

        # Create secondary axis for significance plotting
        ax2 = plt.twinx()
        lkws = {'alpha': 0.85}
        xmin, xtop = stats.index.min(), stats.index.max()
        ax2.plot((xmin, xtop), (0, 0), linestyle='dashed', color='grey',
                 linewidth=0.85, **lkws)
        # Find top of original y-axis and create a buffer for twin to
        # create a prettier plot
        botAdd = 2.75*-Sett.ylim
        ax2.set_ylim(bottom=botAdd, top=Sett.ylim)
        ax2.set_yticks(np.arange(0, Sett.ylim, 5))
        ax2.set_yticklabels(np.arange(0, Sett.ylim, 5))
        ax2.yaxis.set_label_coords(1.04, 0.85)

        # Creation of -log2 P-value axis and line plot
        if Sett.negLog2:
            Sett.stars = False  # Force stars to be False when plotting neglog
            Y = stats.iloc[:, 7]
            X = Y.index.tolist()
            # Find locations where the log line should be drawn
            ind = Y[Y.notnull()].index
            logvals = pd.Series(np.zeros(Y.shape[0]), index=Y.index)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                logvals.loc[ind] = np.log2(Y[ind].astype(np.float64))
            # Create twin axis with -log2 P-values
            ax2.plot(X, np.negative(logvals), color='dimgrey', linewidth=1.5,
                     **lkws)
            ax2.set_ylabel('P value\n(-log2)')
        # Create significance stars and color fills
        for index, row in stats.iterrows():
            self.plot_significance(index, row, ax2, yaxis, yheight=0)
        # Add info on sliding window to plot
        if 'windowed' in kws:
            comment = "Window: lead {}, trail {}".format(Sett.lead, Sett.trail)
            plt.annotate(comment, (5, 5), xycoords='figure pixels')

    def stats_total(self, **kws):
        # Loop through the plot axes
        order = kws.get('x_order')
        ctrl_x = order.index(Sett.cntrlGroup)
        for ind, ax in enumerate(self.g.axes.flat):
            # Find rejected H0 for current axis
            row = self.sec_data.iloc[ind, :]
            rejects = row.iloc[row.index.get_level_values(1).str.contains(
                'Reject')].where(row).dropna()
            rejectN = np.count_nonzero(rejects.to_numpy())
            ax.set_ylim(bottom=0)
            if rejectN > 0:  # If any rejected H0
                # Raise y-limit of axis to fit significance plots
                __, ytop = ax.get_ylim()
                tytop = ytop*1.3
                ax.set_ylim(top=tytop)
                # Find heights for significance lines
                heights = np.linspace(ytop, ytop*1.15, rejectN)
                # Loop groups with rejected H0
                for i, grp in enumerate(rejects.index.get_level_values(0)):
                    y = heights[i]  # Get height for the group's line
                    grp_x = order.index(grp)  # Get x-axis location of group
                    line = sorted([grp_x, ctrl_x])
                    # Plot line
                    ax.hlines(y=y, xmin=line[0], xmax=line[1], color='dimgrey')
                    # Locate P-value and get significance stars
                    Pvalue = row.loc[(grp, 'P Two-sided')]
                    pStr, _ = significance_marker(Pvalue, vert=True)
                    # Define plot location for stars and plot
                    ax.annotate(pStr, (line[0]+.5, y), ha='center')

    def save_plot(self):
        fig = plt.gcf()
        fig.savefig(str(self.filepath), format=Sett.saveformat)
        plt.close()

    def visible_labels(self, **kws):
        for ax in self.g.axes.flat:
            ax.yaxis.set_tick_params(which='both', labelleft=True)
            ax.set_ylabel(visible=True)
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
            ax.set_xlabel(visible=True)

    def xticks(self):
        """Set plot xticks & tick labels to be shown every 5 ticks."""
        xticks = np.arange(0, self.handle.total_length, 5)
        plt.setp(self.g.axes, xticks=xticks, xticklabels=xticks)


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
                 'ylabel': 'collect', 'sharey': 'row'}
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
        m_kws = {'IDs': ['Sample Group', 'Channel'], 'title_y': 0.95,
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
        # id_vars = ['Sample Group', 'Channel', 'Type']
        new_kws.update({'drop_grouper': ['Sample Group', 'Type'],
                        'row': 'Type', 'col': None,
                        'melt': {'id_vars': ['Sample Group', 'Channel',
                                             'DistBin'],
                                 'var_name': 'Type', 'value_name': 'Value'},
                        'gridspec': {'left': 0.15, 'right': 0.8,
                                     'hspace': 0.45}})
        paths = [p for s in self.sgroups._samplePaths for p in s.glob('*.csv')
                 if p.stem not in ['Vector', 'MPs']]
        # Collect and plot each channel separately:
        for channel in store.channels:
            print("    {}  ...".format(channel))
            ch_paths = [p for p in paths if p.stem == channel]
            handle = DataHandler(self.sgroups, ch_paths)
            all_data = handle.get_sample_data(Sett.AddData.keys(),
                                              'drop_outlier', **new_kws)
            # Make plot:
            p_title = 'Distributions - Additional {} Data'.format(channel)
            plotter = MakePlot(all_data, handle, p_title)
            plotter(pfunc.distribution, 'title', 'legend', 'labels', **new_kws)

    def heatmaps(self):
        # Get and plot _sample group averages_
        HMpaths = self.sgroups.paths.datadir.glob("ChanAvg_*")
        handle = DataHandler(self.sgroups, HMpaths)
        new_kws = remove_from_kws(self.kws, 'melt')
        new_kws.update({'IDs': ['Channel', 'Sample Group']})
        all_data = handle.get_data(array='Sample Group', **new_kws)
        all_data.index = all_data['Sample Group'].tolist()
        all_data.drop('Sample Group', axis=1, inplace=True)
        plotter = MakePlot(all_data, handle, 'Heatmaps - Groups')
        p_kws = {'col': None, 'hue': None}
        plotter(pfunc.heatmap, 'centerline', 'ticks', 'title', **p_kws)

        # Get and plot heatmap with _samples_
        HMpaths = self.sgroups.paths.datadir.glob("Norm_*")
        handle = DataHandler(self.sgroups, HMpaths)
        new_kws.update({'IDs': ['Channel', 'Sample']})
        all_data = handle.get_data(array=False, **new_kws)
        all_data.index = all_data['Sample'].tolist()
        all_data.drop('Sample', axis=1, inplace=True)
        plotter = MakePlot(all_data, handle, 'Heatmaps - Samples')
        p_kws.update({'Sample_plot': True})
        plotter(pfunc.heatmap, 'centerline', 'ticks', 'title', **p_kws)

    def stat_totals(self, total_stats, path):
        plot_data = total_stats.data
        ctrlN = int(len(total_stats.groups) / 2)
        order = total_stats.tstGroups
        order.insert(ctrlN, Sett.cntrlGroup)

        # Melt data to long form and drop missing observation points
        plot_data = pd.melt(plot_data, id_vars=['Sample Group', 'Variable'],
                            var_name='Linear Position',
                            value_name='Value')
        plot_data = plot_data.dropna(subset=['Value'])
        # Make sure that data is in float format
        plot_data['Value'] = plot_data['Value'].astype('float64')
        # Assign variable indication the order of plotting
        plot_data['Ord'] = plot_data.loc[:, 'Sample Group'].apply(order.index)
        plot_data.sort_values(by=['Ord', 'Variable'], axis=0, inplace=True)
        # Find group order number for control group for plotting significances
        # total_stats.statData.sort_index(inplace=True)
        # Create plot:
        savepath = total_stats.plotDir
        handle = DataHandler(self.sgroups, path, savepath)
        plotter = MakePlot(plot_data, handle, total_stats.filename,
                           sec_data=total_stats.statData)
        p_kws = {'row': None, 'col': 'Variable', 'x_order': order,
                 'height': 3, 'aspect': 1, 'title_y': 1,
                 'ylabel': 'collect', 'xlabel': 'Sample Group',
                 'gridspec': {'wspace': 0.25}}
        plotter(pfunc.violin, 'title', 'total_stats', 'labels', 'legend',
                **p_kws)

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
        ylabel = get_unit(data_name[-1])
        p_kws = {'col': None, 'row': None, 'ylabel': ylabel,
                 'label_first_only': True,
                 'melt': {'id_vars': ['Sample Group'],
                          'var_name': 'Linear Position',
                          'value_name': 'Value'},
                 'gridspec': {'bottom': 0.2}}
        if Sett.windowed:
            p_kws.update({'windowed': True})

        plotter(pfunc.lines, 'centerline', 'ticks', 'title', 'stats', 'labels',
                'legend', **p_kws)

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


def drop_outliers(all_data, melted=False, raw=False, **kws):
    def drop(data, col):
        """Drop outliers from a dataframe."""
        # Get mean and std of input data
        if raw:
            values = data
        else:
            values = data.loc[:, col].sort_values(ascending=False)
        with warnings.catch_warnings():  # Ignore empty bin warnings
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean = np.nanmean(values.astype('float'))
            std = np.nanstd(values.astype('float'))
        drop_val = Sett.dropSTD * std
        if raw:  # If data is not melted, replace outliers with NaN
            data.where(np.abs(values - mean) <= drop_val, other=np.nan,
                       inplace=True)
        else:  # If data is melted and sorted, find indexes until val < drop
            idx = []
            for ind, val in values.iteritems():
                if np.abs(val - mean) < drop_val:
                    break
                idx.append(ind)
            # Select data that fills criteria for validity
            data = data.loc[(data.index.difference(idx)), :]
        return data

    if raw:
        all_data = drop(all_data, col=None)
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
    all_data = grp_data.apply(lambda grp: drop(grp, col=names))
    if isinstance(all_data.index, pd.MultiIndex):
        all_data = all_data.droplevel(grouper)
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


def significance_marker(value, colors=MakePlot.GRcolors, vert=False):
    """Designation of number of significance stars."""
    if value <= 0.001:
        pStr = ["*", "*", "*"]
        color = colors[3]
    elif value <= 0.01:
        pStr = ["*", "*"]
        color = colors[2]
    elif value <= Sett.alpha:
        if value <= 0.05:
            pStr = ["*"]
        else:
            pStr = [""]
        color = colors[1]
    else:
        pStr = [" "]
        color = colors[0]
    if vert:
        ret_str = ' '.join(pStr)
    else:
        ret_str = '\n'.join(pStr)
    return ret_str, color
