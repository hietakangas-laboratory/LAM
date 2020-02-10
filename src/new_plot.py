# -*- coding: utf-8 -*-
"""
LAM-module for plot creation.

Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""

# LAM modules
from settings import settings as Sett
import logger as lg
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
    
    # Base keywords utilized in plots.
    basekws = {'id_str': 'Sample Group', 'hue': 'Sample Group',
               'row': 'Sample Group', 'height': 5, 'aspect': 3,
               'var_str': 'Longitudinal Position', 'flierS': 2,
               'title_y': 0.95, 'sharey': True,
               'gridspec': {'hspace': 0.3}}
    
    def __init__(self, samplegroups):
        
        
    def melt_data(**kws):
        """Melt dataframes to long form."""
        if 'var_str' in kws.keys():
            varname = kws.get('var_str')
        else:
            varname = 'variable'
        if 'value_str' in kws.keys():
            valname = kws.get('value_str')
        else:
            valname = 'value'
        plotData = pd.melt(self.data, id_vars=kws.get('id_str'),
                           value_name=valname, var_name=varname)
        return plotData, varname, valname

    def select(paths, adds=True):
        """Find different types of data for versus plot."""
        retPaths = []
        # Find target names from settings
        if adds:
            targets = Sett.vs_adds
        else:
            targets = Sett.vs_channels
        for trgt in targets:  # For each target, find corresponding file
            if adds:  # File name dependent on data type
                namer = "^Avg_.*_{}.*".format(trgt)
            else:
                namer = "^Norm_{}.*".format(trgt)
            reg = re.compile(namer, re.I)
            selected = [p for p in paths if reg.search(str(p.stem))]
            retPaths.extend(selected)  # Add found paths to list
        return retPaths
    
    def base(paths, func, ylabel='Cell Count', name_sep=1,
               **kws):
        """
        General plotting for LAM, i.e. variable on y-axis, and bins on x.
    
        Args:
        ----
        Name_sep : int
            the start of data's name when file name is split by '_', e.g.
            name_sep=1 would name the data as DAPI when file name is
            'Avg_DAPI'.
        """
        savepath = self._plotDir
        # For each data file to be plotted:
        for path in paths:
            # Find whether outliers are to be dropped
            dropB = Sett.Drop_Outliers
            # Read data from file and the pass it on to the plotter-class
            plotData, name, cntr = self.read_channel(path, self._groups,
                                                     drop=dropB,
                                                     name_sep=name_sep)
            plot_maker = plotter(plotData, self._plotDir, center=cntr,
                                 title=name, palette=self._grpPalette)
            # Give additional keywords for plotting
            kws2 = {'centerline': plot_maker.MPbin, 'value_str': ylabel,
                    'title': plot_maker.title, 'xlen': self._length,
                    'ylabel': ylabel}
            kws2.update(basekws)  # update to include the base keywords
            # If no given y-label, get name from file name / settings:
            if ylabel is None:  # Find labels for additional data
                addName = plot_maker.title.split('-')[0].split('_')[1]
                if "DistanceMeans" in addName:
                    newlabel = "Distance"
                else:
                    temp = Sett.AddData.get(addName)
                    if temp is None:
                        newlabel = 'Cell Count'
                    else:
                        newlabel = temp[1]
                kws2.update({'ylabel': newlabel, 'value_str': newlabel})
            kws2.update(kws)  # Update with kws passed to this function
            plot_maker.plot_Data(func, savepath, **kws2)  # Plotting
    
    def heat(paths, samples=False):
        """Create heatmaps of cell counts for each sample group."""
        savepath = self._plotDir
        fullData = pd.DataFrame()
        # Loop through the given channels and read the data file. Each
        # channel is concatenated to one dataframe for easier plotting.
        for path in paths:
            plotData, name, cntr = self.read_channel(path, self._groups)
            # change each sample's group to be contained in its index
            # within the dataframe.
            if not samples:
                plotData.index = plotData.loc[:, 'Sample Group']
            plotData.drop(labels='Sample Group', axis=1, inplace=True)
            # Channel-variable is added for identification
            plotData.loc[:, 'Channel'] = name
            fullData = pd.concat([fullData, plotData], axis=0, copy=False)
        # The plotting can't handle NaN's, so they are changed to zero.
        fullData = fullData.replace(np.nan, 0)
        if not samples:
            name = "All Channels Heatmaps"
        else:
            name = "All Samples Channel Heatmaps"
        # Initialize plotting-class, create kws, and plot all channel data.
        plot_maker = plotter(fullData, self._plotDir, center=cntr,
                             title=name, palette=None)
        kws = {'height': 3, 'aspect': 5, 'sharey': False, 'row': 'Channel',
               'title_y': 0.93, 'center': plot_maker.MPbin,
               'xlen': self._length, 'gridspec': {'hspace': 0.5}}
        if samples:  # MAke height of plot dependant on sample size
            val = fullData.index.unique().size / 2
            kws.update({'height': val})
        plot_maker.plot_Data(plotter.Heatmap, savepath, **kws)
    
    def versus(paths1, paths2=None, folder=None):
        """Creation of bivariant jointplots."""
        if folder:
            savepath = self._plotDir.joinpath(folder)
        else:
            savepath = self._plotDir
        savepath.mkdir(exist_ok=True)
        # Pass given paths to a function that pairs each variable
        self.Joint_looper(paths1, paths2, savepath)
    
    def pair():
        """Create pairplot-grid, i.e. each channel vs each channel."""
        allData = pd.DataFrame()
        # Loop through all channels.
        for path in self._dataDir.glob('ChanAvg_*'):
            # Find whether to drop outliers and then read data
            dropB = Sett.Drop_Outliers
            plotData, __, cntr = self.read_channel(path, self._groups,
                                                   drop=dropB)
            # get channel name from path, and add identification ('Sample')
            channel = str(path.stem).split('_')[1]
            # Change data into long form (one observation per row):
            plotData = pd.melt(plotData, id_vars=['Sample Group'],
                               var_name='Longitudinal Position',
                               value_name=channel)
            if allData.empty:
                allData = plotData
            else:  # Merge data so that each row contains all channel
                # counts from one bin of one sample
                allData = allData.merge(plotData, how='outer', copy=False,
                                        on=['Sample Group',
                                            'Longitudinal Position'])
        name = 'All Channels Pairplots'
        # Initialize plotter, create plot keywords, and then create plots
        plot_maker = plotter(allData, self._plotDir, title=name,
                             center=cntr, palette=self._grpPalette)
        kws = {'hue': 'Sample Group', 'kind': 'reg', 'diag_kind': 'kde',
               'height': 3.5, 'aspect': 1, 'title_y': 1}
        plot_maker.plot_Data(plotter.pairPlot, self._plotDir, **kws)
    
    def distributions():
        """Create density distribution plots of different data types."""
        savepath = self._plotDir.joinpath('Distributions')
        savepath.mkdir(exist_ok=True)
        kws = {'hue': 'Sample Group', 'row': 'Sample Group', 'height': 5,
               'sharey': True, 'aspect': 1, 'title_y': 0.95,
               'gridspec': {'hspace': 0.4}}
        ylabel = 'Density'
        # Channels
        print("Channels  ...")
        for path in self._dataDir.glob('All_*'):
            plotData, name, cntr = self.read_channel(path, self._groups)
            plot_maker = plotter(plotData, self._plotDir, center=cntr,
                                 title=name, palette=self._grpPalette)
            xlabel = 'Cell Count'
            kws.update({'id_str': 'Sample Group', 'var_str': xlabel,
                        'ylabel': ylabel, 'value_str': ylabel})
            plot_maker.plot_Data(plotter.distPlot, savepath, **kws)
    
        # Additional data
        for key in Sett.AddData.keys():
            print("{}  ...".format(key))
            ban = ['Group', 'Channel']
            AllVals = pd.DataFrame(columns=[key, 'Group', 'Channel'])
            paths = [p for s in self._samplePaths for p in s.glob('*.csv')
                     if p.stem not in ['MPs', 'Vector']]
            # read and concatenate all found data files:
            for path in paths:
                try:
                    data = system.read_data(path, header=0, test=False,
                                            index_col=False)
                    values = data.loc[:, data.columns.str.contains(key)]
                    group = str(path.parent.name).split('_')[0]
                    channel = str(path.stem)
                    # Assign identification columns
                    values = values.assign(Group=group, Channel=channel)
                    for col in values.loc[:, ~values.columns.isin(ban)
                                          ].columns:
                        # If no variance, drop data
                        if values.loc[:, col].nunique() == 1:
                            values.loc[:, str(col)] = np.nan
                    AllVals = pd.concat([AllVals, values], axis=0,
                                        ignore_index=True, sort=True)
                except AttributeError:
                    msg = 'AttributeError when handling'
                    print('{} {}'.format(msg, path.name))
                    lg.logprint(LAM_logger, msg, 'e')
            # Find unit of data
            xlabel = Sett.AddData.get(key)[1]
            kws.update({'id_str': ['Group', 'Channel'], 'ylabel': ylabel,
                        'var_str': xlabel, 'value_str': ylabel,
                        'hue': 'Group', 'row': 'Group'})
            # Go through data columns dropping NaN and plot each
            for col in AllVals.loc[:, ~AllVals.columns.isin(ban)].columns:
                allData = AllVals.loc[:, ['Group', 'Channel',
                                          col]].dropna()
                for plotChan in allData.Channel.unique():
                    name = "{}_{}".format(plotChan, col)
                    plotData = allData.loc[allData.Channel == plotChan, :]
                    plot_maker = plotter(plotData, self._plotDir,
                                         title=name,
                                         palette=self._grpPalette)
                    plot_maker.plot_Data(plotter.distPlot, savepath, **kws)
    
    def __clusters():
        """Handle data for cluster plots."""
        # Find paths to sample-specific data based on found cluster data:
        # Find cluster channels
        clchans = [str(p.stem).split('-')[1] for p in
                   self._dataDir.glob('Clusters-*.csv')]
    
        # Creation of sample-specific position plots:
        if clchans:
            # Find all channels of each sample
            chanpaths = [c for p in Samplegroups._samplePaths for c in
                         p.glob('*.csv')]
            # Find all channel paths relevant to cluster channels
            clpaths = [p for c in clchans for p in chanpaths if
                       p.name == "{}.csv".format(c)]
            # Create directory for cluster plots
            savepath = self._plotDir.joinpath('Clusters')
            savepath.mkdir(exist_ok=True)
            # Create sample-specific position plots:
            for path in clpaths:
                data = system.read_data(path, header=0, test=False)
                if 'ClusterID' in data.columns:
                    name = "{} clusters {}".format(path.parts[-2],
                                                   path.stem)
                    plot_maker = plotter(data, savepath, title=name)
                    plot_maker.clustPlot()
        else:
            msg = 'No cluster count files found (Clusters_*)'
            print('WARNING: {}'.format(msg))
            lg.logprint(LAM_logger, msg, 'w')
    
        # Creation of cluster heatmaps:
        paths = list(self._dataDir.glob('ClNorm_*.csv'))
        if paths:  # Only if cluster data is found
            fullData = pd.DataFrame()
            for path in paths:  # Read all cluster data and concatenate
                channel = str(path.stem).split('_')[1]
                data = system.read_data(path, header=0, test=False).T
                # Alter DF index to contain sample groups
                data.index = data.index.map(lambda x: str(x).split('_')[0])
                groups = data.index.unique()
                for grp in groups:  # for each group:
                    # find means of each bin
                    temp = data.loc[data.index == grp, :]
                    avgs = temp.mean(axis=0, numeric_only=True,
                                     skipna=True)
                    # Add channel identification
                    avgs['Channel'] = channel
                    avgs.rename(grp, inplace=True)
                    # Append the averages and channel to full data
                    fullData = fullData.append(avgs.to_frame().T)
            # The plotting can't handle NaN's, so they are changed to zero.
            fullData = fullData.replace(np.nan, 0)
            # Initialize plotting-class and plot all channel data.
            name = "All Cluster Heatmaps"
            cntr = Samplegroups._center
            # Plotting
            plot_maker = plotter(fullData, self._plotDir, center=cntr,
                                 title=name, palette=None)
            kws = {'height': 3, 'aspect': 5, 'gridspec': {'hspace': 0.5},
                   'row': 'Channel', 'title_y': 1.05, 'sharey': False,
                   'center': plot_maker.MPbin, 'xlen': self._length}
            plot_maker.plot_Data(plotter.Heatmap, savepath.parent, **kws)
        else:  # When no cluster data is found
            msg = 'No normalized cluster count files found (ClNorm_*)'
            print('WARNING: {}'.format(msg))
            lg.logprint(LAM_logger, msg, 'w')

class plotter:
    """For holding data and variables, and consequent plotting."""

    plot_error = False

    def __init__(self, plotData, savepath, center=0, title=None,
                 palette=None, color='b'):
        # Seaborn style settings
        sns.set_style(Sett.seaborn_style)
        sns.set_context(Sett.seaborn_context)
        # Relevant variables for plotting:
        self.data = plotData
        self.title = title
        self.savepath = savepath
        self.palette = palette
        self.color = color
        self.ext = ".{}".format(Sett.saveformat)
        self.format = Sett.saveformat
        # Define center index for plots
        if center != 0:
            self.MPbin = center
        else:
            self.MPbin = 0

    def vector(self, samplename, vectordata, X, Y, binaryArray=None,
               skeleton=None):
        """Plot sample-specific vectors and skeleton plots."""
        # Create skeleton plots if using skeleton vectors
        if skeleton is not None and Sett.SkeletonVector:
            figskel, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6),
                                         sharex=True, sharey=True)
            ax = axes.ravel()
            ax[0].imshow(binaryArray)
            ax[0].axis('off')
            ax[0].set_title('modified', fontsize=16)
            ax[1].imshow(skeleton)
            ax[1].axis('off')
            ax[1].set_title('skeleton', fontsize=16)
            figskel.tight_layout()
            name = str('Skeleton_' + samplename + self.ext)
            figskel.savefig(str(self.savepath.joinpath(name)),
                            format=self.format)
        # Create vector plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = sns.scatterplot(x=X, y=Y, color='xkcd:tan', linewidth=0)
        ax = plt.plot(*vectordata.xy)
        plt.axis('equal')
        name = str('Vector_' + samplename + self.ext)
        fig.savefig(str(self.savepath.parent.joinpath(name)),
                    format=self.format)
        plt.close('all')

    def plot_Data(self, plotfunc, savepath, palette=None, **kws):
        """General plotting function for many kinds of data."""
        
        def __set_xtick():
            """Set plot xticks/labels to be shown every 5 ticks."""
            length = kws.get('xlen')
            xticks = np.arange(0, length, 5)
            plt.setp(g.axes, xticks=xticks, xticklabels=xticks)

        def __centerline():
            """Plot centerline, i.e. the anchoring point of samples."""
            MPbin = kws.get('centerline')
            __, ytop = plt.ylim()
            for ax in g.axes.flat:
                ax.plot((MPbin, MPbin), (0, ytop), 'r--', zorder=0)

        def __stats():
            """Plot statistical elements within data plots."""
            def __marker(value, colors):
                """Designation of number of significance stars."""
                if value <= 0.001:
                    pStr = "*\n*\n*"
                    color = colors[3]
                elif value <= 0.01:
                    pStr = "*\n*"
                    color = colors[2]
                elif value <= Sett.alpha:
                    if value <= 0.05:
                        pStr = "*"
                    else:
                        pStr = ""
                    color = colors[1]
                else:
                    pStr = " "
                    color = colors[0]
                return pStr, color

            stats = kws.pop('Stats')
            __, ytop = plt.ylim()
            tytop = ytop*1.35
            ax = plt.gca()
            ax.set_ylim(top=tytop)
            MPbin = kws.get('centerline')

            # Creation of -log2 P-valueaxis and line plot
            if Sett.negLog2:
                Sett.stars = False
                Y = stats.iloc[:, 7]
                X = Y.index.tolist()
                # Find locations where the log line should be drawn
                ind = Y[Y.notnull()].index
                logvals = pd.Series(np.zeros(Y.shape[0]), index=Y.index)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    logvals.loc[ind] = np.log2(Y[ind].astype(np.float64))
                xmin, xtop = stats.index.min(), stats.index.max()
                # Create twin axis with -log2 P-values
                ax2 = plt.twinx()
                lkws = {'alpha': 0.85}
                ax2.plot(X, np.negative(logvals), color='dimgrey', linewidth=1,
                         **lkws)
                ax2.plot((xmin, xtop), (0, 0), linestyle='dashed',
                         color='grey', linewidth=0.85, **lkws)
                ax2.set_ylabel('P value\n(-log2)')
                # Find top of original y-axis and create a buffer for twin to
                # create a prettier plot
                botAdd = 2.75*-Sett.ylim
                ax2.set_ylim(bottom=botAdd, top=Sett.ylim)
                ytick = np.arange(0, Sett.ylim, 5)
                ax2.set_yticks(ytick)
                ax2.set_yticklabels(ytick, fontdict={'fontsize': 14})
                ax2.yaxis.set_label_coords(1.04, 0.85)
                # Create centerline:
                ybot, ytop = ax.get_ylim()
                yaxis = [ytop, ytop]
                ax.plot((MPbin, MPbin), (ybot, ytop), 'r--', zorder=0)
            # Initiation of variables when not using -log2 & make centerline
            else:
                yaxis = [tytop, tytop]
                yheight = ytop*1.1
                ax.plot((MPbin, MPbin), (0, tytop), 'r--')

            # Create significance stars and color fills
            if 'windowed' in kws:
                comment = "Window: lead {}, trail {}".format(Sett.lead,
                                                             Sett.trail)
                ax.annotate(comment, (0, tytop*1.02), ha='center')
            # Get colors for fills
            LScolors = sns.color_palette('Reds', n_colors=4)
            GRcolors = sns.color_palette('Blues', n_colors=4)
            # Plot significances:
            for index, row in stats.iterrows():
                # If both hypothesis rejections have same value, continue
                if row[3] == row[6]:
                    continue
                xaxis = [index-0.5, index+0.5]
                if row[3] is True:  # ctrl is greater
                    pStr, color = __marker(row[1], LScolors)
                    if Sett.fill:
                        ax.fill_between(xaxis, yaxis, color=color, alpha=0.2,
                                        zorder=0)
                    if Sett.stars:
                        plt.annotate(pStr, (index, yheight), fontsize=14,
                                     ha='center')
                if row[6] is True:  # ctrl is lesser
                    pStr, color = __marker(row[4], GRcolors)
                    if Sett.fill:
                        ax.fill_between(xaxis, yaxis, color=color, alpha=0.2,
                                        zorder=0)
                    if Sett.stars:
                        plt.annotate(pStr, (index, yheight), fontsize=14,
                                     ha='center')

        def __add(centerline=True):
            """Label, tick, and centerline creation/altering."""
            if 'centerline' in kws.keys() and centerline:
                __centerline()
            if 'xlen' in kws.keys():
                __set_xtick()
            if 'ylabel' in kws.keys():
                g.set(ylabel=kws.get('ylabel'))
            if 'xlabel' in kws.keys():
                plt.xlabel(kws.get('xlabel'), labelpad=20)
            return g

        self.plot_error = False
        # The input data is melted if id_str is found in kws:
        if 'id_str' in kws and kws.get('id_str') is not None:
            plotData, varname, valname = __melt_data(**kws)
            kws.update({'xlabel': varname, 'ylabel': valname,
                        'data': plotData})
        else:  # Otherwise data is used as is
            plotData = self.data
            kws.update({'data': plotData})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            if plotfunc.__name__ == 'jointPlot':  # If jointplot:
                # Seaborn unfortunately does not support multi-axis jointplots,
                # consequently these are created as individual files.
                key = plotData.iat[0, 0]
                g = sns.jointplot(data=plotData,
                                  x=plotData.loc[:, kws.get('x')],
                                  y=plotData.loc[:, kws.get('y')], kind='kde',
                                  color=palette.get(key),
                                  joint_kws={'shade_lowest': False})
            elif plotfunc.__name__ == 'catPlot':  # Stat plots
                g = self.catPlot(self.palette, **kws)
                __stats()
                __add(centerline=False)
            elif plotfunc.__name__ == 'pairPlot':  # Pair plot
                g = self.pairPlot(**kws)
                if self.plot_error:  # If error is found in plotting, return
                    print('STOPPING PAIRPLOT')
                    return
            else:  # General handling of plots
                g = sns.FacetGrid(plotData, row=kws.get('row'),
                                  col=kws.get('col'), hue=kws.get('hue'),
                                  sharex=True, sharey=kws.get('sharey'),
                                  gridspec_kws=kws.get('gridspec'),
                                  height=kws.get('height'),
                                  aspect=kws.get('aspect'), legend_out=True,
                                  dropna=False, palette=self.palette)
                g = g.map_dataframe(plotfunc, self.palette, **kws).add_legend()
                if plotfunc.__name__ == 'distPlot':
                    g._legend.remove()
                for ax in g.axes.flat:
                    ax.xaxis.set_tick_params(labelbottom=True)
                __add()
        # Giving a title and then saving the plot
        plt.suptitle(self.title, weight='bold', y=kws.get('title_y'))
        filepath = savepath.joinpath(self.title + self.ext)
        fig = fig = plt.gcf()
        fig.savefig(str(filepath), format=self.format)
        plt.close('all')

    def boxPlot(palette, **kws):
        """Creation of box plots."""
        axes = plt.gca()
        data = kws.pop('data')
        sns.boxplot(data=data, x=kws.get('xlabel'), y=kws.get('ylabel'),
                    hue=kws.get('id_str'), saturation=0.5, linewidth=0.8,
                    showmeans=False, notch=False, palette=palette,
                    fliersize=kws.get('flierS'), ax=axes)
        return axes

    def pairPlot(self, **kws):
        """Creation of pair plots."""
        # Drop bins where no values exists in any channel. Then change missing
        # values to 0 (required for plot)
        data = self.data.sort_values(by="Sample Group").drop(
            'Longitudinal Position', axis=1)
        data = data.dropna(how='all',
                           subset=data.columns[data.columns != 'Sample Group']
                           ).replace(np.nan, 0)
        grpOrder = data["Sample Group"].unique().tolist()  # Plot order
        colors = [self.palette.get(k) for k in grpOrder]
        # Create color variables for scatter edges
        edgeC = []
        for color in colors:
            edgeC.append(tuple([0.7 * v for v in color]))
        # Settings for plotting:
        pkws = {'x_ci': None, 'order': 2, 'truncate': True,
                'scatter_kws': {'linewidth': 0.05, 's': 25, 'alpha': 0.4,
                                'edgecolors': edgeC},
                'line_kws': {'alpha': 0.7, 'linewidth': 1.5}}
        if Sett.plot_jitter:
            pkws.update({'x_jitter': 0.49, 'y_jitter': 0.49})
        dkws = {'linewidth': 2}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            try:
                g = sns.pairplot(data=data, hue=kws.get('hue'),
                                 kind=kws.get('kind'), diag_kind=kws.get(
                                     'diag_kind'), palette=self.palette,
                                 plot_kws=pkws, diag_kws=dkws)
            # In case of missing or erroneous data, linalgerror can be raised
            except np.linalg.LinAlgError:  # Then, exit plotting
                msg = '-> Confirm that all samples have proper channel data'
                fullmsg = 'Pairplot singular matrix\n{}'.format(msg)
                lg.logprint(LAM_logger, fullmsg, 'ex')
                print('ERROR: Pairplot singular matrix')
                print(msg)
                self.plot_error = True
                return None
        # Enhance legends
        for lh in g._legend.legendHandles:
            lh.set_alpha(1)
            lh._sizes = [30]
        # Set bottom values to zero, as no negatives in count data
        for ax in g.axes.flat:
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0)
        return g

    def distPlot(palette, **kws):
        """Creation of distributions."""
        axes = plt.gca()
        data = kws.pop('data')
        values = kws.get('value_str')
        try:
            color = palette[data[kws.get('hue')].iloc[0]]
            sns.distplot(a=data[values], hist=True, rug=True, norm_hist=True,
                         color=color, axlabel=kws.get('xlabel'), ax=axes)
        # In case of missing or erroneous data, linalgerror can be raised
        except np.linalg.LinAlgError:
            msg = '-> Confirm that all samples have proper channel data'
            fullmsg = 'Distribution plot singular matrix\n{}'.format(msg)
            lg.logprint(LAM_logger, fullmsg, 'ex')
            print('ERROR: Distribution plot singular matrix')
            print(msg)
            axes.text(x=0.1, y=0.1, s="ERROR")
        return axes

    def linePlot(palette, **kws):
        """Creation of line plots of additional data."""
        axes = plt.gca()
        data = kws.pop('data')
        err_kws = {'alpha': 0.4}
        sns.lineplot(data=data, x=kws.get('xlabel'), y=kws.get('ylabel'),
                     hue=kws.get('hue'), alpha=0.5, dashes=False,
                     err_style='band', ci='sd', palette=palette, ax=axes,
                     err_kws=err_kws)
        return axes

    def jointPlot(palette, **kws):
        """Creation of bivariable joint plots with density and distribution."""
        sns.set(style="white")
        axes = plt.gca()
        data = kws.pop('data')
        key = data.iat[(0, 0)]
        sns.jointplot(data=data, x=data.loc[:, kws.get('X')],
                      y=data.loc[:, kws.get('Y')], kind='kde',
                      color=palette[key], ax=axes, space=0,
                      joint_kws={'shade_lowest': False})

    def catPlot(self, palette, fliers=True, **kws):
        """Creation of statistical versus plots."""
        data = kws.pop('data')
        fkws = {'dropna': False}
        xlabel, ylabel = kws.get('xlabel'), kws.get('ylabel')
        data = data.replace(np.nan, 0)
        flierprops = kws.pop('fliersize')
        fliers = not Sett.observations
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            g = sns.catplot(data=data, x=xlabel, y=ylabel, hue="Sample Group",
                            kind="box", palette=palette, linewidth=0.15,
                            height=kws.get('height'), aspect=kws.get('aspect'),
                            facet_kws=fkws, showfliers=fliers, legend_out=True,
                            **flierprops)
            if Sett.observations:  # Create scatters of individual observations
                g = sns.swarmplot(data=data, x=xlabel, y=ylabel, zorder=1,
                                  hue="Sample Group", size=2.5, linewidth=0.05,
                                  palette=palette)
                g.get_legend().set_visible(False)
        return g

    def Heatmap(palette, **kws):
        """Creation of heat maps."""
        axes = plt.gca()
        data = kws.pop('data')
        sns.heatmap(data=data.iloc[:, :-2], cmap='coolwarm', robust=True,
                    ax=axes)
        plt.yticks(rotation=45)
        MPbin = kws.get('center')
        axes.plot((MPbin, MPbin), (0, data.shape[0]), 'r--')
        return axes

    def total_plot(self, stats, order):
        """Creation of statistical plots of variable totals."""
        def __marker(value):
            if value <= 0.001:
                pStr = "***"
            elif value <= 0.01:
                pStr = "**"
            elif value <= Sett.alpha:
                if value <= 0.05:
                    pStr = "*"
                else:
                    pStr = ""
            else:
                pStr = ""
            return pStr

        # Melt data to long form and drop missing observation points
        plotData = pd.melt(self.data, id_vars=['Sample Group', 'Variable'],
                           value_name='Value')
        plotData = plotData.dropna(subset=['Value'])
        # Make sure that data is in float format
        plotData['Value'] = plotData['Value'].astype('float64')
        # Assign variable indication the order of plotting
        plotData['Ord'] = plotData.loc[:, 'Sample Group'].apply(order.index)
        plotData.sort_values(by=['Ord', 'Variable'], axis=0, inplace=True)
        g = sns.catplot('Sample Group', 'Value', data=plotData,
                        col='Variable', palette=self.palette, kind='violin',
                        sharey=False, saturation=0.5)
        # Find group order number for control group for plotting significances
        stats.sort_index(inplace=True)
        ctrl_x = order.index(Sett.cntrlGroup)
        # Loop through the plot axes
        for axInd, ax in enumerate(g.axes.flat):
            # Find rejected H0 for current axis
            statRow = stats.iloc[axInd, :]
            rejects = statRow.iloc[statRow.index.get_level_values(1).str
                                   .contains('Reject')
                                   ].where(statRow).dropna()
            rejectN = np.count_nonzero(rejects.to_numpy())
            ax.set_ylim(bottom=0)
            if rejectN > 0:  # If any rejected H0
                # Raise y-limit of axis to fit significance plots
                __, ytop = ax.get_ylim()
                tytop = ytop*1.3
                ax.set_ylim(top=tytop)
                # Find heights for significance lines
                heights = np.linspace(ytop, ytop*1.2, rejectN)
                # Loop groups with rejected H0
                for i, grp in enumerate(rejects.index.get_level_values(0)):
                    y = heights[i]  # Get height for the group's line
                    grp_x = order.index(grp)  # Get x-axis location of group
                    line = sorted([grp_x, ctrl_x])
                    # Plot line
                    ax.hlines(y=y, xmin=line[0], xmax=line[1], color='dimgrey')
                    # Locate P-value and get significance stars
                    Pvalue = statRow.loc[(grp, 'P Two-sided')]
                    pStr = __marker(Pvalue)
                    # Define plot location for stars and plot
                    ax.annotate(pStr, (line[0]+.5, y), ha='center')
        plt.suptitle(self.title, weight='bold', y=1.02)
        filepath = self.savepath.joinpath(self.title + self.ext)
        g.savefig(str(filepath), format=self.format)
        plt.close('all')

    def clustPlot(self):
        """Creation of sample-specific cluster position plots."""
        # Drop all features without designated cluster
        fData = self.data.dropna(subset=["ClusterID"])
        # Select data to be plotted
        plotData = fData.loc[:, ["Position X", "Position Y", "ClusterID"]]
        # Create unique color for each cluster
        IDs = pd.unique(plotData.loc[:, "ClusterID"])
        colors = sns.color_palette("hls", len(IDs))
        shuffle(colors)
        palette = {}
        for ind, ID in enumerate(IDs):
            palette.update({ID: colors[ind]})
        # Get non-clustered cells for background plotting
        baseData = self.data[self.data["ClusterID"].isnull()]
        # Initialization of figure
        figure, ax = plt.subplots(figsize=(13, 4.75))
        kws = dict(linewidth=0.1)
        # Plot background
        ax.scatter(baseData.loc[:, "Position X"],
                   baseData.loc[:, "Position Y"], s=1.5, c='xkcd:tan')
        # Plot clusters
        sns.scatterplot(data=plotData, x="Position X", y="Position Y",
                        hue="ClusterID", palette=palette, ax=ax, s=5,
                        legend=False, **kws)
        plt.title(self.title)
        plt.axis('equal')
        # Save figure
        filepath = self.savepath.joinpath(self.title+self.ext)
        figure.savefig(str(filepath), format=self.format)
        plt.close()
