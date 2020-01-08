# -*- coding: utf-8 -*-

"""
Created on Wed Mar  6 12:42:28 2019
@author: Arto I. Viitanen

"""

# LAM modules
from settings import settings as Sett
import logger as lg
# Standard libraries
import warnings
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
            ax[0].imshow(binaryArray, cmap=(plt.cm.gray))
            ax[0].axis('off')
            ax[0].set_title('modified', fontsize=16)
            ax[1].imshow(skeleton, cmap=(plt.cm.gray))
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
        def __melt_data(Data, **kws):
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
                ax.plot((MPbin, MPbin), (0, ytop), 'r--')

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
                    if value <= 0.5:
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
#                Y.replace(0, np.nan, inplace=True)
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
                ybot2, ytop2 = ax2.get_ylim()
                yaxis = [ybot2, ybot2]
                # Create centerline
                ax2.plot((MPbin, MPbin), (ybot2, ytop2), 'r--')
            # Initiation of variables when not using -log2 & make centerline
            else:
                yaxis = [tytop, tytop]
                yheight = ytop*1.1
                ax.plot((MPbin, MPbin), (0, tytop), 'r--')

            # Create significance stars and color fills
            if 'windowed' in kws:
                comment = "Window: lead {}, trail {}".format(Sett.lead,
                                                             Sett.trail)
                ax.text(0, tytop*1.02, comment)
            LScolors = sns.color_palette('Reds', n_colors=4)
            GRcolors = sns.color_palette('Blues', n_colors=4)
            for index, row in stats.iterrows():
                # If both hypothesis rejections have same value, continue
                if row[3] == row[6]:
                    continue
                xaxis = [index-0.5, index+0.5]
                if row[3] is True:  # cntrl is greater
                    pStr, color = __marker(row[1], LScolors)
                    if Sett.fill:
                        plt.fill_between(xaxis, yaxis, color=color, alpha=0.2)
                    if Sett.stars:
                        plt.text(index, yheight, pStr,
                                 fontdict={'fontsize': 14})
                if row[6] is True:  # cntrl is lesser
                    pStr, color = __marker(row[4], GRcolors)
                    if Sett.fill:
                        plt.fill_between(xaxis, yaxis, color=color, alpha=0.2)
                    if Sett.stars:
                        plt.text(index, yheight, pStr,
                                 fontdict={'fontsize': 14})

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
            plotData, varname, valname = __melt_data(self.data, **kws)
            kws.update({'xlabel': varname,  'ylabel': valname,
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
                    hue=kws.get('id_str'), saturation=0.5, linewidth=0.2,
                    showmeans=False, notch=False, palette=palette,
                    fliersize=kws.get('flierS'), ax=axes)
        return axes

    def pairPlot(self, **kws):
        """Creation of pair plots."""
        # Missing values changed to 0 (required for plot)
        data = self.data.sort_values(by="Sample Group").drop(
                            'Longitudinal Position', axis=1).replace(np.nan, 0)
        grpOrder = data["Sample Group"].unique().tolist()  # Plot order
        colors = [self.palette.get(k) for k in grpOrder]
        # Create color variables for scatter edges
        edgeC = []
        for color in colors:
            edgeC.append(tuple([0.7 * v for v in color]))
        # Settings for plotting:
        pkws = {'x_ci': None, 'order': 4, 'truncate': True, 'x_jitter': 0.49,
                'y_jitter': 0.49,
                'scatter_kws': {'linewidth': 0.05, 's': 25, 'alpha': 0.4,
                                'edgecolors': edgeC},
                'line_kws': {'alpha': 0.7, 'linewidth': 1.5}}
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
            color = palette.get(data[kws.get('hue')].iloc[0])
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
                      color=palette.get(key), ax=axes, space=0,
                      joint_kws={'shade_lowest': False})

    def catPlot(self, palette, fliers=True, **kws):
        """Creation of statistical versus plots."""
        data = kws.pop('data')
        fkws = {'dropna': False}
        xlabel, ylabel = kws.get('xlabel'), kws.get('ylabel')
        data = data.replace(np.nan, 0)
        flierprops = kws.pop('fliersize')
        if Sett.observations:
            fliers = False
            flierprops = {}
        g = sns.catplot(data=data, x=xlabel, y=ylabel, hue="Sample Group",
                        kind="box", palette=palette, linewidth=0.15,
                        height=kws.get('height'), aspect=kws.get('aspect'),
                        facet_kws=fkws, showfliers=fliers, legend_out=True,
                        **flierprops)
        if Sett.observations:  # Create scatters of individual observations
            g = sns.swarmplot(data=data, x=xlabel, y=ylabel,
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
        ybot, ytop = plt.ylim()
        axes.plot((MPbin, MPbin), (0, data.shape[0]), 'r--')
        return axes

    def total_plot(self, stats, order):
        """Creation of statistical plots of variable totals."""
        def __marker(value):
            if value <= 0.001:
                pStr = "***"
                offset = 0.38
            elif value <= 0.01:
                pStr = "**"
                offset = 0.42
            elif value <= Sett.alpha:
                if value <= 0.5:
                    pStr = "*"
                else:
                    pStr = ""
                offset = 0.47
            else:
                pStr = ""
                offset = 0
            return pStr, offset

        # Melt data to long form and drop missing observation points
        plotData = pd.melt(self.data, id_vars=['Sample Group', 'Variable'],
                           value_name='Value')
        plotData = plotData.dropna(subset=['Value'])
        # Make sure that data is in float format
        plotData['Value'] = plotData['Value'].astype('float64')
        # Assign variable indication the order of plotting
        plotData['Ord'] = plotData.loc[:, 'Sample Group'].apply(lambda x:
                                                                order.index(x))
        plotData.sort_values(by=['Ord', 'Variable'], axis=0, inplace=True)
        g = sns.catplot('Sample Group', 'Value', data=plotData,
                        col='Variable', palette=self.palette, kind='violin',
                        sharey=False, saturation=0.5)
        # Find group order number for control group for plotting significances
        stats.sort_index(inplace=True)
        Cntrl_x = order.index(Sett.cntrlGroup)
        # Loop through the plot axes
        for axInd, ax in enumerate(g.axes.flat):
            # Find rejected H0 for current axis
            statRow = stats.iloc[axInd, :]
            rejects = statRow.iloc[statRow.index.get_level_values(1).str
                                   .contains('Reject')
                                   ].where(statRow == True).dropna()
            rejectN = np.count_nonzero(rejects.to_numpy())
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
                    line = sorted([grp_x, Cntrl_x])
                    # Plot line
                    ax.hlines(y=y, xmin=line[0], xmax=line[1], color='dimgrey')
                    # Locate P-value and get significance stars
                    Pvalue = statRow.loc[(grp, 'P Two-sided')]
                    pStr, offset = __marker(Pvalue)
                    # Define plot location for stars and plot
                    x = xmin + offset
                    ax.text(x, y, pStr)
        for ax in g.axes.flat:
            ax.set_ylim(bottom=0)
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
        palette = {}
        for ind, ID in enumerate(IDs):
            palette.update({ID: colors[ind]})
        # Get non-clustered cells for background plotting
        baseData = self.data[self.data["ClusterID"].isnull()]
        # Initialization of figure
        figure, ax = plt.subplots(figsize=(13, 4.75))
        kws = dict(linewidth=0)
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
