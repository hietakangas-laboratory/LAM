# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:34:33 2020

Individual plot functions for LAM.

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
import pandas as pd

try:
    LAM_logger = lg.get_logger(__name__)
except AttributeError:
    print('Cannot get logger')


def bivariate_kde(plotter, **in_kws):
    """Plot bivariate density estimations."""
    kws = in_kws.get('plot_kws')
    data = plotter.data.drop('Channel', axis=1)
    if plotter.sec_data is not None:
        plot_data = data.merge(plotter.sec_data, how='outer',
                               on=['Sample Group', 'Sample',
                                   'Linear Position'])
    else:
        plot_data = data
    g = sns.FacetGrid(data=plot_data, row=kws.get('row'), col=kws.get('col'),
                      hue="Sample Group", sharex=False, sharey=False,
                      height=5)
    with warnings.catch_warnings():  # If
        warnings.simplefilter('ignore', category=UserWarning)
        g = g.map(sns.kdeplot, 'Value_y', 'Value_x', shade_lowest=False,
                  shade=False, linewidths=2.5, alpha=0.6)
    return g


def channel_matrix(plotter, **kws):
    """Creation of pair plots."""
    # Settings for plotting:
    pkws = {'x_ci': None, 'truncate': True, 'order': 2,
            'scatter_kws': {'linewidth': 0.1, 's': 15, 'alpha': 0.4},
            'line_kws': {'alpha': 0.45, 'linewidth': 1}}
    if Sett.plot_jitter:
        pkws.update({'x_jitter': 0.49, 'y_jitter': 0.49})

    # Drop unneeded data and replace NaN with zero (required for plot)
    data = plotter.data.drop('Linear Position', axis=1)
    cols = data.columns != 'Sample Group'
    data = data.dropna(how='all', subset=data.columns[cols]).replace(np.nan, 0)
    try:
        g = sns.pairplot(data=data, hue=kws.get('hue'),
                         height=1.5, aspect=1,
                         kind=kws.get('kind'),
                         diag_kind=kws.get('diag_kind'),
                         palette=plotter.handle.palette,
                         plot_kws=pkws,
                         diag_kws={'linewidth': 1.25})
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
        plotter.plot_error = True
        return None
    except RuntimeError:
        msg = '-> Confirm that all samples have proper channel data'
        fullmsg = 'Pairplot RuntimeError\n{}'.format(msg)
        lg.logprint(LAM_logger, fullmsg, 'ex')
        print('ERROR: Pairplot RuntimeError')
        print(msg)
        plotter.plot_error = True
        return None
    return g


def cluster_positions(plotter, **kws):
    """Creation of sample-specific cluster position plots."""
    p_kws = dict(linewidth=0.1, edgecolor='dimgrey')

    # Create unique color for each cluster
    IDs = pd.unique(plotter.data.ClusterID)
    colors = sns.color_palette("hls", len(IDs))
    shuffle(colors)
    palette = {}
    for ind, ID in enumerate(IDs):
        palette.update({ID: colors[ind]})
    # Get non-clustered cells for background plotting
    b_data = kws.get('b_data')
    chans = plotter.data.Channel.unique()
    for ind, ax in enumerate(plotter.g.axes.flat):  # Plot background
        ax.axis('equal')
        ax.scatter(b_data.loc[:, "Position X"], b_data.loc[:, "Position Y"],
                   s=10, c='xkcd:tan')
        # Plot clusters
        plot_data = plotter.data.loc[plotter.data.Channel == chans[ind], :]
        sns.scatterplot(data=plot_data, x="Position X", y="Position Y",
                        hue="ClusterID", palette=palette, s=20, legend=False,
                        ax=ax, **p_kws)
        ax.set_title("{} Clusters".format(chans[ind]))
    return plotter.g


def distribution(plotter, **kws):
    """Plot distributions."""
    try:
        g = (plotter.g.map(sns.distplot, 'Value', kde=True, hist=True,
                           norm_hist=True, hist_kws={"alpha": 0.3,
                                                     "linewidth": 1}))
    except RuntimeError:
        g = (plotter.g.map(sns.distplot, 'Value', kde=True, hist=True,
                           norm_hist=True, hist_kws={"alpha": 0.3,
                                                     "linewidth": 1},
                           kde_kws={'bw': 20}))
    except np.linalg.LinAlgError:
        msg = '-> Confirm that all samples have proper channel data'
        fullmsg = 'Distribution plot singular matrix\n{}'.format(msg)
        lg.logprint(LAM_logger, fullmsg, 'ex')
        print('ERROR: Distribution plot singular matrix')
        print(msg)
        return None
    for ax in g.axes.flat:
        ax.set_xlim(left=0)
    return g


def heatmap(plotter, **kws):
    """Creation of heat maps."""
    data = plotter.data.replace(np.nan, 0)
    rows = data.loc[:, kws.get('row')].unique()
    for ind, ax in enumerate(plotter.g.axes.flat):
        sub_data = data.loc[data[kws.get('row')] == rows[ind],
                            data.columns != kws.get('row')]
        sns.heatmap(data=sub_data, cmap='coolwarm', robust=True,
                    linewidth=0.05, linecolor='dimgrey', ax=ax)
        ax.set_title(rows[ind])
        if kws.get('Sample_plot'):
            ylabels = ax.get_yticklabels()
            ax.set_yticklabels(ylabels, rotation=35, fontsize=8)
            strings = [x[0] for x in sub_data.index.str.split('_')]
            inds = [strings.index(i) for i in sorted(set(strings))[1:]]
            left, right = ax.get_xlim()
            for idx in inds:
                ax.hlines(idx, xmin=left, xmax=right, linestyles='dotted',
                          color=plotter.handle.palette.get(strings[idx]),
                          linewidth=1.5)
    plt.subplots_adjust(left=0.25, right=0.99)
    return plotter.g


def lines(plotter, **kws):
    """Plot lines."""
    err_dict = {'alpha': 0.3}
    data = plotter.data
    # data = plotter.data.dropna()
    melt_kws = kws.get('melt')
    g = (plotter.g.map_dataframe(sns.lineplot, data=data,
                                 x=data.loc[:, melt_kws.get('var_name')
                                            ].astype(float),
                                 y=data.loc[:, melt_kws.get('value_name')],
                                 ci='sd', err_style='band',
                                 hue=kws.get('hue'), dashes=False, alpha=1,
                                 palette=plotter.handle.palette,
                                 err_kws=err_dict))
    return g


def vector_plots(savepath, samplename, vectordata, X, Y, binaryArray=None,
                 skeleton=None):
    """Plot sample-specific vectors and skeleton plots."""
    ext = ".{}".format(Sett.saveformat)
    # Get vector creation settings and create string
    if Sett.SkeletonVector:
        sett_dict = {'Type': 'Skeleton', 'Simplif.': Sett.simplifyTol,
                     'Resize': Sett.SkeletonResize, 'Distance': Sett.find_dist,
                     'Dilation': Sett.BDiter, 'Smooth': Sett.SigmaGauss}
    else:
        sett_dict = {'Type': 'Median', 'Simplif.': Sett.simplifyTol,
                     'Bins': Sett.medianBins}
    sett_string = '  |  '.join(["{} = {}".format(k, v) for k, v in
                                sett_dict.items()])

    # Create skeleton plots if using skeleton vectors
    if skeleton is not None and Sett.SkeletonVector:
        figskel, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6),
                                     sharex=True, sharey=True)
        ax = axes.ravel()
        # Plot of binary array
        ax[0].imshow(binaryArray)
        ax[0].axis('off')
        ax[0].set_title('modified', fontsize=14)
        # Plot of skeletonized binary array
        ax[1].imshow(skeleton)
        ax[1].axis('off')
        ax[1].set_title('skeleton', fontsize=14)
        figskel.tight_layout()
        # Add settings string to plot
        plt.annotate(sett_string, (5, 5), xycoords='figure points')
        # Save
        name = str('Skeleton_' + samplename + ext)
        figskel.savefig(str(savepath.joinpath(name)), format=Sett.saveformat)
        plt.close()
    # Create vector plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.scatterplot(x=X, y=Y, color='xkcd:tan', linewidth=0)
    ax.plot(*vectordata.xy)
    plt.axis('equal')
    plt.title("Vector " + samplename)
    # Add settings string to plot
    plt.annotate(sett_string, (5, 5), xycoords='figure points')
    # Save plot
    name = str('Vector_' + samplename + ext)
    fig.savefig(str(savepath.parent.joinpath(name)), format=Sett.saveformat)
    plt.close()


def violin(plotter, **kws):
    """Plot violins."""
    plotter.g = sns.catplot(x='Sample Group', y='Value',
                            data=plotter.data, row=kws.get('row'),
                            col=kws.get('col'),
                            height=kws.get('height'), aspect=kws.get('aspect'),
                            palette=plotter.handle.palette, kind='violin',
                            sharey=False, saturation=0.5)
    return plotter.g
