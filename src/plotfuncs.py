# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:34:33 2020

@author: Arto
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

def channel_matrix(plot, **kws):
    """Creation of pair plots."""
    # Settings for plotting:
    pkws = {'x_ci': None, 'truncate': True, 'order': 2,
            'scatter_kws': {'linewidth': 0.1, 's': 15, 'alpha': 0.4},
            'line_kws': {'alpha': 0.6, 'linewidth': 1}}
    if Sett.plot_jitter:
        pkws.update({'x_jitter': 0.49, 'y_jitter': 0.49})

    # Drop unneeded data and replace NaN with zero (required for plot)
    data = plot.data.drop('Linear Position', axis=1)
    cols = data.columns != 'Sample Group'
    data = data.dropna(how='all', subset=data.columns[cols]).replace(np.nan, 0)
    try:
        g = sns.pairplot(data=data, hue=kws.get('hue'),
                         height=1.5, aspect=1,
                         kind=kws.get('kind'),
                         diag_kind=kws.get('diag_kind'),
                         palette=plot.handle.palette,
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
        plot.plot_error = True
        return None
    return g

def clusters():
    pass

def heatmap(plot, **kws):
    """Creation of heat maps."""
    data = plot.data.replace(np.nan, 0)
    rows = data.loc[:, kws.get('row')].unique()
    for ind, ax in enumerate(plot.g.axes.flat):
        sub_data = data.loc[data[kws.get('row')] == rows[ind],
                            data.columns != kws.get('row')]
        sns.heatmap(data=sub_data, cmap='coolwarm', robust=True,
                    linewidth=0.05, linecolor='dimgrey', ax=ax)
        ax.set_title(rows[ind])
    plt.subplots_adjust(left=0.25, right=0.99)
    return plot.g

def bivariate_kde(plot, **kws):
    b_kws = {'sharex': 'False', 'sharey': 'False'}
    b_kws.update(kws)
    plot_data = plot.data.drop('Channel', axis=1)
    full_data = plot_data.merge(plot.sec_data, on=['Sample Group', 'Sample',
                                                   'Linear Position'],
                                how='outer')
    
    # id_cols = kws.get('col_ban')
    # data_cols = plot.data.columns.difference(id_cols)
    # sec_cols = plot.sec_data.columns.difference(id_cols)
    
    # shared_cols = plot.data.columns.intersection(plot.sec_data.columns)
    # melt_cols = shared_cols.union(id_cols)
    # plot_data = plot.data.merge(plot.sec_data, how='outer', copy=True,
    #                             on=shared_cols.to_list())
    # plot_data = plot_data.melt(id_vars=melt_cols)
    
    g = sns.FacetGrid(data=full_data, row='Channel', col='Type',
                      hue="Sample Group", sharex=False, sharey=False)
    g = g.map(sns.kdeplot, 'None_y', 'None_x')
    
    print('done')
    # for ind, col in sec_cols:
    #     for ind2, col2 in data_cols:
    #         plot_data = data.loc[col2]
    #         plot_sec_data = sec_data.loc[:, col]
    #         sns.kdeplot()
    # g = sns.FacetGrid(data, hue="Sample Group")
    # g = g.map_diag(plt.hist)
    # g = g.map_offdiag(plt.scatter)
    return

def lines(plot, **kws):
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

def totals():
    pass

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

def widths():
    pass
