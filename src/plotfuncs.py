# -*- coding: utf-8 -*-
"""
Individual plot functions for LAM.

Created on Mon Mar 16 16:34:33 2020
@author: Arto I. Viitanen
"""

# Standard libraries
import warnings
from random import shuffle
# Packages
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# LAM modules
from src.settings import Settings as Sett
import src.logger as lg

LAM_logger = None


def bivariate_kde(plotter, **in_kws):
    """Plot bivariate density estimations."""
    kws = in_kws.get('plot_kws')
    data = plotter.data.drop('Channel', axis=1)
    if plotter.sec_data is not None:
        data = data.merge(plotter.sec_data, how='outer', on=['Sample Group', 'Sample', 'Linear Position'])
    # Create plot grid
    g = sns.FacetGrid(data=data, row=kws.get('row'), col=kws.get('col'), hue="Sample Group", sharex=False,
                      sharey=False, height=5, aspect=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        try:
            # Create plots
            g = g.map(sns.kdeplot, 'Value_y', 'Value_x', shade_lowest=False, shade=False, linewidths=1.75, alpha=0.6)
        except np.linalg.LinAlgError:
            msg = '-> Confirm that all samples have proper channel data'
            fullmsg = f'Bivariate plot singular matrix\n{msg}'
            lg.logprint(LAM_logger, fullmsg, 'ex')
            print('ERROR: Bivariate plot singular matrix')
            print(msg)
            return g
    return g


def channel_matrix(plotter, **kws):
    """Creation of pair plots."""
    # Settings for plotting:
    pkws = {'x_ci': None, 'truncate': True, 'order': 2, 'scatter_kws': {'linewidth': 0.1, 's': 15, 'alpha': 0.4},
            'line_kws': {'alpha': 0.45, 'linewidth': 1}}
    if Sett.plot_jitter:
        pkws.update({'x_jitter': 0.49, 'y_jitter': 0.49})

    # Drop unneeded data and replace NaN with zero (required for plot)
    data = plotter.data.drop('Linear Position', axis=1)
    cols = data.columns != 'Sample Group'
    data = data.dropna(how='all', subset=data.columns[cols]).replace(np.nan, 0)
    try:
        g = sns.pairplot(data=data, hue=kws.get('hue'), height=1.5, aspect=1, kind=kws.get('kind'), plot_kws=pkws,
                         diag_kind=kws.get('diag_kind'), palette=plotter.handle.palette, diag_kws={'linewidth': 1.25})
        # Set bottom values to zero, as no negatives in count data
        for ax in g.axes.flat:
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0)
    # In case of missing or erroneous data, linalgerror can be raised
    except np.linalg.LinAlgError:  # Then, exit plotting
        msg = '-> Confirm that all channels have proper data'
        fullmsg = 'Pairplot singular matrix\n{}'.format(msg)
        lg.logprint(LAM_logger, fullmsg, 'ex')
        print('ERROR: Pairplot singular matrix')
        print(msg)
        plotter.plot_error = True
        return None
    except RuntimeError:
        msg = '-> Confirm that all channels have proper data'
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
    identifiers = pd.unique(plotter.data.ClusterID)
    colors = sns.color_palette("hls", len(identifiers))
    shuffle(colors)
    palette = {}
    for ind, ID in enumerate(identifiers):
        palette.update({ID: colors[ind]})
    # Get non-clustered cells for background plotting
    b_data = kws.get('b_data')
    chans = plotter.data.Channel.unique()
    for ind, ax in enumerate(plotter.g.axes.flat):  # Plot background
        ax.axis('equal')
        ax.scatter(b_data.loc[:, "Position X"], b_data.loc[:, "Position Y"], s=10, c='xkcd:tan')
        # Plot clusters
        plot_data = plotter.data.loc[plotter.data.Channel == chans[ind], :]
        sns.scatterplot(data=plot_data, x="Position X", y="Position Y", hue="ClusterID", palette=palette, s=20,
                        legend=False, ax=ax, **p_kws)
        ax.set_title("{} Clusters".format(chans[ind]))
    return plotter.g


def distribution(plotter, **kws):
    """Plot distributions."""
    data = plotter.data
    data = data.reset_index()
    row_order = data.loc[:, kws.get('row')].unique()
    discrete = False
    if kws.get('row') == 'Channel':
        discrete = True
    try:
        g = sns.displot(data=data, x='Value', hue=kws.get('hue'), col=kws.get('col'), row=kws.get('row'), alpha=0.4,
                        kind='hist', stat='probability', palette=plotter.handle.palette, kde=True, height=2.5, aspect=2.25,
                        row_order=row_order, discrete=discrete, multiple='stack',
                        facet_kws={'legend_out': True, 'sharex': False, 'sharey': False,
                                   'gridspec_kws': kws.get('gridspec')})
    except np.linalg.LinAlgError:
        msg = '-> Confirm that all samples have proper channel data'
        fullmsg = 'Distribution plot singular matrix\n{}'.format(msg)
        lg.logprint(LAM_logger, fullmsg, 'ex')
        print('ERROR: Distribution plot singular matrix')
        print(msg)
        return None
    for ind, ax in enumerate(g.axes.flat):
        xmax = data.loc[(data.loc[:, kws.get('row')] == row_order[ind]), 'Value'].max()
        ax.set_xlim(left=0, right=xmax*1.2)
        ax.set_xlabel('')
    return g


def heatmap(plotter, **kws):
    """Creation of heat maps."""
    data = plotter.data.replace(np.nan, 0)
    rows = data.loc[:, kws.get('row')].unique()
    for ind, ax in enumerate(plotter.g.axes.flat):
        sub_data = data.loc[data[kws.get('row')] == rows[ind], data.columns != kws.get('row')].sort_index()
        sns.heatmap(data=sub_data, cmap='coolwarm', robust=True, linewidth=0.05, linecolor='dimgrey', ax=ax)
        ax.set_title(rows[ind])
        ylabels = ax.get_yticklabels()
        ax.set_yticklabels(ylabels, rotation=35)
        if kws.get('Sample_plot'):
            ax.set_yticklabels(ylabels, fontsize=8)
            strings = [x[0] for x in sub_data.index.str.split('_')]
            inds = [strings.index(i) for i in sorted(set(strings))[1:]]
            left, right = ax.get_xlim()
            for idx in inds:
                ax.hlines(idx, xmin=left, xmax=right, linestyles='dotted',
                          color=plotter.handle.palette.get(strings[idx]), linewidth=1.5)
    plt.subplots_adjust(left=0.25, right=0.99)
    return plotter.g


def lines(plotter, **kws):
    """Plot lines."""
    err_dict = {'alpha': 0.3}
    data = plotter.data
    # data = plotter.data.dropna()
    melt_kws = kws.get('melt')
    g = (plotter.g.map_dataframe(sns.lineplot, data=data, x=melt_kws.get('var_name'), y=melt_kws.get('value_name'),
                                 errorbar='sd', err_style='band', hue=kws.get('hue'), dashes=False, alpha=1,
                                 palette=plotter.handle.palette, err_kws=err_dict))
    # for ax in g.axes.flat:
    #     ax.set_ylim(100, 300)
    return g


def skeleton_plot(savepath, samplename, binary_array, skeleton):
    sett_dict = {'Type': 'Skeleton', 'Simplif.': Sett.simplifyTol, 'Resize': Sett.SkeletonResize,
                 'Distance': Sett.find_dist, 'Dilation': Sett.BDiter, 'Smooth': Sett.SigmaGauss}
    sett_string = '  |  '.join(["{} = {}".format(k, v) for k, v in sett_dict.items()])
    figskel, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), sharex='row', sharey='row')
    ax = axes.ravel()
    # Plot of binary array
    ax[0].imshow(binary_array)
    ax[0].axis('off')
    ax[0].set_title('modified', fontsize=14)
    ax[0].invert_yaxis()
    # Plot of skeletonized binary array
    ax[1].imshow(skeleton)
    ax[1].axis('off')
    ax[1].set_title('skeleton', fontsize=14)
    ax[1].invert_yaxis()
    figskel.tight_layout()
    # Add settings string to plot
    plt.annotate(sett_string, (5, 5), xycoords='figure points')
    # Save
    name = str('Skeleton_' + samplename + f'.{Sett.saveformat}')
    figskel.savefig(str(savepath.joinpath(name)), format=Sett.saveformat)
    plt.close()


def create_vector_plots(workdir, savedir, sample_dirs, settings=None, non_valid: (list, bool)=False):
    def _get_ch_data(sample_name, dir_glob, ch_name):
        try:
            dir_path = [p for p in dir_glob if (str(p).startswith(f'{ch_name}_') or f'_{ch_name}_' in str(p))
                        and p.is_dir()][0]
            data_path = dir_path.glob('*Position.csv')
            data = pd.read_csv(next(data_path), header=Sett.header_row)
            return data.loc[:, ['Position X', 'Position Y']].assign(sample=sample_name)
        except (IndexError, StopIteration, FileNotFoundError) as err:
            print(f'WARNING: Cannot read {ch_name}-file for {sample_name}')
            print(f'  --> error message: {err}')
            return pd.DataFrame({'Position X': [np.nan], 'Position Y': [np.nan], 'sample': [sample_name]})

    full = pd.DataFrame()
    vectors = pd.DataFrame()
    mp_full = pd.DataFrame()
    for path in sample_dirs:
        sample = path.name
        data = _get_ch_data(sample, workdir.joinpath(sample).glob(f'*{Sett.vectChannel}_*'), Sett.vectChannel)
        try:
            vector = pd.read_csv(next(path.glob('Vector.*')))
            vector = vector.assign(sample=sample)
        except (FileNotFoundError, StopIteration):
            print(f'WARNING: Cannot find vector-file for {sample}')
            vector = pd.DataFrame({'X': [np.nan], 'Y': [np.nan], 'sample': [sample]})
        full = pd.concat([full, data])
        vectors = pd.concat([vectors, vector])

        # read MP data
        if not Sett.useMP:
            continue
        mp_data = _get_ch_data(sample, workdir.joinpath(sample).glob(f'*{Sett.MPname}_*'), Sett.MPname)
        mp_full = pd.concat([mp_full, mp_data])

    # create plot
    grid = sns.FacetGrid(data=full, col='sample', col_wrap=4, sharex=False, sharey=False, height=2, aspect=3.5)
    plt.subplots_adjust(hspace=1)
    samples = pd.unique(full.loc[:, 'sample'])
    for ind, ax in enumerate(grid.axes.flat):
        sample_name = samples[ind]
        # Assign colors depending on validity of vectors
        if non_valid and sample_name not in non_valid:
            bgcol, vcol, mpcol = 'xkcd:beige', 'green', 'brown'
        else:
            bgcol, vcol, mpcol = 'xkcd:tan', None, 'red'
        # Plot data underlying vector
        data = full.loc[full.loc[:, 'sample'] == sample_name, :]
        sns.scatterplot(data=data, x='Position X', y='Position Y', color=bgcol, linewidth=0, ax=ax)
        # plot vector
        vector_data = vectors.loc[vectors.loc[:, 'sample'] == sample_name, :]
        ax.plot(vector_data.X, vector_data.Y, color=vcol)
        # plot MP:
        if Sett.useMP and not mp_full.empty:
            mp_data = mp_full.loc[mp_full.loc[:, 'sample'] == samples[ind], :]
            ax.scatter(x=mp_data.at[0, 'Position X'], y=mp_data.at[0, 'Position Y'], c=mpcol, s=36, zorder=4)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(samples[ind])
    if settings is not None:
        plt.annotate(settings, (5, 5), xycoords='figure pixels')
    grid.savefig(str(savedir.joinpath('Vectors.png')))
    plt.close('all')
    print("Vector plot created in 'Analysis Data/Samples'.")


def violin(plotter, **kws):
    """Plot violins."""
    plotter.g = sns.catplot(x='Sample Group', y='Value', data=plotter.data, row=kws.get('row'), col=kws.get('col'),
                            height=kws.get('height'), aspect=kws.get('aspect'), palette=plotter.handle.palette,
                            kind='violin', sharey=False, saturation=0.5)
    plotter.g.set_xticklabels(rotation=30)
    return plotter.g
