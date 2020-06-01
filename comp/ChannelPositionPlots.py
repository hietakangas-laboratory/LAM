# -*- coding: utf-8 -*-
"""
For creating positional plots of all channels for all samples in a dataset.

Uses channel data found in the 'Samples'-directory (created during LAM
'Project').

PLOTS CREATED TO THE "ANALYSIS DATA\SAMPLES"-DIRECTORY

Created on Tue May  5 08:20:28 2020

@author: ArtoVi
"""

import pathlib2 as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

analysis_dir = pl.Path(r'E:\Code_folder\Josef_Indexerror\MARCM3 Statistics')
base_channel = 'DAPI'
channel_names = ['DAPI', 'GFP']


def main():
    sample_dir = analysis_dir.joinpath('Analysis Data', 'Samples')
    samplepaths = [p for p in sample_dir.iterdir() if p.is_dir()]
    colors = assign_colors(channel_names)
    for sample in samplepaths:
        print(f'{sample.name}  ...')
        base_data = get_chan_data(sample, base_channel)
        make_plots(channel_names, sample, base_data, colors)


def assign_colors(channel_names):
    """Assign colors for each channel."""
    colors = {}
    palette = sns.color_palette('colorblind', n_colors=len(channel_names))
    for ind, name in enumerate(channel_names):
        colors.update({name: palette[ind]})
    return colors


def get_chan_data(dir_path, channel_name):
    """Collect data from a sample based on channel name."""
    filepath = dir_path.joinpath(f'{channel_name}.csv')
    try:
        data = pd.read_csv(filepath, index_col=False)
        pos_data = data.loc[:, ['Position X', 'Position Y']]
    except FileNotFoundError:
        print(f'-> File {filepath.name} not found.')
        return None
    return pos_data


def make_plots(channel_names, sample_path, base_data, colors):
    """Create a plot of all channels of a sample."""
    full_data = pd.DataFrame()
    for channel in channel_names:
        data = get_chan_data(sample_path, channel)
        if data is None:
            continue
            # vals = {'Position X': np.nan, 'Position Y': np.nan,
            #         'channel': channel}
            # data = pd.DataFrame(data=vals)
        data.loc[:, 'channel'] = channel
        full_data = pd.concat([full_data, data])
    
    # create canvas
    g = sns.FacetGrid(data=full_data, row='channel', hue='channel', height=3,
                      aspect=3)
    # plot the base channel to show outline of midgut
    g = g.map(sns.scatterplot, data=base_data, x='Position X', y='Position Y',
              color='xkcd:tan', linewidth=0, s=10, legend=False)
    grouped_data = full_data.groupby('channel')
    # plot feature positions of the channels of interest
    num = 0
    for group, data in grouped_data:
        sns.scatterplot(data=data, x='Position X', y='Position Y', s=7,
                        linewidth=0.002, hue='channel', ax=g.axes.flat[num],
                        legend=False, palette=colors)
        g.axes.flat[num].set_aspect('equal')
        num += 1
    # title
    g.fig.suptitle(sample_path.name)

    # save figure
    savepath = sample_path.parent.joinpath(f'{sample_path.name}_positions.pdf')
    g.savefig(savepath, format='pdf', adjustable='box')
    plt.close()


if __name__== '__main__':
    main()
    print('----------\nDONE')