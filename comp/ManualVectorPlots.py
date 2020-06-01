# -*- coding: utf-8 -*-
"""
For creating vector plots for all samples in a dataset.

Uses channel and vector data found in the 'Samples'-directory (channel files
created during LAM 'Project').

PLOTS CREATED TO THE "ANALYSIS DATA\SAMPLES"-DIRECTORY

Created on Tue May  5 08:20:28 2020

@author: ArtoVi
"""

import pathlib2 as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Path to analysis directory
analysis_dir = pl.Path(r'E:\Code_folder\Josef_Indexerror\MARCM3 Statistics')
base_channel = 'DAPI'


def main():
    sample_dir = analysis_dir.joinpath('Analysis Data', 'Samples')
    samplepaths = [p for p in sample_dir.iterdir() if p.is_dir()]
    for sample in samplepaths:
        print(f'{sample.name}  ...')
        base_data = get_chan_data(sample, base_channel)
        vector_data = get_vector_data(sample)
        if base_data is not None and vector_data is not None:
            make_plots(sample, base_data, vector_data)


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

def get_vector_data(dir_path):
    """Collect data from a sample based on channel name."""
    for file in dir_path.iterdir():
        if 'vector.' in file.name.lower():
            if file.suffix == '.txt':
                data = pd.read_csv(file, sep="\t", header=None)
                data.columns = ["X", "Y"]
            elif file.suffix == '.csv':
                data = pd.read_csv(file, index_col=False)
            return data
    print('--> No vector file found.')
    return None


def make_plots(sample_path, base_data, vector_data):
    """Create a plot of all channels of a sample."""
    # create canvas
    g = sns.FacetGrid(data=base_data, height=3, aspect=3)
    # plot the base channel to show outline of midgut
    g = g.map(sns.scatterplot, data=base_data, x='Position X', y='Position Y',
              color='xkcd:tan', linewidth=0, s=10, legend=False)
    g = g.map(sns.lineplot, data=vector_data, x='X', y='Y',
              color='cyan', linewidth=1)
    g.axes.flat[0].set_aspect('equal')
    # title
    g.fig.suptitle(sample_path.name)

    # save figure
    savepath = sample_path.parent.joinpath(f'{sample_path.name}_vector.pdf')
    g.savefig(savepath, format='pdf', adjustable='box')
    plt.close()


if __name__== '__main__':
    main()
    print('----------\nDONE')