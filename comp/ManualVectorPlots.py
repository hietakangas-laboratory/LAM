# -*- coding: utf-8 -*-
"""
For creating vector plots for all samples in a dataset.

Uses channel and vector data found in the 'Samples'-directory (channel files
created during LAM 'Project').

PLOTS CREATED TO THE "ANALYSIS DATA\SAMPLES"-DIRECTORY

USAGE:
    1. Input path to the analysis directory to variable analysis_dir below.
    2. Give name of a channel to variable base_channel, which will be plotted
       alongside the vector, i.e. to show that vector is on top of cells
    3. Give header row of the channel data into channel_data_header
    4. Run script

Created on Tue May  5 08:20:28 2020

@author: ArtoVi
"""

import pathlib2 as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Path to analysis directory
analysis_dir = pl.Path(r'C:')
base_channel = 'DAPI'
channel_data_header = 2


def main():
    sample_dir = analysis_dir.joinpath('Analysis Data', 'Samples')
    samplepaths = [p for p in sample_dir.iterdir() if p.is_dir()]
    create_vector_plots(analysis_dir, sample_dir, samplepaths)


def get_vector_data(files):
    """Collect vector data based file type."""
    files = list(files)
    if len(files) > 1:
        try:
            file = [p for p in files if p.suffix=='.txt'][0]
        except IndexError:
            file = files[0]
    else:
        file = files[0]
    if file.suffix == '.txt':
        data = pd.read_csv(file, sep="\t", header=None)
        data.columns = ["X", "Y"]
    elif file.suffix == '.csv':
        data = pd.read_csv(file, index_col=False)
    return data

def create_vector_plots(workdir, savedir, sample_dirs):
    """"Create single plot file that contains vectors of all found samples."""
    full = pd.DataFrame()
    vectors = pd.DataFrame()
    cols = ['Position X', 'Position Y']
    for path in sample_dirs:
        print(path.name)
        cpath = workdir.joinpath(path.name).glob(f'*_{base_channel}_*')
        cpath = [p for p in cpath if p.is_dir()][0]
        try:
            dpath = cpath.glob('*Position.csv')
            data = pd.read_csv(next(dpath), header=channel_data_header)
            data = data.loc[:, cols].assign(sample=path.name)
        except StopIteration:
            data = pd.DataFrame(data=[np.nan, np.nan], columns=cols)
        try:
            files = path.glob('Vector.*')
            vector = get_vector_data(files)
            vector = vector.assign(sample=path.name)
        except (FileNotFoundError, StopIteration):
            vector = pd.DataFrame(data=[np.nan, np.nan], columns=['X', 'Y'])
        full = pd.concat([full, data])
        vectors = pd.concat([vectors, vector])
    grid = sns.FacetGrid(data=full, col='sample', col_wrap=4, sharex=False,
                         sharey=False, height=2, aspect=3.5)
    plt.subplots_adjust(hspace=1)
    samples = pd.unique(full.loc[:, 'sample'])
    for ind, ax in enumerate(grid.axes.flat):
        data = full.loc[full.loc[:, 'sample'] == samples[ind], :]
        sns.scatterplot(data=data, x='Position X', y='Position Y',
                        color='xkcd:tan', linewidth=0, ax=ax)
        vector_data = vectors.loc[vectors.loc[:, 'sample'] == samples[ind], :]
        ax.plot(vector_data.X, vector_data.Y)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(samples[ind])
    grid.savefig(str(savedir.joinpath('Vectors.png')))
    print(f"\nSave location: {str(savedir.joinpath('Vectors.png'))}")
    plt.close('all')


if __name__== '__main__':
    print('Creating vector plot')
    main()
    print('----------\nDONE')